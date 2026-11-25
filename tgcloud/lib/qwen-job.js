import { db } from "sdk";
import { loadMeta } from "./qwen-store.js";
import { runtime } from "./qwen-runtime.js";
import { createQwen, pickToken } from "./qwen-model.js";
import { createTokenizer } from "./qwen-tokenizer.js";
import { saveKv, restoreKv, removeKv } from "./qwen-checkpoint.js";

// Budget is measured in "token equivalents" of continuous compute. The platform
// allows roughly 34 per invocation (with run-to-run variance); a prefill token is
// cheaper than a generated one because it skips the output projection. Deeper
// context is pre-filled across several invocations, driven by the VPS worker.
const EQUIVALENTS = 34;
const PREFILL_COST = 0.78;
const MAX_OUTPUT = 64;
const INPUT_CAP = 160;
const PUBLISH_EVERY = 6;
const HISTORY_MESSAGES = 8;

// Lease held by one invocation; a real step finishes well within it. A lease that
// started longer ago than STALE_MS is assumed abandoned (CPU-killed) and reclaimed.
export const LEASE_MS = 20000;
export const STALE_MS = 15000;
const ACQUIRE = "UPDATE qwen_generations SET revision = revision + 1, busy_until = :until WHERE id = :id AND busy_until <= :now RETURNING revision";
const ACQUIRE_FORCE = "UPDATE qwen_generations SET revision = revision + 1, busy_until = :until WHERE id = :id RETURNING revision";

async function loadHistory(owner) {
  const row = await db.get("SELECT messages FROM qwen_history WHERE id = :id", { ":id": owner });
  return row ? JSON.parse(row.messages) : [];
}

async function saveHistory(owner, messages) {
  let list = messages.slice(-HISTORY_MESSAGES);
  while (list.length && list[0].role !== "user") list = list.slice(1);
  await db.run(
    "INSERT INTO qwen_history (id, messages) VALUES (:id, :m) ON CONFLICT(id) DO UPDATE SET messages = :m",
    { ":id": owner, ":m": JSON.stringify(list) });
}

const RUN = /([^\s])\1{3,}/gu;
// Characters that legitimately repeat in Markdown/code dividers and lists.
const DIVIDER = new Set(["-", "=", "*", "_", "#", "~", "|"]);
// A degenerate answer must not become conversation context: the model happily
// continues its own garbage (a bad answer was replayed verbatim by the next
// reply). Cheap, conservative signatures only: bad decode, a run of one
// character, or several tokens that glue two scripts together.
export function looksDegenerate(text) {
  if (typeof text !== "string" || !text) return false;
  if (text.includes("\uFFFD")) return true;
  for (const match of text.matchAll(RUN)) if (!DIVIDER.has(match[1])) return true;
  let mixed = 0;
  for (const token of text.split(/\s+/))
    if (/[A-Za-z]/.test(token) && /[\u0400-\u04FF]/.test(token) && ++mixed >= 3) return true;
  return false;
}

// Keep the newest turns that still fit the input cap, always starting with a user turn.
function fitTranscript(messages, tokenizer, cap) {
  let list = messages.slice();
  let ids = tokenizer.chatMessages(list);
  while (ids.length > cap && list.length > 1) {
    list = list.slice(1);
    if (list.length && list[0].role === "assistant") list = list.slice(1);
    if (!list.length) break;
    ids = tokenizer.chatMessages(list);
  }
  return { messages: list, ids };
}

export async function generationStep(owner, { prompt, requestId, displayMessageId, automatic = false, publish, budget: budgetOverride } = {}) {
  await db.run("INSERT INTO qwen_generations (id, state, revision, busy_until) VALUES (:id, '{}', 0, 0) ON CONFLICT(id) DO NOTHING", { ":id": owner });
  const isNew = prompt !== undefined;
  const now = Date.now();
  const until = now + LEASE_MS;
  let locked = await db.run(ACQUIRE,
    { ":id": owner, ":until": until, ":now": now });
  if (locked.rowsAffected !== 1) {
    // A new question always supersedes the previous answer. A lease left behind by
    // a CPU-killed invocation is also safe to take over once it is clearly stale.
    const current = isNew ? null
      : await db.get("SELECT busy_until FROM qwen_generations WHERE id = :id", { ":id": owner });
    const stale = current && now - (current.busy_until - LEASE_MS) > STALE_MS;
    if (isNew || stale) locked = await db.run(ACQUIRE_FORCE, { ":id": owner, ":until": until });
  }
  if (locked.rowsAffected !== 1) throw new Error("Предыдущая порция ещё обрабатывается. Подожди немного.");
  const revision = locked.rows[0].revision;
  let published = false;
  try {
    const row = await db.get("SELECT state FROM qwen_generations WHERE id = :id", { ":id": owner });
    let state = JSON.parse(row.state);
    await removeKv(owner, state.checkpoint?.snapshot);
    const tokenizer = createTokenizer(await loadMeta("tokenizer"), { encoder: isNew });
    if (isNew) {
      if (typeof prompt !== "string" || !prompt.trim() || prompt.length > 500)
        throw new Error("Отправь короткий вопрос, от 1 до 500 символов.");
      const history = await loadHistory(owner);
      const fitted = fitTranscript([...history, { role: "user", text: prompt }], tokenizer, INPUT_CAP);
      if (fitted.ids.length > INPUT_CAP)
        throw new Error("Вопрос слишком длинный для контекста. Сократи его.");
      state = {
        prompt, messages: fitted.messages, input: fitted.ids, prefill: 0,
        generated: [], requestId: String(requestId), displayMessageId,
        finished: false, checkpoint: null, historySaved: false,
      };
      tokenizer.releaseEncoder();
      // Persist the initial state before the heavy step, so a CPU-killed first
      // invocation can still be resumed by the worker instead of hanging.
      await db.run(
        "UPDATE qwen_generations SET state = :state, auto_enabled = :enabled, failure_count = 0 WHERE id = :id AND revision = :revision",
        { ":state": JSON.stringify(state), ":enabled": automatic ? 1 : 0, ":id": owner, ":revision": revision });
    } else {
      if (String(requestId) !== state.requestId) throw new Error("Это продолжение предыдущего ответа. Задай вопрос заново.");
      if (displayMessageId !== undefined) state.displayMessageId = displayMessageId;
      // Legacy state from before chunked prefill.
      if (state.prefill === undefined) state.prefill = state.input.length;
      if (state.finished) {
        if (publish) await publish(state);
        return state;
      }
    }
    const engine = await runtime();
    const model = await createQwen(engine, { capacity: 256 });
    if (state.checkpoint) await restoreKv(state.checkpoint, model);

    const equivLimit = budgetOverride ? Infinity : EQUIVALENTS;
    const genCap = budgetOverride ?? MAX_OUTPUT;
    let used = 0, logits = null;

    // Prefill the yet-unseen input tokens, in chunks that fit the CPU budget.
    while (state.prefill < state.input.length) {
      const room = Math.floor((equivLimit - used) / PREFILL_COST);
      if (room <= 0) break;
      const chunk = Math.min(state.input.length - state.prefill, room, 48);
      const last = state.prefill + chunk === state.input.length;
      logits = await model.forward(state.input.slice(state.prefill, state.prefill + chunk), { head: last });
      state.prefill += chunk;
      used += chunk * PREFILL_COST;
      if (!last) { logits = null; break; }
    }

    // Generate while the prefill is complete and the budget allows.
    if (state.prefill >= state.input.length) {
      if (!logits) {
        if (!state.generated.length) throw new Error("Состояние ответа повреждено. Задай вопрос заново.");
        logits = await model.forward([state.generated.at(-1)]);
        used += 1;
      }
      let produced = 0;
      while (logits) {
        if (produced >= genCap) break;
        const token = pickToken(logits, state.generated);
        if (tokenizer.eos.has(token)) { state.finished = true; break; }
        state.generated.push(token);
        produced++;
        if (state.generated.length >= MAX_OUTPUT) {
          state.finished = true;
          state.lengthLimit = true;
          break;
        }
        // Stream partial text so a slow or killed call still shows progress.
        if (publish && state.generated.length % PUBLISH_EVERY === 0) {
          state.text = tokenizer.decode(state.generated);
          state.outputTokens = state.generated.length;
          try { await publish(state); }
          catch (error) { console.warn("Partial publish failed:", error.message); }
        }
        if (used + 1 > equivLimit) { logits = null; break; }
        logits = await model.forward([token]);
        used += 1;
      }
    }

    state.text = tokenizer.decode(state.generated);
    state.outputTokens = state.generated.length;
    state.inputTokens = state.input.length;
    state.prefillTokens = state.prefill;
    state.bytesRead = engine.store.bytesRead;
    state.updatedAt = Date.now();
    if (state.finished && !state.historySaved && state.messages) {
      const turns = [...state.messages];
      if (looksDegenerate(state.text)) console.warn("Degenerate answer kept out of history");
      else turns.push({ role: "assistant", text: state.text });
      await saveHistory(owner, turns);
      state.historySaved = true;
    }
    const snapshot = `${owner}/${revision}`;
    state.checkpoint = state.finished ? null : await saveKv(owner, snapshot, model);
    // Publish under the lease. A failed delivery leaves the previous checkpoint
    // valid, so a retry can regenerate and deliver the same text.
    if (publish) await publish(state);
    const saved = await db.run(
      "UPDATE qwen_generations SET state = :state, auto_enabled = CASE WHEN :finished = 1 THEN 0 ELSE auto_enabled END, next_run_at = :next, failure_count = 0 WHERE id = :id AND revision = :revision",
      { ":id": owner, ":state": JSON.stringify(state), ":revision": revision,
        ":finished": state.finished ? 1 : 0, ":next": Date.now() });
    if (saved.rowsAffected !== 1) throw new Error("Ответ уже изменился, повтори запрос.");
    published = true;
    await removeKv(owner, state.checkpoint?.snapshot);
    return state;
  } finally {
    if (!published) {
      // Remove only incomplete writes from this attempt; preserve the last valid snapshot.
      await db.run("DELETE FROM qwen_kv WHERE owner = :owner AND id LIKE :prefix",
        { ":owner": owner, ":prefix": `${owner}/${revision}/%` });
    }
    await db.run("UPDATE qwen_generations SET busy_until = 0 WHERE id = :id AND revision = :revision",
      { ":id": owner, ":revision": revision });
  }
}

// Continuation is driven by the VPS worker; there is no manual button.
export function answerMarkup() {
  return { inline_keyboard: [] };
}

export function answerText(state) {
  const text = state.text || (state.finished ? "Модель завершила ответ без текста." : "Думаю над ответом...");
  return state.lengthLimit ? `${text}\n\nДостигнут предел длины ответа (64 токена).` : text;
}