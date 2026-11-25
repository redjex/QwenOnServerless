import { api, db } from "sdk";
import { generationStep } from "./qwen-job.js";
import { deliverAnswer } from "./qwen-delivery.js";
import { quotedText } from "./qwen-answer.js";

// Bot-to-bot continuation without a VPS: the model bot posts a tick into a private
// group, a second bot echoes it, and the echo arrives as an update that advances the
// answer. Bots never receive their own messages, so the ping-pong is what keeps the
// chain alive. Everything stays inside Telegram Serverless.
const TICK = "#qwen-tick ";
const KEY = "relay_group";

export async function relayGroup() {
  const row = await db.get("SELECT value FROM qwen_meta WHERE id = :id", { ":id": KEY });
  return row ? Number(row.value) : null;
}

async function storeGroup(id) {
  await db.run(
    "INSERT INTO qwen_meta (id, value) VALUES (:id, :v) ON CONFLICT(id) DO UPDATE SET value = :v",
    { ":id": KEY, ":v": String(id) });
}

async function forgetGroup() {
  await db.run("DELETE FROM qwen_meta WHERE id = :id", { ":id": KEY });
}

// Handle any non-private message: enable/disable the relay, or process a tick.
export async function handleGroupMessage(message) {
  if (!message.from || typeof message.text !== "string") return;
  const text = message.text.trim();
  const command = text.split(/\s|@/)[0].toLowerCase();
  if (command === "/relay") {
    await storeGroup(message.chat.id);
    return api.sendMessage({
      chat_id: message.chat.id,
      ...quotedText("Ретранслятор включён. Ответы в личке будут продолжаться через этот чат."),
    });
  }
  if (command === "/relay_off") {
    await forgetGroup();
    return api.sendMessage({ chat_id: message.chat.id, ...quotedText("Ретранслятор выключен.") });
  }
  // Only another bot's tick drives continuation.
  if (!message.from.is_bot || !text.startsWith(TICK)) return;
  const owner = text.slice(TICK.length).trim();
  if (!/^user:\d+$/.test(owner)) return;
  const row = await db.get("SELECT state FROM qwen_generations WHERE id = :id", { ":id": owner });
  if (!row) return;
  let state;
  try { state = JSON.parse(row.state); } catch { return; }
  if (state.finished || !state.requestId) return;
  try {
    const next = await generationStep(owner, {
      requestId: state.requestId,
      publish: value => deliverAnswer(Number(owner.slice("user:".length)), value),
    });
    await scheduleRelay(owner, next);
  } catch (error) {
    // A busy or killed step stops this chain; the next user message restarts it.
    console.warn("Relay step failed:", error.description || error.message);
  }
}

// Ask the second bot for the next ping when an answer is still in progress.
export async function scheduleRelay(owner, state) {
  if (!state || state.finished) return false;
  const group = await relayGroup();
  if (group === null) return false;
  await api.sendMessage({ chat_id: group, text: `${TICK}${owner}`, disable_notification: true });
  return true;
}