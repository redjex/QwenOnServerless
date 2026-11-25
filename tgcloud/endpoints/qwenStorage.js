// Administrative CLI utility, excluded from deployment.
import { db } from "sdk";
import { loadMeta } from "../lib/qwen-store.js";

export default async function ({ action = "stats", owner, milliseconds } = {}) {
  if (action === "jobText") {
    const row = await db.get("SELECT state FROM qwen_generations WHERE id = :id", { ":id": owner });
    if (!row) return { exists: false };
    const state = JSON.parse(row.state);
    return { prompt: state.prompt, text: state.text, outputTokens: state.outputTokens,
      finished: state.finished, lengthLimit: state.lengthLimit, display: state.displayMessageId };
  }
  if (action === "jobs") {
    const rows = await db.all(
      "SELECT id, revision, busy_until, auto_enabled, next_run_at, failure_count, state FROM qwen_generations");
    const now = Date.now();
    return rows.map(row => {
      let state = {};
      try { state = JSON.parse(row.state); } catch { state = {}; }
      return {
        id: row.id, revision: row.revision, auto: row.auto_enabled,
        busyInSec: Math.round((row.busy_until - now) / 1000),
        nextInSec: Math.round((row.next_run_at - now) / 1000),
        failures: row.failure_count, finished: state.finished,
        outputTokens: state.outputTokens, display: state.displayMessageId,
        updatedAgoSec: state.updatedAt ? Math.round((now - state.updatedAt) / 1000) : null,
        text: typeof state.text === "string" ? state.text.slice(0, 60) : null,
      };
    });
  }
  if (action === "setBusy") {
    await db.run("UPDATE qwen_generations SET busy_until = :until WHERE id = :id",
      { ":id": owner, ":until": Date.now() + Number(milliseconds ?? 60000) });
    return { ok: true, owner, milliseconds: Number(milliseconds ?? 60000) };
  }
  if (action === "resetState") {
    const generations = await db.run("DELETE FROM qwen_generations");
    const kv = await db.run("DELETE FROM qwen_kv");
    const history = await db.run("DELETE FROM qwen_history");
    return { generations: generations.rowsAffected, kv: kv.rowsAffected, history: history.rowsAffected };
  }
  if (action === "dropStaleEmbed") {
    // Remove embedding blocks left over from a larger vocabulary so they do not
    // keep occupying the database quota.
    const manifest = await loadMeta("manifest");
    const valid = new Set(Object.keys(manifest.blocks)
      .filter(name => name.startsWith("embed.")).map(name => `block:${name}`));
    const blocks = await db.all("SELECT id FROM qwen_blocks WHERE id LIKE 'block:embed.%'");
    const stale = new Set();
    for (const { id } of blocks) {
      const key = id.slice(0, id.lastIndexOf("/"));
      if (!valid.has(key)) stale.add(key);
    }
    let removedBlocks = 0;
    for (const key of stale) {
      const result = await db.run("DELETE FROM qwen_blocks WHERE id LIKE :like", { ":like": `${key}/%` });
      removedBlocks += result.rowsAffected;
    }
    const metas = await db.all("SELECT id FROM qwen_meta WHERE id LIKE 'block:embed.%'");
    let removedMeta = 0;
    for (const { id } of metas)
      if (!valid.has(id)) { await db.run("DELETE FROM qwen_meta WHERE id = :id", { ":id": id }); removedMeta++; }
    return { valid: valid.size, staleBundles: stale.size, removedBlocks, removedMeta };
  }
  if (action === "owners") {
    const models = await db.all("SELECT user_id FROM tiny_text_models");
    const generations = await db.all("SELECT id FROM qwen_generations");
    return { models, generations };
  }
  if (action === "compact") {
    // Delete any weight chunk that the current manifest or asset descriptor does
    // not reference. Upserts leave the tail of a shrunk block behind.
    const CHUNK = 256 * 1024;
    const expected = new Map();
    const manifest = await loadMeta("manifest");
    for (const [name, info] of Object.entries(manifest.blocks))
      expected.set(`block:${name}`, Math.ceil(info.bytes / CHUNK));
    const metas = await db.all("SELECT id, value FROM qwen_meta WHERE id LIKE 'meta:%'");
    for (const { id, value } of metas) {
      // Metadata payload chunks are stored under the same key as the descriptor:
      // descriptor id "meta:tokenizer", payload ids "meta:tokenizer/0", ...
      expected.set(id, JSON.parse(value).parts);
    }
    const rows = await db.all("SELECT id FROM qwen_blocks");
    const remove = [];
    for (const { id } of rows) {
      const slash = id.lastIndexOf("/");
      const key = id.slice(0, slash), part = Number(id.slice(slash + 1));
      const parts = expected.get(key);
      if (parts === undefined || !Number.isInteger(part) || part < 0 || part >= parts) remove.push(id);
    }
    for (const id of remove) await db.run("DELETE FROM qwen_blocks WHERE id = :id", { ":id": id });
    return { expectedAssets: expected.size, removedChunks: remove.length };
  }
  if (action === "removeInt8Layers") {
    const chunks = await db.run("DELETE FROM qwen_blocks WHERE id LIKE 'block:model.layers.%'");
    await db.run("DELETE FROM qwen_meta WHERE id LIKE 'block:model.layers.%'");
    return { removedChunks: chunks.rowsAffected };
  }
  const weights = await db.get("SELECT count(*) AS chunks, sum(length(payload)) AS bytes FROM qwen_blocks");
  const descriptors = await db.get("SELECT count(*) AS assets FROM qwen_meta WHERE id LIKE 'block:%'");
  return { weights, descriptors };
}