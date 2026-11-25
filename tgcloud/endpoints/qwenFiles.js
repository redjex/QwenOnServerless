// CLI-only. Moves packed weights to Telegram file storage and measures reads.
// Excluded from deployment: it is run with `tgcloud run` during maintenance.
import { api, db, InputFile } from "sdk";
import { decode64 } from "../lib/qwen-bytes.js";
import { loadMeta, loadBlock } from "../lib/qwen-store.js";
import records from "./qwenStaging.js";

async function adminChat() {
  const row = await db.get("SELECT value FROM qwen_meta WHERE id = 'admin_chat'");
  if (!row) throw new Error("admin_chat is not set; run {action:'setChat', chat:<id>} first");
  return Number(row.value);
}

async function put(id, value) {
  await db.run(
    "INSERT INTO qwen_meta (id, value) VALUES (:id, :value) ON CONFLICT(id) DO UPDATE SET value = excluded.value",
    { ":id": id, ":value": value });
}

function decode(record) {
  const bytes = new Uint8Array(record.length);
  let offset = 0;
  for (const part of record.parts) {
    const chunk = decode64(part);
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (offset !== record.length) throw new Error("Upload length mismatch");
  return bytes;
}

export default async function ({ action = "status", chat, name, bundleBytes, from, count } = {}) {
  if (action === "setChat") {
    await put("admin_chat", String(chat));
    return { chat };
  }
  if (action === "upload") {
    const to = await adminChat();
    const saved = [];
    for (const record of records) {
      const bytes = decode(record);
      const sent = await api.sendDocument({
        chat_id: to,
        document: new InputFile(bytes, `${record.name.replaceAll("/", "_")}.bin`,
          { type: "application/octet-stream" }),
      });
      await put(`file:${record.name}`, JSON.stringify({ file_id: sent.document.file_id, bytes: bytes.length }));
      saved.push({ name: record.name, bytes: bytes.length });
    }
    return saved;
  }
  if (action === "fetch") {
    const names = name ? [name]
      : (await db.all("SELECT id FROM qwen_meta WHERE id LIKE 'file:%'")).map(row => row.id.slice(5));
    const out = [];
    for (const entry of names) {
      const row = await db.get("SELECT value FROM qwen_meta WHERE id = :id", { ":id": `file:${entry}` });
      if (!row) { out.push({ name: entry, missing: true }); continue; }
      const info = JSON.parse(row.value);
      const started = Date.now();
      const bytes = await api.getFileContent(info.file_id);
      out.push({ name: entry, bytes: bytes.length, ms: Date.now() - started });
    }
    return out;
  }
  if (action === "pack") {
    // Build bundles inside the invocation from SQLite and send them to Telegram.
    // This avoids the CLI's per-call payload limit (large staged modules 500).
    const manifest = await loadMeta("manifest");
    const limit = Number(bundleBytes ?? 14 * 1024 * 1024);
    const names = Object.keys(manifest.blocks);
    const groups = [];
    let group = [], bytes = 0;
    for (const name of names) {
      const size = manifest.blocks[name].bytes;
      if (group.length && bytes + size > limit) { groups.push(group); group = []; bytes = 0; }
      group.push(name); bytes += size;
    }
    if (group.length) groups.push(group);
    const start = Number(from ?? 0), take = Number(count ?? 1);
    const to = await adminChat();
    const saved = [];
    for (let g = start; g < Math.min(groups.length, start + take); g++) {
      const payloads = [];
      let total = 0;
      for (const name of groups[g]) { const data = await loadBlock(name); payloads.push({ name, data }); total += data.length; }
      const bundle = new Uint8Array(total);
      const blocks = [];
      let at = 0;
      for (const { name, data } of payloads) {
        bundle.set(data, at);
        blocks.push({ name, at, length: data.length });
        at += data.length;
      }
      const sent = await api.sendDocument({
        chat_id: to,
        document: new InputFile(bundle, `qwen-bundle-${g}.bin`, { type: "application/octet-stream" }),
      });
      await put(`file:bundle-${g}`, JSON.stringify({ file_id: sent.document.file_id, bytes: total, blocks }));
      saved.push({ bundle: g, bytes: total });
    }
    return { totalBundles: groups.length, saved };
  }
  if (action === "dropSqlite") {
    const rows = await db.all("SELECT value FROM qwen_meta WHERE id LIKE 'file:bundle-%'");
    const names = new Set();
    for (const { value } of rows)
      for (const block of JSON.parse(value).blocks) names.add(block.name);
    let removedChunks = 0, removedMeta = 0;
    for (const name of names) {
      const chunks = await db.run("DELETE FROM qwen_blocks WHERE id LIKE :like",
        { ":like": `block:${name}/%` });
      removedChunks += chunks.rowsAffected;
      const meta = await db.run("DELETE FROM qwen_meta WHERE id = :id", { ":id": `block:${name}` });
      removedMeta += meta.rowsAffected;
    }
    return { blocks: names.size, removedChunks, removedMeta };
  }
  if (action === "dropFiles") {
    const result = await db.run("DELETE FROM qwen_meta WHERE id LIKE 'file:%'");
    return { removed: result.rowsAffected };
  }
  const files = await db.all("SELECT id FROM qwen_meta WHERE id LIKE 'file:%'");
  const chatRow = await db.get("SELECT value FROM qwen_meta WHERE id = 'admin_chat'");
  return { files: files.length, chat: chatRow ? chatRow.value : null };
}