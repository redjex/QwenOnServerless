// Local CLI only. Never deploy this administrative endpoint.
import { db } from "sdk";
import { qwenBlocks, qwenMeta } from "../schema.js";
import { decode64 } from "../lib/qwen-bytes.js";
import records from "./qwenStaging.js";

export default async function () {
  const saved = [];
  for (const record of records) {
    const bytes = new Uint8Array(record.length);
    let offset = 0;
    for (const part of record.parts) {
      const chunk = decode64(part);
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    if (offset !== record.length) throw new Error("Upload length mismatch");
    const key = `${record.kind}:${record.name}`;
    const chunkSize = 256 * 1024;
    for (let at = 0, part = 0; at < bytes.length; at += chunkSize, part++) {
      const payload = bytes.slice(at, at + chunkSize);
      await db.insert(qwenBlocks).values({ id: `${key}/${part}`, payload, checksum: record.sha256 })
        .onConflictDoUpdate({ target: qwenBlocks.id, set: { payload, checksum: record.sha256 } }).run();
    }
    const value = JSON.stringify({ bytes: bytes.length, parts: Math.ceil(bytes.length / chunkSize), sha256: record.sha256 });
    await db.insert(qwenMeta).values({ id: key, value })
      .onConflictDoUpdate({ target: qwenMeta.id, set: { value } }).run();
    saved.push({ name: record.name, bytes: bytes.length });
  }
  return saved;
}
