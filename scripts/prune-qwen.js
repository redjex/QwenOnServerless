// Builds a vocabulary-pruned copy of the packed model. Development aid, never deployed.
//
// The output projection is tied to the embeddings, and they are the single largest
// tensor (149 MiB, ~36% of the model) and ~31% of the per-token compute. Most of
// Qwen3's 151 936 rows serve scripts this bot never sees, so we keep the highest
// frequency tokens (BPE merge order is frequency order) plus every single-byte token
// as a fallback, which guarantees any text still encodes. Transformer blocks are
// reused unchanged, so only the embedding blocks, manifest and tokenizer change.
//
// Usage: node scripts/prune-qwen.js [safe|freq] [targetVocab]
//   safe (default) — keep everything Latin/Cyrillic needs; exact original quality.
//   freq <target>   — also trim by merge order (smaller, but re-tokenizes prompts).
import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../models/", import.meta.url));
const COLS = 1024, CHUNK = 4096;

const packed = path.join(root, "packed");
const out = path.join(root, "pruned");
await fs.mkdir(out, { recursive: true });

const manifest = JSON.parse(await fs.readFile(path.join(packed, "manifest.json"), "utf8"));
const tokenizer = JSON.parse(await fs.readFile(path.join(packed, "tokenizer.json"), "utf8"));
const tokens = tokenizer.tokens;

const byteToChar = new Array(256);
const charToByte = new Map();
let extra = 0;
for (let b = 0; b < 256; b++) {
  const visible = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
  const ch = String.fromCodePoint(visible ? b : 256 + extra++);
  byteToChar[b] = ch;
  charToByte.set(ch, b);
}
const bytesOf = token => {
  const out = [];
  for (const ch of token) { const b = charToByte.get(ch); if (b === undefined) return null; out.push(b); }
  return out;
};

const ids = new Map(tokens.map((token, id) => [token, id]));
const kept = new Set();
// "safe" drops only tokens whose bytes can never appear in Latin/Cyrillic text, so
// Russian and English keep their exact original tokenization and quality. "freq"
// additionally trims by BPE merge order (smaller, but re-tokenizes prompts).
const mode = process.argv[2] ?? "safe";
const target = Number(process.argv[3] ?? 49152);
const LATIN = b => b === 0x09 || b === 0x0a || b === 0x0d || (b >= 0x20 && b <= 0x7e);
const CONT = b => b >= 0x80 && b <= 0xbf;
const allowed = b => LATIN(b) || b === 0xc2 || b === 0xc3 || b === 0xd0 || b === 0xd1 || CONT(b) || b === 0xe2;
for (let id = 0; id < tokens.length; id++) {
  const bytes = bytesOf(tokens[id]);
  if (!bytes || bytes.length === 1) { kept.add(id); continue; }  // every byte stays encodable
  if (mode === "safe" ? bytes.every(allowed) : false) kept.add(id);
}
for (const item of tokenizer.added) kept.add(item.id);
if (mode !== "safe")
  for (const pair of tokenizer.merges) {
    if (kept.size >= target) break;
    const id = ids.get(pair[0] + pair[1]);
    if (id !== undefined) kept.add(id);
  }
const keepIds = [...kept].sort((a, b) => a - b);
const remap = new Map(keepIds.map((oldId, newId) => [oldId, newId]));
const dropped = tokens.length - keepIds.length;

// Rebuild the embedding blocks: keep the selected rows, in old-id order so the new
// indexing is simply the row's position.
async function loadEmbed(name) {
  const bytes = new Uint8Array(await fs.readFile(path.join(packed, `${name}.bin`)));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rows = view.getUint32(0, true), cols = view.getUint32(4, true);
  return {
    cols,
    scales: new Float32Array(bytes.buffer, bytes.byteOffset + 8, rows),
    weights: new Int8Array(bytes.buffer, bytes.byteOffset + 8 + rows * 4, rows * cols),
  };
}
const chunkCount = Object.keys(manifest.blocks).filter(name => name.startsWith("embed.")).length;
const source = [];
for (let chunk = 0; chunk < chunkCount; chunk++) source.push(await loadEmbed(`embed.${chunk}`));

const newBlocks = {};
const uploads = [];
let written = 0;
for (let at = 0; at < keepIds.length; at += CHUNK) {
  const slice = keepIds.slice(at, at + CHUNK);
  const rows = slice.length;
  const bytes = new Uint8Array(8 + rows * 4 + rows * COLS);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, rows, true);
  view.setUint32(4, COLS, true);
  const scales = new Float32Array(bytes.buffer, 8, rows);
  const weights = new Int8Array(bytes.buffer, 8 + rows * 4, rows * COLS);
  slice.forEach((oldId, row) => {
    const block = source[Math.floor(oldId / CHUNK)];
    const from = (oldId % CHUNK);
    scales[row] = block.scales[from];
    weights.set(block.weights.subarray(from * COLS, from * COLS + COLS), row * COLS);
  });
  const name = `embed.${written}`;
  const file = path.join(out, `${name}.bin`);
  await fs.writeFile(file, bytes);
  newBlocks[name] = { rows, cols: COLS, bytes: bytes.length };
  uploads.push({ kind: "block", name, path: file, sha256: createHash("sha256").update(bytes).digest("hex") });
  written++;
}

const prunedManifest = {
  ...manifest,
  config: { ...manifest.config, vocab_size: keepIds.length },
  embeddingChunkRows: CHUNK,
  blocks: { ...manifest.blocks, ...newBlocks },
};
for (const name of Object.keys(prunedManifest.blocks))
  if (name.startsWith("embed.") && !newBlocks[name]) delete prunedManifest.blocks[name];
const manifestBytes = Buffer.from(JSON.stringify(prunedManifest));
await fs.writeFile(path.join(out, "manifest.json"), manifestBytes);
uploads.push({ kind: "meta", name: "manifest", path: path.join(out, "manifest.json"),
  sha256: createHash("sha256").update(manifestBytes).digest("hex") });

const prunedTokenizer = {
  tokens: keepIds.map(id => tokens[id]),
  merges: tokenizer.merges.filter(pair => ids.has(pair[0] + pair[1]) && remap.has(ids.get(pair[0] + pair[1]))),
  added: tokenizer.added.map(item => ({ ...item, id: remap.get(item.id) })),
};
const tokenizerBytes = Buffer.from(JSON.stringify(prunedTokenizer));
await fs.writeFile(path.join(out, "tokenizer.json"), tokenizerBytes);
uploads.push({ kind: "meta", name: "tokenizer", path: path.join(out, "tokenizer.json"),
  sha256: createHash("sha256").update(tokenizerBytes).digest("hex") });
await fs.writeFile(path.join(out, "uploads.json"), JSON.stringify(uploads, null, 2));

const embeddingBytes = Object.values(newBlocks).reduce((sum, b) => sum + b.bytes, 0);
await fs.writeFile(path.join(out, "plan.json"), JSON.stringify({
  mode, target, kept: keepIds.length, dropped,
  embeddingBytes, embeddingBlocks: written,
  upload: ["meta:manifest", "meta:tokenizer", ...Object.keys(newBlocks).map(name => `block:${name}`)],
}, null, 2));

console.log(JSON.stringify({
  mode, target, kept: keepIds.length, dropped,
  embeddingMiB: +(embeddingBytes / 1024 / 1024).toFixed(1),
  blocks: written,
}, null, 2));
