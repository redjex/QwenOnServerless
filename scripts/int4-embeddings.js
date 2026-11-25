// Re-quantizes the pruned int8 embeddings to int4 (group size 32), the same format
// prepare-qwen.py writes for transformer blocks. Development aid, never deployed.
import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../models/", import.meta.url));
const source = path.join(root, "pruned");
const out = path.join(root, "pruned4");
await fs.mkdir(out, { recursive: true });

const manifest = JSON.parse(await fs.readFile(path.join(source, "manifest.json"), "utf8"));
const uploads = [];
let total = 0, blocks = 0;

const names = Object.keys(manifest.blocks)
  .filter(name => name.startsWith("embed."))
  .sort((a, b) => Number(a.split(".")[1]) - Number(b.split(".")[1]));

for (const name of names) {
  const bytes = new Uint8Array(await fs.readFile(path.join(source, `${name}.bin`)));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rows = view.getUint32(0, true), cols = view.getUint32(4, true);
  const rowScales = new Float32Array(bytes.buffer, bytes.byteOffset + 8, rows);
  const weights = new Int8Array(bytes.buffer, bytes.byteOffset + 8 + rows * 4, rows * cols);
  const groups = rows * cols / 32;
  const scales = new Float32Array(groups);
  const packed = new Uint8Array(rows * cols / 2);
  for (let r = 0; r < rows; r++) {
    const rowScale = rowScales[r];
    for (let g = 0; g < cols / 32; g++) {
      let maxabs = 0;
      for (let j = 0; j < 32; j++) {
        const v = Math.abs(weights[r * cols + g * 32 + j] * rowScale);
        if (v > maxabs) maxabs = v;
      }
      const scale = Math.max(maxabs / 7, 1e-12);
      scales[r * (cols / 32) + g] = scale;
      for (let j = 0; j < 32; j++) {
        let q = Math.round(weights[r * cols + g * 32 + j] * rowScale / scale);
        if (q > 7) q = 7; else if (q < -7) q = -7;
        const nibble = q + 8;
        const index = r * cols + g * 32 + j;
        const at = index >> 1;
        // Even indices in the low nibble, odd in the high nibble (matches the kernel).
        if ((index & 1) === 0) packed[at] = (packed[at] & 0xf0) | nibble;
        else packed[at] = (packed[at] & 0x0f) | (nibble << 4);
      }
    }
  }
  const data = new Uint8Array(12 + groups * 4 + packed.length);
  const header = new DataView(data.buffer);
  header.setUint32(0, rows, true);
  header.setUint32(4, cols, true);
  header.setUint32(8, 4, true);
  data.set(new Uint8Array(scales.buffer), 12);
  data.set(packed, 12 + groups * 4);
  const file = path.join(out, `${name}.bin`);
  await fs.writeFile(file, data);
  manifest.blocks[name] = { rows, cols, bytes: data.length };
  total += data.length; blocks++;
  uploads.push({ kind: "block", name, path: file, sha256: createHash("sha256").update(data).digest("hex") });
}

const manifestBytes = Buffer.from(JSON.stringify(manifest));
await fs.writeFile(path.join(out, "manifest.json"), manifestBytes);
uploads.push({ kind: "meta", name: "manifest", path: path.join(out, "manifest.json"),
  sha256: createHash("sha256").update(manifestBytes).digest("hex") });
await fs.copyFile(path.join(source, "tokenizer.json"), path.join(out, "tokenizer.json"));
const tokenizerBytes = await fs.readFile(path.join(out, "tokenizer.json"));
uploads.push({ kind: "meta", name: "tokenizer", path: path.join(out, "tokenizer.json"),
  sha256: createHash("sha256").update(tokenizerBytes).digest("hex") });
await fs.writeFile(path.join(out, "uploads.json"), JSON.stringify(uploads, null, 2));
await fs.writeFile(path.join(out, "plan.json"), JSON.stringify({
  blocks, embeddingMiB: +(total / 1024 / 1024).toFixed(1),
  vocab: manifest.config.vocab_size,
}, null, 2));
console.log(JSON.stringify({ blocks, embeddingMiB: +(total / 1024 / 1024).toFixed(1) }, null, 2));