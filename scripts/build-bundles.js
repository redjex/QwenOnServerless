// Concatenates weight blocks into ~BUNDLE-byte files for Telegram storage.
// Development aid, never deployed. Usage: node scripts/build-bundles.js [bytes]
import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../models/", import.meta.url));
const packed = path.join(root, "packed"), pruned = path.join(root, "pruned");
const out = path.join(root, "bundles");
await fs.mkdir(out, { recursive: true });
await fs.mkdir(path.join(out, "meta"), { recursive: true });

const manifest = JSON.parse(await fs.readFile(path.join(pruned, "manifest.json"), "utf8"));
const BUNDLE = Number(process.argv[2] ?? 14 * 1024 * 1024);
const fileFor = name => path.join(name.startsWith("embed.") ? pruned : packed, `${name}.bin`);

const index = {};
const uploads = [];
let current = [], currentBytes = 0, bundleId = 0;

async function flush() {
  if (!current.length) return;
  const name = `bundle-${bundleId}`;
  const file = path.join(out, `${name}.bin`);
  const handle = await fs.open(file, "w");
  let at = 0;
  for (const { block, bytes } of current) {
    await handle.write(bytes, 0, bytes.length, at);
    index[block] = { bundle: name, at, length: bytes.length };
    at += bytes.length;
  }
  await handle.close();
  uploads.push({ kind: "bundle", name, path: file,
    sha256: createHash("sha256").update(await fs.readFile(file)).digest("hex") });
  bundleId++; current = []; currentBytes = 0;
}

for (const [block, info] of Object.entries(manifest.blocks)) {
  if (info.bytes > BUNDLE) throw new Error(`block ${block} is larger than a bundle`);
  if (currentBytes && currentBytes + info.bytes > BUNDLE) await flush();
  current.push({ block, bytes: new Uint8Array(await fs.readFile(fileFor(block))) });
  currentBytes += info.bytes;
}
await flush();

const indexBytes = Buffer.from(JSON.stringify(index));
await fs.writeFile(path.join(out, "index.json"), indexBytes);
await fs.writeFile(path.join(out, "meta/index.json"), indexBytes);
await fs.writeFile(path.join(out, "meta/uploads.json"), JSON.stringify([{
  kind: "meta", name: "bundleindex", path: path.join(out, "meta/index.json"),
  sha256: createHash("sha256").update(indexBytes).digest("hex"),
}], null, 2));
await fs.writeFile(path.join(out, "uploads.json"), JSON.stringify(uploads, null, 2));

let total = 0;
for (const item of uploads) total += (await fs.stat(item.path)).size;
console.log(JSON.stringify({
  bundles: bundleId, blocks: Object.keys(index).length,
  totalMiB: +(total / 1048576).toFixed(1),
  indexKiB: +(indexBytes.length / 1024).toFixed(1),
}, null, 2));