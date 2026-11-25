// Local benchmark for the packed model. Development aid, never deployed.
// Usage: node scripts/bench-qwen.js [tokens] [prompt]
import fs from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createTokenizer } from "../tgcloud/lib/qwen-tokenizer.js";
import { pickToken } from "../tgcloud/lib/qwen-model.js";

const root = new URL("../models/", import.meta.url);
const tokens = Number(process.argv[2] ?? 8);
const prompt = process.argv[3] ?? "Привет!";

const manifest = JSON.parse(await fs.readFile(new URL("packed/manifest.json", root), "utf8"));
const data = JSON.parse(await fs.readFile(new URL("packed/tokenizer.json", root), "utf8"));
const tokenizer = createTokenizer(data);
const input = tokenizer.chat(prompt);
tokenizer.releaseEncoder();

const readBytes = async name => new Uint8Array(await fs.readFile(new URL(`packed/${name}.bin`, root)));

const { createEngine } = await import("../tgcloud/lib/qwen-engine.js");
const { createQwen } = await import("../tgcloud/lib/qwen-model.js");
const started = performance.now();
const engine = await createEngine({ manifest, readBytes });
const built = performance.now() - started;
const model = await createQwen(engine, { capacity: 128 });
const generated = [];
let pending = input;
const marks = [];
for (let i = 0; i < tokens; i++) {
  const at = performance.now();
  const logits = await model.forward(pending);
  const token = pickToken(logits, generated);
  if (tokenizer.eos.has(token)) break;
  generated.push(token);
  pending = [token];
  marks.push(performance.now() - at);
}
const total = performance.now() - started;
const peakRss = process.memoryUsage().rss / 1024 / 1024;
console.log(JSON.stringify({
  prompt, inputTokens: input.length, outputTokens: generated.length,
  buildMs: Math.round(built), totalMs: Math.round(total),
  perTokenMs: marks.map(value => Math.round(value)),
  mbPerSecond: Math.round(engine.store.bytesRead / 1024 / 1024 / (total / 1000)),
  bytesReadMiB: +(engine.store.bytesRead / 1024 / 1024).toFixed(1),
  peakRssMiB: Math.round(peakRss),
  text: tokenizer.decode(generated),
}, null, 2));
void fileURLToPath;
