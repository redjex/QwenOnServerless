// Development aid: compare int8-embedding (pruned) and int4-embedding (pruned4)
// models on the same prompts, and report logit drift on the first token.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createTokenizer } from "../tgcloud/lib/qwen-tokenizer.js";
import { createEngine } from "../tgcloud/lib/qwen-engine.js";
import { createQwen, pickToken } from "../tgcloud/lib/qwen-model.js";

const root = fileURLToPath(new URL("../models/", import.meta.url));

async function load(dir) {
  const manifest = JSON.parse(await fs.readFile(path.join(root, dir, "manifest.json"), "utf8"));
  const data = JSON.parse(await fs.readFile(path.join(root, dir, "tokenizer.json"), "utf8"));
  const readBytes = name => fs.readFile(path.join(name.startsWith("embed.") ? path.join(root, dir) : path.join(root, "packed"), `${name}.bin`));
  const engine = await createEngine({ manifest, readBytes });
  return { engine, tokenizer: createTokenizer(data) };
}

const variants = { int8: await load("pruned"), int4: await load("pruned4") };
const prompts = ["Привет! Как дела?", "Меня зовут Иван. Запомни это."];

for (const prompt of prompts) {
  const out = {};
  let firstLogits = {};
  for (const [name, { engine, tokenizer }] of Object.entries(variants)) {
    const input = tokenizer.chat(prompt);
    const model = await createQwen(engine, { capacity: 256 });
    const generated = [];
    let pending = input;
    for (let i = 0; i < 20; i++) {
      const logits = await model.forward(pending);
      if (i === 0) firstLogits[name] = logits;
      const token = pickToken(logits, generated);
      if (tokenizer.eos.has(token)) break;
      generated.push(token);
      pending = [token];
    }
    out[name] = { tokens: generated.length, text: tokenizer.decode(generated), bytesMiB: +(engine.store.bytesRead / 1024 / 1024).toFixed(1) };
  }
  let drift = 0, same = true;
  const a = firstLogits.int8, b = firstLogits.int4;
  for (let i = 0; i < a.length; i++) drift = Math.max(drift, Math.abs(a[i] - b[i]));
  const topA = pickToken(a), topB = pickToken(b);
  same = topA === topB;
  console.log(JSON.stringify({ prompt, sameFirstToken: same, maxLogitDrift: +drift.toFixed(3), ...out }, null, 2));
}