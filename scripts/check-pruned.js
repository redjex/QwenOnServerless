// Development aid: run the pruned model locally and print a sample.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createTokenizer } from "../tgcloud/lib/qwen-tokenizer.js";
import { createEngine } from "../tgcloud/lib/qwen-engine.js";
import { createQwen, pickToken } from "../tgcloud/lib/qwen-model.js";

const root = fileURLToPath(new URL("../models/", import.meta.url));
const packed = path.join(root, "packed"), pruned = path.join(root, "pruned");
const manifest = JSON.parse(await fs.readFile(path.join(pruned, "manifest.json"), "utf8"));
const tokenizerData = JSON.parse(await fs.readFile(path.join(pruned, "tokenizer.json"), "utf8"));
const tokenizer = createTokenizer(tokenizerData);
for (const text of ["Привет! Как дела?", "Hello, how are you?", "кот спит, кот ест"])
  console.log("encode", JSON.stringify(text), tokenizer.encode(text).length, "tokens");
const prompt = process.argv[2] ?? "Привет!";
const input = tokenizer.chat(prompt);
tokenizer.releaseEncoder();
const readBytes = name => name.startsWith("embed.")
  ? fs.readFile(path.join(pruned, `${name}.bin`))
  : fs.readFile(path.join(packed, `${name}.bin`));
const engine = await createEngine({ manifest, readBytes });
const model = await createQwen(engine, { capacity: 128 });
const count = Number(process.argv[3] ?? 12);
const generated = [];
let pending = input;
for (let i = 0; i < count; i++) {
  const token = pickToken(await model.forward(pending), generated);
  if (tokenizer.eos.has(token)) break;
  generated.push(token);
  pending = [token];
}
console.log(JSON.stringify({ vocab: manifest.config.vocab_size, inputTokens: input.length,
  bytesReadMiB: +(engine.store.bytesRead / 1024 / 1024).toFixed(1), text: tokenizer.decode(generated) }));
