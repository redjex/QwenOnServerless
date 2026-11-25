// Development aid: generate from a multi-turn transcript with the pruned model.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createTokenizer } from "../tgcloud/lib/qwen-tokenizer.js";
import { createEngine } from "../tgcloud/lib/qwen-engine.js";
import { createQwen, pickToken } from "../tgcloud/lib/qwen-model.js";

const root = fileURLToPath(new URL("../models/", import.meta.url));
const manifest = JSON.parse(await fs.readFile(path.join(root, "pruned/manifest.json"), "utf8"));
const data = JSON.parse(await fs.readFile(path.join(root, "pruned/tokenizer.json"), "utf8"));
const tokenizer = createTokenizer(data);
const messages = [
  { role: "user", text: "Меня зовут Иван. Запомни это." },
  { role: "assistant", text: "Хорошо, Иван! Я запомню." },
  { role: "user", text: "Как меня зовут?" },
];
const input = tokenizer.chatMessages(messages);
console.log("inputTokens", input.length);
tokenizer.releaseEncoder();
const readBytes = name => fs.readFile(path.join(name.startsWith("embed.") ? path.join(root, "pruned") : path.join(root, "packed"), `${name}.bin`));
const engine = await createEngine({ manifest, readBytes });
const model = await createQwen(engine, { capacity: 128 });
const generated = [];
let pending = input;
for (let i = 0; i < 24; i++) {
  const token = pickToken(await model.forward(pending, { head: true }), generated);
  if (tokenizer.eos.has(token)) break;
  generated.push(token);
  pending = [token];
}
console.log(JSON.stringify({ outputTokens: generated.length, text: tokenizer.decode(generated) }));