import fs from "node:fs/promises";
import assert from "node:assert/strict";
import { createTokenizer } from "../tgcloud/lib/qwen-tokenizer.js";
import { createQwen, pickToken } from "../tgcloud/lib/qwen-model.js";
import { createEngine } from "../tgcloud/lib/qwen-engine.js";

const root = new URL("../models/", import.meta.url);
const reference = JSON.parse(await fs.readFile(new URL("reference.json", root), "utf8"));
const manifest = JSON.parse(await fs.readFile(new URL("packed/manifest.json", root), "utf8"));
const data = JSON.parse(await fs.readFile(new URL("packed/tokenizer.json", root), "utf8"));
const tokenizer = createTokenizer(data);
for (const vector of reference.tokenizer) {
  assert.deepEqual(tokenizer.encode(vector.text), vector.ids);
  assert.deepEqual(tokenizer.chat(vector.text), vector.chat);
  assert.equal(tokenizer.decode(vector.ids), vector.text);
}
const load = name => fs.readFile(new URL(`packed/${name}.bin`, root));
const engine = await createEngine({ manifest, readBytes: load });
const model = await createQwen(engine, { capacity: 128 });
const hidden = await model.layer(await model.embed([198, 198]), 0, 2);
const golden = reference.layer0.flat();
let error = 0;
for (let i = 0; i < golden.length; i++) error = Math.max(error, Math.abs(hidden[i] - golden[i]));
assert.ok(error < 0.003, `Layer mismatch: ${error}`);
const logits = await model.forward([198, 198]);
const chosen = pickToken(logits);
assert.equal(chosen, reference.diagnosticTop.ids[0]);
let maxLogitError = 0;
reference.diagnosticTop.ids.forEach((id, i) => {
  maxLogitError = Math.max(maxLogitError, Math.abs(logits[id] - reference.diagnosticTop.logits[i]));
});
assert.ok(maxLogitError < 0.03, `Logit mismatch: ${maxLogitError}`);
model.position = 0;
const chatLogits = await model.forward(tokenizer.chat("Привет!"));
assert.equal(pickToken(chatLogits), reference.chatTop.ids[0]);
console.log(JSON.stringify({ tokenizer: "matches official", layerMaxError: error, maxLogitError,
  firstChatToken: tokenizer.decode([pickToken(chatLogits)]), referenceSample: reference.chatSample }, null, 2));
