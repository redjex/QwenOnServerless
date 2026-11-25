// CLI-only benchmark; not deployed. Reports load cost, per-token cost and whether
// the engine was already resident (warm isolate) at the start of this invocation.
import { createTokenizer } from "../lib/qwen-tokenizer.js";
import { createQwen, pickToken } from "../lib/qwen-model.js";
import { engineLoaded, runtime } from "../lib/qwen-runtime.js";
import { loadMeta } from "../lib/qwen-store.js";

export default async function ({ prompt = "Привет!", maxTokens = 4 } = {}) {
  const warmBefore = engineLoaded();
  const started = Date.now();
  const engine = await runtime();
  const buildMs = Date.now() - started;
  const tokenizer = createTokenizer(await loadMeta("tokenizer"));
  const input = tokenizer.chat(prompt);
  tokenizer.releaseEncoder();
  const model = await createQwen(engine, { capacity: 256 });
  const generated = [];
  const perTokenMs = [];
  let tokens = input;
  for (let i = 0; i < maxTokens; i++) {
    const at = Date.now();
    const logits = await model.forward(tokens);
    const token = pickToken(logits, generated);
    if (tokenizer.eos.has(token)) break;
    generated.push(token);
    tokens = [token];
    perTokenMs.push(Date.now() - at);
  }
  return { warmBefore, buildMs, perTokenMs, bytesRead: engine.store.bytesRead,
    text: tokenizer.decode(generated) };
}
