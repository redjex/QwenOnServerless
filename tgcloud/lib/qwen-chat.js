import { createQwen, pickToken } from "./qwen-model.js";
import { createTokenizer } from "./qwen-tokenizer.js";
import { loadMeta } from "./qwen-store.js";
import { runtime } from "./qwen-runtime.js";

export async function qwenChat(prompt, { maxTokens = 24, trace = () => {} } = {}) {
  if (typeof prompt !== "string" || !prompt.trim() || prompt.length > 500)
    throw new Error("Для первого теста отправь от 1 до 500 символов.");
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 48)
    throw new Error("maxTokens must be between 1 and 48");
  const tokenizer = createTokenizer(await loadMeta("tokenizer"));
  const input = tokenizer.chat(prompt);
  if (input.length > 128) throw new Error("Для первого теста сократи вопрос до 128 токенов.");
  tokenizer.releaseEncoder();
  const engine = await runtime();
  const model = await createQwen(engine, { capacity: 256 });
  const generated = [];
  let tokens = input, finished = false;
  for (let i = 0; i < maxTokens; i++) {
    const logits = await model.forward(tokens);
    const token = pickToken(logits, generated);
    if (tokenizer.eos.has(token)) { finished = true; break; }
    generated.push(token);
    tokens = [token];
    await trace({ generated: generated.length, text: tokenizer.decode(generated), bytesRead: engine.store.bytesRead });
  }
  return {
    text: tokenizer.decode(generated), inputTokens: input.length,
    outputTokens: generated.length, finished, bytesRead: engine.store.bytesRead,
  };
}
