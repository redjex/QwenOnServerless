import { createQwen } from "../lib/qwen-model.js";
import { engineLoaded, runtime } from "../lib/qwen-runtime.js";

export default async function ({ tokens = [198, 198], layer = 0 } = {}) {
  const engine = await runtime();
  const model = await createQwen(engine, { capacity: 16 });
  let hidden = await model.embed(tokens);
  hidden = await model.layer(hidden, layer, tokens.length);
  let sum = 0, norm = 0;
  for (const x of hidden) { sum += x; norm += x * x; }
  return { length: hidden.length, sum, norm, first: Array.from(hidden.slice(0, 16)), warm: engineLoaded() };
}
