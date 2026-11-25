// Character MLP trained from random weights. No runtime dependencies.
export const alphabet = Array.from(
  "\n abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789.,!?;:-()«»\"'/"
);
export const V = alphabet.length;
export const CONTEXT = 8;
export const EMBED = 16;
export const HIDDEN = 64;
const INPUT = CONTEXT * EMBED;
const E = 0;
const W1 = V * EMBED;
const B1 = W1 + INPUT * HIDDEN;
const W2 = B1 + HIDDEN;
const B2 = W2 + HIDDEN * V;
export const PARAMETERS = B2 + V;
const index = new Map(alphabet.map((c, i) => [c, i]));

export function encode(text) {
  return Uint16Array.from(Array.from(text.toLowerCase().replace(/\r\n?/g, "\n")),
    c => index.get(c) ?? index.get("?"));
}

export function random(seed = 42) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

export function createModel(seed = 42) {
  const rng = random(seed);
  const weights = new Float32Array(PARAMETERS);
  for (let i = E; i < W1; i++) weights[i] = (rng() * 2 - 1) * 0.2;
  for (let i = W1; i < B1; i++) weights[i] = (rng() * 2 - 1) * Math.sqrt(3 / INPUT);
  for (let i = W2; i < B2; i++) weights[i] = (rng() * 2 - 1) * Math.sqrt(3 / HIDDEN);
  return { weights, steps: 0 };
}

export function workspace() {
  return {
    x: new Float32Array(INPUT),
    h: new Float32Array(HIDDEN),
    p: new Float32Array(V),
    dh: new Float32Array(HIDDEN),
    dx: new Float32Array(INPUT),
    gradient: new Float32Array(PARAMETERS),
  };
}

export function forward(model, context, work, temperature = 1) {
  const w = model.weights;
  for (let c = 0; c < CONTEXT; c++)
    for (let d = 0; d < EMBED; d++)
      work.x[c * EMBED + d] = w[E + context[c] * EMBED + d];
  for (let j = 0; j < HIDDEN; j++) {
    let sum = w[B1 + j];
    for (let i = 0; i < INPUT; i++) sum += work.x[i] * w[W1 + j * INPUT + i];
    work.h[j] = Math.tanh(sum);
  }
  let max = -Infinity;
  for (let k = 0; k < V; k++) {
    let sum = w[B2 + k];
    for (let j = 0; j < HIDDEN; j++) sum += work.h[j] * w[W2 + k * HIDDEN + j];
    work.p[k] = sum / temperature;
    max = Math.max(max, work.p[k]);
  }
  let total = 0;
  for (let k = 0; k < V; k++) total += (work.p[k] = Math.exp(work.p[k] - max));
  for (let k = 0; k < V; k++) work.p[k] /= total;
  return work.p;
}

export function lossAndGradient(model, context, target, work) {
  const p = forward(model, context, work);
  const loss = -Math.log(Math.max(p[target], 1e-30));
  const g = work.gradient;
  const w = model.weights;
  g.fill(0);
  work.dh.fill(0);
  work.dx.fill(0);
  for (let k = 0; k < V; k++) {
    const delta = p[k] - (k === target ? 1 : 0);
    g[B2 + k] = delta;
    for (let j = 0; j < HIDDEN; j++) {
      g[W2 + k * HIDDEN + j] = delta * work.h[j];
      work.dh[j] += delta * w[W2 + k * HIDDEN + j];
    }
  }
  for (let j = 0; j < HIDDEN; j++) {
    const delta = work.dh[j] * (1 - work.h[j] * work.h[j]);
    g[B1 + j] = delta;
    for (let i = 0; i < INPUT; i++) {
      g[W1 + j * INPUT + i] = delta * work.x[i];
      work.dx[i] += delta * w[W1 + j * INPUT + i];
    }
  }
  // += is necessary: a character may appear repeatedly in one context.
  for (let c = 0; c < CONTEXT; c++)
    for (let d = 0; d < EMBED; d++)
      g[E + context[c] * EMBED + d] += work.dx[c * EMBED + d];
  return loss;
}

export function train(model, text, { steps = 32, budgetMs = 100, rate = 0.03, rng = Math.random } = {}) {
  const tokens = encode(text);
  if (tokens.length <= CONTEXT) throw new Error("Нужно больше 8 символов текста.");
  const work = workspace();
  const start = Date.now();
  let totalLoss = 0;
  let completed = 0;
  while (completed < steps && (completed === 0 || Date.now() - start < budgetMs)) {
    const pos = CONTEXT + Math.floor(rng() * (tokens.length - CONTEXT));
    totalLoss += lossAndGradient(model, tokens.subarray(pos - CONTEXT, pos), tokens[pos], work);
    let norm2 = 0;
    for (const g of work.gradient) norm2 += g * g;
    const scale = rate * Math.min(1, 5 / Math.sqrt(norm2 || 1));
    for (let i = 0; i < PARAMETERS; i++) model.weights[i] -= scale * work.gradient[i];
    model.steps++;
    completed++;
  }
  return { completed, meanLoss: totalLoss / completed, elapsedMs: Date.now() - start };
}

export function generate(model, prompt = "", { length = 120, temperature = 0.8, rng = Math.random, budgetMs = 100 } = {}) {
  if (!(temperature > 0)) throw new Error("temperature must be positive");
  const context = new Uint16Array(CONTEXT).fill(index.get(" "));
  const prefix = encode(prompt);
  context.set(prefix.slice(-CONTEXT), Math.max(0, CONTEXT - prefix.length));
  const work = workspace();
  let output = Array.from(prefix, id => alphabet[id]).join("");
  const start = Date.now();
  for (let n = 0; n < length && (n === 0 || Date.now() - start < budgetMs); n++) {
    const p = forward(model, context, work, temperature);
    let r = rng();
    let chosen = V - 1;
    for (let k = 0; k < V; k++) {
      r -= p[k];
      if (r <= 0) { chosen = k; break; }
    }
    output += alphabet[chosen];
    context.copyWithin(0, 1);
    context[CONTEXT - 1] = chosen;
  }
  return output;
}

export function serialize(model) {
  return JSON.stringify({ format: 1, alphabet: alphabet.join(""), steps: model.steps, weights: Array.from(model.weights) });
}

export function deserialize(json) {
  const data = JSON.parse(json);
  if (data.format !== 1 || data.alphabet !== alphabet.join("") ||
      !Number.isSafeInteger(data.steps) || data.steps < 0 ||
      !Array.isArray(data.weights) || data.weights.length !== PARAMETERS ||
      !data.weights.every(x => typeof x === "number" && Number.isFinite(x) && Math.abs(x) < 1e6))
    throw new Error("Неверный формат сохранённой модели.");
  return { weights: Float32Array.from(data.weights), steps: data.steps };
}

export function memoryPlan() {
  const work = workspace();
  const workspaceBytes = Object.values(work).reduce((sum, a) => sum + a.byteLength, 0);
  return {
    vocabulary: V, context: CONTEXT, embedding: EMBED, hidden: HIDDEN,
    parameters: PARAMETERS, weightsBytes: PARAMETERS * 4,
    trainingArraysBytes: PARAMETERS * 4 + workspaceBytes,
    note: "Only numerical buffers; excludes JS runtime, corpus, SDK, JSON and DB copies.",
  };
}
