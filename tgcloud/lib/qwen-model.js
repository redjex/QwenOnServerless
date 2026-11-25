const WIDTH = 1024, HEAD = 128, QHEADS = 16, KVHEADS = 8, KVWIDTH = HEAD * KVHEADS;
export const MAX_CONTEXT = 256;

export function rms(input, weights, width) {
  const output = new Float32Array(input.length);
  for (let start = 0; start < input.length; start += width) {
    let sum = 0;
    for (let i = 0; i < width; i++) sum += input[start + i] ** 2;
    const scale = 1 / Math.sqrt(sum / width + 1e-6);
    for (let i = 0; i < width; i++) output[start + i] = input[start + i] * scale * weights[i];
  }
  return output;
}

function rope(values, heads, position) {
  const batch = values.length / (heads * HEAD);
  for (let b = 0; b < batch; b++) {
    for (let j = 0; j < HEAD / 2; j++) {
      const angle = (position + b) / (1000000 ** (2 * j / HEAD));
      const cos = Math.cos(angle), sin = Math.sin(angle);
      for (let h = 0; h < heads; h++) {
        const at = (b * heads + h) * HEAD + j;
        const x = values[at], y = values[at + HEAD / 2];
        values[at] = x * cos - y * sin;
        values[at + HEAD / 2] = y * cos + x * sin;
      }
    }
  }
}

function attention(q, k, v, cache, position, batch) {
  cache.k.set(k, position * KVWIDTH);
  cache.v.set(v, position * KVWIDTH);
  const output = new Float32Array(batch * QHEADS * HEAD);
  const scores = new Float32Array(position + batch);
  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < QHEADS; h++) {
      const qAt = (b * QHEADS + h) * HEAD;
      const kvHead = Math.floor(h / (QHEADS / KVHEADS));
      let max = -Infinity;
      for (let t = 0; t <= position + b; t++) {
        let score = 0;
        const kvAt = t * KVWIDTH + kvHead * HEAD;
        for (let d = 0; d < HEAD; d++) score += q[qAt + d] * cache.k[kvAt + d];
        scores[t] = score / Math.sqrt(HEAD);
        max = Math.max(max, scores[t]);
      }
      let denominator = 0;
      for (let t = 0; t <= position + b; t++) denominator += (scores[t] = Math.exp(scores[t] - max));
      for (let t = 0; t <= position + b; t++) {
        const probability = scores[t] / denominator;
        const kvAt = t * KVWIDTH + kvHead * HEAD;
        for (let d = 0; d < HEAD; d++) output[qAt + d] += probability * cache.v[kvAt + d];
      }
    }
  }
  return output;
}

// `engine` is the object from qwen-engine.js: shared manifest, kernel and weights.
export async function createQwen(engine, { capacity = MAX_CONTEXT, trace = () => {} } = {}) {
  const { manifest, kernel, store } = engine;
  const config = manifest.config;
  if (manifest.format !== 1 || config.hidden_size !== WIDTH || config.head_dim !== HEAD ||
      config.num_hidden_layers !== 28 || !config.tie_word_embeddings)
    throw new Error("Unsupported Qwen configuration");
  const floats = capacity * KVWIDTH;
  const kvBytes = floats * 4;
  // One allocation, then views: the KV cache lives in WebAssembly memory, not the
  // 128 MiB JS heap, so the context is bounded by the DB snapshot, not by the heap.
  const kvBase = kernel.alloc(28 * 2 * kvBytes);
  const caches = Array.from({ length: 28 }, (_, layer) => ({
    k: new Float32Array(kernel.memory.buffer, kvBase + (layer * 2) * kvBytes, floats),
    v: new Float32Array(kernel.memory.buffer, kvBase + (layer * 2 + 1) * kvBytes, floats),
  }));
  let position = 0;
  async function matrix(name, inputs, batch) {
    const block = await store.load(name);
    return kernel.matmulAt(block.weightsAt, block.scalesAt, block.bits, inputs, block.rows, block.cols, batch);
  }
  async function embed(tokens) {
    const output = new Float32Array(tokens.length * WIDTH);
    const groups = new Map();
    tokens.forEach((id, i) => {
      if (!Number.isInteger(id) || id < 0 || id >= config.vocab_size) throw new Error("Invalid token");
      const chunk = Math.floor(id / manifest.embeddingChunkRows);
      if (!groups.has(chunk)) groups.set(chunk, []);
      groups.get(chunk).push([id % manifest.embeddingChunkRows, i]);
    });
    for (const [chunk, entries] of groups) {
      const block = await store.load(`embed.${chunk}`);
      if (block.bits === 8) {
        const scales = new Float32Array(kernel.memory.buffer, block.scalesAt, block.rows);
        const weights = new Int8Array(kernel.memory.buffer, block.weightsAt, block.rows * block.cols);
        for (const [row, i] of entries) {
          const scale = scales[row], rowAt = row * WIDTH, outAt = i * WIDTH;
          for (let j = 0; j < WIDTH; j++) output[outAt + j] = weights[rowAt + j] * scale;
        }
      } else if (block.bits === 4) {
        const groupScales = new Float32Array(kernel.memory.buffer, block.scalesAt, block.rows * block.cols / 32);
        const packed = new Uint8Array(kernel.memory.buffer, block.weightsAt, block.rows * block.cols / 2);
        for (const [row, i] of entries) {
          const rowAt = row * block.cols, outAt = i * WIDTH;
          for (let j = 0; j < WIDTH; j++) {
            const index = rowAt + j;
            const byte = packed[index >> 1];
            // Even indices use the low nibble, odd use the high nibble.
            const nibble = (index & 1) ? (byte >> 4) : (byte & 0x0f);
            output[outAt + j] = (nibble - 8) * groupScales[row * (block.cols / 32) + (j >> 5)];
          }
        }
      } else throw new Error("Unsupported embedding quantization");
    }
    return output;
  }
  async function layer(hidden, index, batch) {
    const prefix = `model.layers.${index}`;
    const norm = rms(hidden, manifest.norms[`${prefix}.input_layernorm.weight`], WIDTH);
    const q = rms(await matrix(`${prefix}.self_attn.q_proj.weight`, norm, batch),
      manifest.norms[`${prefix}.self_attn.q_norm.weight`], HEAD);
    const k = rms(await matrix(`${prefix}.self_attn.k_proj.weight`, norm, batch),
      manifest.norms[`${prefix}.self_attn.k_norm.weight`], HEAD);
    const v = await matrix(`${prefix}.self_attn.v_proj.weight`, norm, batch);
    rope(q, QHEADS, position);
    rope(k, KVHEADS, position);
    const attended = attention(q, k, v, caches[index], position, batch);
    const projected = await matrix(`${prefix}.self_attn.o_proj.weight`, attended, batch);
    for (let i = 0; i < hidden.length; i++) hidden[i] += projected[i];
    const post = rms(hidden, manifest.norms[`${prefix}.post_attention_layernorm.weight`], WIDTH);
    const gate = await matrix(`${prefix}.mlp.gate_proj.weight`, post, batch);
    const up = await matrix(`${prefix}.mlp.up_proj.weight`, post, batch);
    for (let i = 0; i < gate.length; i++) gate[i] = gate[i] / (1 + Math.exp(-gate[i])) * up[i];
    const down = await matrix(`${prefix}.mlp.down_proj.weight`, gate, batch);
    for (let i = 0; i < hidden.length; i++) hidden[i] += down[i];
    return hidden;
  }
  return {
    caches,
    get position() { return position; },
    set position(value) {
      if (!Number.isInteger(value) || value < 0 || value > capacity) throw new Error("Invalid KV position");
      position = value;
    },
    embed, layer,
    async forward(tokens, { head = true } = {}) {
      if (!tokens.length || position + tokens.length > capacity) throw new Error("Context limit reached");
      let hidden = await embed(tokens);
      for (let i = 0; i < 28; i++) {
        hidden = await layer(hidden, i, tokens.length);
        trace(i);
      }
      position += tokens.length;
      // Intermediate prefill chunks only need the KV cache, not the output projection.
      if (!head) return null;
      const last = rms(hidden.subarray(hidden.length - WIDTH), manifest.norms["model.norm.weight"], WIDTH);
      const logits = new Float32Array(config.vocab_size);
      for (let chunk = 0; chunk * manifest.embeddingChunkRows < config.vocab_size; chunk++) {
        const block = await store.load(`embed.${chunk}`);
        logits.set(kernel.matmulAt(block.weightsAt, block.scalesAt, block.bits, last, block.rows, block.cols, 1),
          chunk * manifest.embeddingChunkRows);
      }
      return logits;
    },
  };
}

export function pickToken(logits, generated = []) {
  // Deterministic decoding: a retry after a killed invocation regenerates the same
  // text, so streaming and checkpoints stay consistent. The loop breaker is a mild
  // repetition penalty over the last 32 tokens, as in the reference sampler.
  //
  // Measured on the deployed (pruned int4) model: a stronger penalty over *all*
  // generated tokens (1.15, plus 3-gram blocking) pushed the model onto odd synonyms
  // ("присутствие", "обладать") and out-of-language junk, and that junk then fed
  // back through the chat history and snowballed. 1.08 over a bounded window keeps
  // the answer on the reference track and still breaks greedy loops.
  const repeated = new Set(generated.slice(-32));
  let best = -Infinity, chosen = 0;
  for (let i = 0; i < logits.length; i++) {
    let score = logits[i];
    if (repeated.has(i)) score = score > 0 ? score / 1.08 : score * 1.08;
    if (score > best) { best = score; chosen = i; }
  }
  return chosen;
}
