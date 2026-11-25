import binary from "./qwen-wasm.js";
import { decode64 } from "./qwen-bytes.js";

// Working area for inputs and outputs of a single matmul. Weights live below it,
// loaded once per engine, so a call never copies a weight matrix again.
const SCRATCH = 4 * 1024 * 1024;

export async function createKernel() {
  const { instance } = await WebAssembly.instantiate(decode64(binary), {});
  const { memory, matmul, matmul_q4, __heap_base } = instance.exports;
  const base = Math.ceil(Number(__heap_base.value) / 64) * 64;
  const inputAt = base;
  const outputAt = base + SCRATCH / 2;
  let cursor = base + SCRATCH;
  function reserve(end) {
    if (end > memory.buffer.byteLength)
      memory.grow(Math.ceil((end - memory.buffer.byteLength) / 65536));
  }
  reserve(cursor);
  return {
    memory,
    // Reserve stable bytes for one tensor; the offset never moves afterwards.
    alloc(bytes) {
      const at = cursor;
      reserve(cursor + bytes);
      cursor = at + bytes;
      return at;
    },
    get size() { return cursor; },
    // Weights stay where they were loaded; only the small input and output are written.
    matmulAt(weightsAt, scalesAt, bits, inputs, rows, cols, batch) {
      if (batch * cols * 4 > SCRATCH / 2 || batch * rows * 4 > SCRATCH / 2)
        throw new Error("Matmul operand exceeds the scratch area");
      new Float32Array(memory.buffer, inputAt, inputs.length).set(inputs);
      (bits === 4 ? matmul_q4 : matmul)(weightsAt, scalesAt, inputAt, outputAt, rows, cols, batch);
      return new Float32Array(memory.buffer, outputAt, batch * rows).slice();
    },
  };
}
