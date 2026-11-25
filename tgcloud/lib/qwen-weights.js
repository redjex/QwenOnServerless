// Stores each weight block once inside the kernel's memory at a fixed offset and
// returns its geometry. `readBytes` is injected (qwen-runtime passes the SDK one)
// so this module and the engine stay free of the SDK and usable by local tooling.
export function createStore(manifest, kernel, readBytes) {
  const blocks = new Map();
  let bytesRead = 0;

  function install(name, bytes) {
    bytesRead += bytes.byteLength;
    const at = kernel.alloc(bytes.byteLength);
    new Uint8Array(kernel.memory.buffer, at, bytes.byteLength).set(bytes);
    const view = new DataView(kernel.memory.buffer, at, bytes.byteLength);
    const rows = view.getUint32(0, true), cols = view.getUint32(4, true);
    const isQ4 = bytes.byteLength === 12 + rows * cols / 8 + rows * cols / 2 && view.getUint32(8, true) === 4;
    if ((!isQ4 && bytes.byteLength !== 8 + rows * 4 + rows * cols) || cols % 32 !== 0)
      throw new Error("Invalid quantized tensor");
    const bits = isQ4 ? 4 : 8;
    const scalesAt = at + (bits === 4 ? 12 : 8);
    const weightsAt = scalesAt + (bits === 4 ? rows * cols / 8 : rows * 4);
    blocks.set(name, { rows, cols, bits, scalesAt, weightsAt });
  }

  async function load(name) {
    const known = blocks.get(name);
    if (known) return known;
    install(name, await readBytes(name));
    return blocks.get(name);
  }

  // Read the whole model in small parallel waves so the database wait overlaps.
  async function preload(concurrency = 8) {
    const names = Object.keys(manifest.blocks).filter(name => !blocks.has(name));
    for (let at = 0; at < names.length; at += concurrency) {
      const wave = names.slice(at, at + concurrency);
      const payloads = await Promise.all(wave.map(name => readBytes(name)));
      wave.forEach((name, index) => { if (!blocks.has(name)) install(name, payloads[index]); });
    }
  }

  return {
    load, preload, blocks,
    get bytesRead() { return bytesRead; },
  };
}
