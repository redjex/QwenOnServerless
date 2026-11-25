export default function () {
  return {
    wasm: typeof WebAssembly, atob: typeof atob, btoa: typeof btoa,
    textEncoder: typeof TextEncoder, textDecoder: typeof TextDecoder,
    performance: typeof performance, bigint: typeof BigInt,
  };
}
