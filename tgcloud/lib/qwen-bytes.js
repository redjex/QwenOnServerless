const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const codes = new Int16Array(128).fill(-1);
for (let i = 0; i < alphabet.length; i++) codes[alphabet.charCodeAt(i)] = i;

export function decode64(text) {
  const end = text.length - (text.endsWith("==") ? 2 : text.endsWith("=") ? 1 : 0);
  const result = new Uint8Array(Math.floor(end * 6 / 8));
  let bits = 0, value = 0, offset = 0;
  for (let i = 0; i < end; i++) {
    const code = codes[text.charCodeAt(i)];
    if (code === undefined || code < 0) throw new Error("Invalid base64 data");
    value = (value << 6) | code;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      result[offset++] = (value >>> bits) & 255;
    }
  }
  return result;
}

export function unpack(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rows = view.getUint32(0, true), cols = view.getUint32(4, true);
  const isQ4 = bytes.byteLength === 12 + rows * cols / 8 + rows * cols / 2 && view.getUint32(8, true) === 4;
  if ((!isQ4 && bytes.byteLength !== 8 + rows * 4 + rows * cols) || cols % 32 !== 0)
    throw new Error("Invalid quantized tensor");
  // Copy if the SDK returns an unaligned BLOB view.
  const aligned = bytes.byteOffset % 4 ? bytes.slice() : bytes;
  return {
    rows, cols, bits: isQ4 ? 4 : 8,
    scales: new Float32Array(aligned.buffer, aligned.byteOffset + (isQ4 ? 12 : 8), isQ4 ? rows * cols / 32 : rows),
    weights: isQ4
      ? new Uint8Array(aligned.buffer, aligned.byteOffset + 12 + rows * cols / 8, rows * cols / 2)
      : new Int8Array(aligned.buffer, aligned.byteOffset + 8 + rows * 4, rows * cols),
  };
}
