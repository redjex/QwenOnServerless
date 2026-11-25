const pattern = /'(?:s|t|re|ve|m|ll|d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu;
const byteToChar = new Array(256);
const charToByte = new Map();
let extra = 0;
for (let b = 0; b < 256; b++) {
  const visible = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
  const ch = String.fromCodePoint(visible ? b : 256 + extra++);
  byteToChar[b] = ch;
  charToByte.set(ch, b);
}

export function createTokenizer(data, { encoder: enableEncoder = true } = {}) {
  const tokens = data.tokens;
  const vocab = new Map(enableEncoder ? tokens.map((token, id) => [token, id]) : []);
  const ranks = new Map(enableEncoder ? data.merges.map((pair, rank) => [pair.join("\0"), rank]) : []);
  const special = new Map(data.added.map(item => [item.content, item.id]));
  const idToSpecial = new Map(data.added.map(item => [item.id, item.content]));
  const encoder = new TextEncoder(), decoder = new TextDecoder();
  let encoderReleased = !enableEncoder;
  function encode(text) {
    if (encoderReleased) throw new Error("Tokenizer encoder has been released");
    const ids = [];
    for (const match of text.matchAll(pattern)) {
      const pieces = Array.from(encoder.encode(match[0]), b => byteToChar[b]);
      while (pieces.length > 1) {
        let best = Infinity, at = -1;
        for (let i = 0; i < pieces.length - 1; i++) {
          const rank = ranks.get(pieces[i] + "\0" + pieces[i + 1]);
          if (rank !== undefined && rank < best) { best = rank; at = i; }
        }
        if (at < 0) break;
        pieces.splice(at, 2, pieces[at] + pieces[at + 1]);
      }
      for (const piece of pieces) {
        const id = vocab.get(piece);
        if (id === undefined) throw new Error("Unknown BPE token");
        ids.push(id);
      }
    }
    return ids;
  }
  function decode(ids) {
    const bytes = [];
    for (const id of ids) {
      if (idToSpecial.has(id)) continue;
      const token = tokens[id];
      if (token === undefined) throw new Error("Unknown token ID");
      for (const char of token) bytes.push(charToByte.get(char));
    }
    return decoder.decode(Uint8Array.from(bytes));
  }
  return {
    encode, decode,
    releaseEncoder() {
      // BPE maps are only needed for the prompt. Keeping them alive makes GC
      // expensive while hundreds of MiB of weight buffers are being loaded.
      vocab.clear();
      ranks.clear();
      encoderReleased = true;
    },
    chat(text) {
      // Official Qwen3 non-thinking template. User text cannot insert control tokens.
      return [
        special.get("<|im_start|>"), ...encode("user\n"), ...encode(text),
        special.get("<|im_end|>"), ...encode("\n"),
        special.get("<|im_start|>"), ...encode("assistant\n"),
        special.get("<think>"), ...encode("\n\n"),
        special.get("</think>"), ...encode("\n\n"),
      ];
    },
    // Multi-turn transcript followed by the assistant prefix. Reuses chat("") to
    // get the exact assistant preamble without duplicating the special tokens.
    chatMessages(messages) {
      const start = special.get("<|im_start|>"), end = special.get("<|im_end|>");
      const ids = [];
      for (const message of messages) {
        const role = message.role === "assistant" ? "assistant\n" : "user\n";
        ids.push(start, ...encode(role), ...encode(message.text), end, ...encode("\n"));
      }
      const single = this.chat("");
      const head = [start, ...encode("user\n"), end, ...encode("\n")];
      ids.push(...single.slice(head.length));
      return ids;
    },
    eos: new Set(data.added
      .filter(item => item.content === "<|endoftext|>" || item.content === "<|im_end|>")
      .map(item => item.id)),
  };
}
