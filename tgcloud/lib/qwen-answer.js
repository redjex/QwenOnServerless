export const MAX_OUTPUT = 192;

// The final answer length is unknown: reserve 10% for prefill and estimate the
// rest against the output budget. Only a finished generation reaches 100%.
export function answerProgress(state) {
  if (state.finished) return 100;
  const input = state.input?.length ?? state.inputTokens ?? 0;
  const prefill = state.prefill ?? state.prefillTokens ?? 0;
  const output = state.generated?.length ?? state.outputTokens ?? 0;
  const preparation = input > 0 ? Math.min(1, prefill / input) * 10 : 0;
  return Math.max(0, Math.min(99, Math.floor(preparation + output / MAX_OUTPUT * 90)));
}

export function quotedText(text, progress = 100) {
  const percent = Math.max(0, Math.min(100, Math.floor(progress)));
  const filled = Math.floor(percent / 25);
  const quote = `Текст сгенерирован Idea Ai. [${"#".repeat(filled)}${"-".repeat(4 - filled)}] ${percent}%`;
  let body = String(text);
  const available = 4096 - quote.length - 2;
  if (body.length > available) {
    // Telegram entities use UTF-16 offsets. Never cut an emoji's surrogate pair.
    body = body.slice(0, available - 1).replace(/[\uD800-\uDBFF]$/, "") + "…";
  }
  const prefix = body ? `${body}\n\n` : "";
  return {
    text: prefix + quote,
    entities: [{ type: "blockquote", offset: prefix.length, length: quote.length }],
  };
}

export function answerMessage(state) {
  let text = state.text || (state.finished ? "Модель завершила ответ без текста." : "Думаю над ответом...");
  if (state.lengthLimit) text += `\n\nДостигнут предел длины ответа (${MAX_OUTPUT} токенов).`;
  return quotedText(text, answerProgress(state));
}

export function answerText(state) {
  return answerMessage(state).text;
}

// Continuation is automatic; there is no manual button.
export function answerMarkup() {
  return { inline_keyboard: [] };
}
