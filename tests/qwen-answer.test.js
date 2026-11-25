import test from "node:test";
import assert from "node:assert/strict";
import { answerMessage, quotedText, MAX_OUTPUT } from "../tgcloud/lib/qwen-answer.js";

function quote(message) {
  const [entity] = message.entities;
  assert.equal(entity.type, "blockquote");
  assert.equal(entity.offset + entity.length, message.text.length);
  return message.text.slice(entity.offset, entity.offset + entity.length);
}

test("progress is visible before the first token and advances through prefill and output", () => {
  assert.equal(quote(answerMessage({})), "Текст сгенерирован Idea Ai. [----] 0%");
  const state = { input: Array(80), prefill: 40, generated: [] };
  assert.match(quote(answerMessage(state)), /\[----\] 5%$/);
  state.prefill = 80;
  state.generated = Array(MAX_OUTPUT / 2);
  assert.match(quote(answerMessage(state)), /\[##--\] 55%$/);
});

test("only completion reaches 100%, including early EOS and the output limit", () => {
  assert.doesNotMatch(quote(answerMessage({ outputTokens: MAX_OUTPUT })), /100%$/);
  assert.match(quote(answerMessage({
    inputTokens: 80, prefillTokens: 80, outputTokens: MAX_OUTPUT,
  })), /\[###-\] 99%$/);
  for (const state of [
    { finished: true, text: "Привет!", outputTokens: 3 },
    { finished: true, text: "Ответ", outputTokens: MAX_OUTPUT, lengthLimit: true },
    { finished: true, outputTokens: 0 },
  ]) {
    const message = answerMessage(state);
    assert.equal(quote(message), "Текст сгенерирован Idea Ai. [####] 100%");
    if (state.lengthLimit) assert.match(message.text, /Достигнут предел длины ответа/);
  }
});

test("the quote preserves arbitrary generated text and uses UTF-16 entity offsets", () => {
  const text = "😊 <blockquote>Привет & мир</blockquote>\n**Ответ** [ссылка](url)";
  const message = answerMessage({ text, finished: true });
  assert.equal(message.text.slice(0, message.entities[0].offset), text + "\n\n");
  assert.equal(quote(message), "Текст сгенерирован Idea Ai. [####] 100%");
  assert.equal(message.parse_mode, undefined);
});

test("long answers leave room for the quote without splitting emoji or exceeding Telegram's limit", () => {
  for (const text of ["я".repeat(5000), "😊".repeat(2500)]) {
    const message = quotedText(text);
    assert.ok(message.text.length <= 4096);
    assert.ok(message.text.isWellFormed());
    assert.match(message.text, /…\n\nТекст сгенерирован/);
    assert.equal(quote(message), "Текст сгенерирован Idea Ai. [####] 100%");
  }
});
