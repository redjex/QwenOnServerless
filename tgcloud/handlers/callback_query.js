import { api } from "sdk";
import { generationStep } from "../lib/qwen-job.js";
import { deliverAnswer } from "../lib/qwen-delivery.js";

export default async function (query) {
  if (!query.data?.startsWith("qwen_next:")) return;
  const message = query.message;
  if (message?.chat?.type !== "private" || message.chat.id !== query.from.id) {
    return api.answerCallbackQuery({ callback_query_id: query.id, text: "Продолжение доступно в личном чате с ботом." });
  }
  await api.answerCallbackQuery({ callback_query_id: query.id, text: "Дописываю ответ…" });
  try {
    await generationStep(`user:${query.from.id}`, {
      requestId: query.data.slice("qwen_next:".length), displayMessageId: message.message_id,
      publish: state => deliverAnswer(message.chat.id, state),
    });
  } catch (error) {
    if (error.description?.includes("message is not modified")) return;
    console.error(error);
    const detail = /[А-Яа-яЁё]/.test(error.message) ? error.message : "Не удалось продолжить. Попробуй /last.";
    await api.sendMessage({ chat_id: message.chat.id, text: detail });
  }
}
