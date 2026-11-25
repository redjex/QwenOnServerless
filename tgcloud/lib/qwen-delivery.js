import { api } from "sdk";
import { answerMessage, answerMarkup } from "./qwen-answer.js";

export async function deliverAnswer(chatId, state) {
  try {
    await api.editMessageText({
      chat_id: chatId, message_id: state.displayMessageId,
      ...answerMessage(state), reply_markup: answerMarkup(state),
    });
  } catch (error) {
    // Retried delivery after an interrupted DB commit is harmless.
    if (!error.description?.includes("message is not modified")) throw error;
  }
}
