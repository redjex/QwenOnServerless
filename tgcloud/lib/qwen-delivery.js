import { api } from "sdk";
import { answerText, answerMarkup } from "./qwen-job.js";

export async function deliverAnswer(chatId, state) {
  try {
    await api.editMessageText({
      chat_id: chatId, message_id: state.displayMessageId,
      text: answerText(state), reply_markup: answerMarkup(state),
    });
  } catch (error) {
    // Retried delivery after an interrupted DB commit is harmless.
    if (!error.description?.includes("message is not modified")) throw error;
  }
}
