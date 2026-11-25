import { api, db } from "sdk";
import { generationStep, answerMarkup, answerText } from "../lib/qwen-job.js";
import { deliverAnswer } from "../lib/qwen-delivery.js";
import { handleGroupMessage, scheduleRelay } from "../lib/qwen-relay.js";

const HELP = [
  "Qwen3-0.6B готова к тесту.",
  "Напиши вопрос. Ответ появится через несколько секунд и будет дописываться автоматически.",
  "Бот помнит последние сообщения диалога и учитывает их в ответе.",
  "",
  "/qwen вопрос — задать вопрос",
  "/last — показать последний ответ",
  "/stats — ограничения теста",
  "/stop — остановить текущий ответ",
  "/reset — очистить контекст диалога",
  "",
  "Ответ выдаётся порциями, до 64 токенов.",
].join("\n");

export default async function (message) {
  if (message.chat?.type !== "private") return handleGroupMessage(message);
  if (!message.from || typeof message.text !== "string") return;
  const reply = (text, reply_markup) => api.sendMessage({ chat_id: message.chat.id, text, reply_markup });
  const match = message.text.match(/^(\/\w+)(?:@\w+)?(?:\s+([\s\S]*))?$/);
  const command = match?.[1].toLowerCase();
  const argument = match?.[2] ?? "";
  const owner = `user:${message.from.id}`;
  if (command === "/start" || command === "/help") return reply(HELP);
  if (command === "/stats")
    return reply("Модель: Qwen3-0.6B (словарь и embeddings урезаны)\nВеса: 325,4 МиБ\nВход: до 160 токенов с шаблоном и историей\nОтвет: до 64 токенов, порциями\nКонтекст: до 256 позиций. Бот помнит последние сообщения; /reset очищает контекст.");
  if (command === "/stop") {
    await db.run("UPDATE qwen_generations SET auto_enabled = 0 WHERE id = :id", { ":id": owner });
    return reply("Текущий ответ остановлен. Уже начатая порция может ещё появиться.");
  }
  if (command === "/reset") {
    await db.run("DELETE FROM qwen_history WHERE id = :id", { ":id": owner });
    await db.run("UPDATE qwen_generations SET auto_enabled = 0 WHERE id = :id", { ":id": owner });
    return reply("Контекст диалога очищен. Следующий вопрос начнёт новый диалог.");
  }
  if (command === "/last") {
    const row = await db.get("SELECT state FROM qwen_generations WHERE id = :id", { ":id": owner });
    const state = row && JSON.parse(row.state);
    return state?.requestId ? reply(answerText(state), answerMarkup(state)) : reply("Сначала задай вопрос.");
  }
  if (command === "/train" || command === "/step")
    return reply("Сейчас работает готовая Qwen3-0.6B. Обучение по примерам ещё не подключено. Просто напиши вопрос.");
  if (command && !["/qwen", "/generate"].includes(command)) return reply(HELP);
  const prompt = command ? argument : message.text;
  if (!prompt.trim() || prompt.length > 500)
    return reply("Отправь короткий вопрос, от 1 до 500 символов.");
  const pending = await reply("Генерирую первую часть ответа…");
  try {
    const state = await generationStep(owner, {
      prompt, requestId: message.message_id, displayMessageId: pending.message_id,
      automatic: true, publish: value => deliverAnswer(message.chat.id, value),
    });
    // With a relay bot configured, continuation runs over Telegram without a VPS.
    await scheduleRelay(owner, state);
  } catch (error) {
    console.error(error);
    const detail = /[А-Яа-яЁё]/.test(error.message) ? error.message : "Попробуй /last или короткий вопрос.";
    return api.editMessageText({
      chat_id: message.chat.id, message_id: pending.message_id,
      text: `Не получилось завершить порцию. ${detail}`,
    });
  }
}
