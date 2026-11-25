import { api } from "sdk";

// The relay bot: when the model bot posts a tick into the shared group, echo it
// back. The model bot does not receive its own messages, but it does receive this
// echo, so the ping-pong drives the next chunk of the answer without any VPS.
const TICK = "#qwen-tick ";

export default async function (message) {
  if (message.chat?.type === "private") return;
  if (!message.from?.is_bot) return;
  if (typeof message.text !== "string" || !message.text.startsWith(TICK)) return;
  await api.sendMessage({
    chat_id: message.chat.id,
    text: message.text,
    disable_notification: true,
  });
}