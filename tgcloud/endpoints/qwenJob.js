// CLI-only integration test; no messages are sent to Telegram chats.
import { db } from "sdk";
import { generationStep } from "../lib/qwen-job.js";
import { removeKv } from "../lib/qwen-checkpoint.js";

export default async function ({ action = "read", prompt = "Привет!", requestId = "test", automatic = false, budget } = {}) {
  const owner = "diagnostic-job";
  if (action === "pause") {
    await db.run("UPDATE qwen_generations SET auto_enabled = 0 WHERE id = :id", { ":id": owner });
    return { paused: true };
  }
  if (action === "cleanup") {
    await removeKv(owner);
    await db.run("DELETE FROM qwen_history WHERE id = :id", { ":id": owner });
    await db.run("DELETE FROM qwen_generations WHERE id = :id", { ":id": owner });
    return { cleaned: true };
  }
  let state;
  if (action === "start") state = await generationStep(owner, { prompt, requestId, automatic, budget });
  else if (action === "next") state = await generationStep(owner, { requestId, budget });
  else {
    const row = await db.get("SELECT state FROM qwen_generations WHERE id = :id", { ":id": owner });
    state = row ? JSON.parse(row.state) : null;
  }
  if (!state) return { exists: false };
  return {
    text: state.text, outputTokens: state.outputTokens, finished: state.finished,
    position: state.checkpoint?.position, bytesRead: state.bytesRead,
    inputTokens: state.inputTokens,
  };
}
