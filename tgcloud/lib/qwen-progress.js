import { db } from "sdk";
import { qwenGenerations } from "../schema.js";

export async function saveProgress(id, prompt, result, status = "generating") {
  const state = JSON.stringify({ prompt, ...result, status, updatedAt: Date.now() });
  await db.insert(qwenGenerations).values({ id, state })
    .onConflictDoUpdate({ target: qwenGenerations.id, set: { state } }).run();
}
