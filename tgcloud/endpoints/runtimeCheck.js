// Manual cloud integration check; excluded from the deployment script.
// Uses a reserved negative ID, never sends chat messages.
import { db, api } from "sdk";
import { createModel, deserialize, serialize, train, generate, memoryPlan, random } from "../lib/model.js";

const ID = -1;
const CORPUS = "кот спит. кот ест. ".repeat(20);

export default async function ({ action = "read" } = {}) {
  if (action === "cleanup") {
    const result = await db.run("DELETE FROM tiny_text_models WHERE user_id = :id", { ":id": ID });
    return { deleted: result.rowsAffected };
  }
  if (action === "init") {
    await db.run(
      "INSERT INTO tiny_text_models (user_id, revision, checkpoint, corpus) VALUES (:id, 0, :checkpoint, :corpus) ON CONFLICT(user_id) DO NOTHING",
      { ":id": ID, ":checkpoint": serialize(createModel(42)), ":corpus": CORPUS }
    );
  } else if (action !== "read" && action !== "train") {
    throw new Error("action must be init, read, train or cleanup");
  }
  const row = await db.get("SELECT * FROM tiny_text_models WHERE user_id = :id", { ":id": ID });
  if (!row) throw new Error("Run init first");
  const model = deserialize(row.checkpoint);
  let progress = null;
  if (action === "train") {
    progress = train(model, row.corpus, { steps: 128, rng: random(model.steps + 1) });
    const saved = await db.run(
      "UPDATE tiny_text_models SET checkpoint = :checkpoint, revision = revision + 1 WHERE user_id = :id AND revision = :revision",
      { ":checkpoint": serialize(model), ":id": ID, ":revision": row.revision }
    );
    if (saved.rowsAffected !== 1) throw new Error("Concurrent modification");
    const persisted = await db.get("SELECT checkpoint FROM tiny_text_models WHERE user_id = :id", { ":id": ID });
    if (persisted.checkpoint !== serialize(model)) throw new Error("Checkpoint round trip mismatch");
  }
  const me = await api.getMe();
  return {
    botUsername: me.username, steps: model.steps, progress, memory: memoryPlan(),
    checkpointCharacters: serialize(model).length,
    sample: generate(model, "кот спит", { length: 80, rng: random(9) }),
  };
}
