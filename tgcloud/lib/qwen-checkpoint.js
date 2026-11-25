import { db } from "sdk";
import { qwenKv } from "../schema.js";
import { eq } from "sdk/db";

const CHUNK = 256 * 1024;
const KV_WIDTH = 1024;
const BUDGET = 120 * 1024 * 1024;

export async function saveKv(owner, snapshot, model) {
  const floats = model.position * KV_WIDTH;
  const layerBytes = floats * 8;
  const required = layerBytes * model.caches.length;
  const usage = await db.get("SELECT coalesce(sum(length(payload)), 0) AS bytes FROM qwen_kv");
  if (usage.bytes + required > BUDGET)
    throw new Error("Недостаточно места для состояния ответа. Заверши или сбрось другой активный ответ.");
  for (let layer = 0; layer < model.caches.length; layer++) {
    const bytes = new Uint8Array(layerBytes);
    const values = new Float32Array(bytes.buffer);
    values.set(model.caches[layer].k.subarray(0, floats));
    values.set(model.caches[layer].v.subarray(0, floats), floats);
    for (let at = 0, part = 0; at < bytes.length; at += CHUNK, part++) {
      await db.insert(qwenKv).values({
        id: `${snapshot}/${layer}/${part}`, owner,
        payload: bytes.slice(at, at + CHUNK),
      }).run();
    }
  }
  return { snapshot, position: model.position, layerBytes };
}

export async function restoreKv(checkpoint, model) {
  const floats = checkpoint.position * KV_WIDTH;
  if (checkpoint.layerBytes !== floats * 8) throw new Error("Invalid KV checkpoint");
  for (let layer = 0; layer < model.caches.length; layer++) {
    const bytes = new Uint8Array(checkpoint.layerBytes);
    for (let at = 0, part = 0; at < bytes.length; at += CHUNK, part++) {
      const row = await db.select({ payload: qwenKv.payload }).from(qwenKv)
        .where(eq(qwenKv.id, `${checkpoint.snapshot}/${layer}/${part}`)).get();
      if (!row) throw new Error("Состояние ответа не найдено. Начни новый вопрос.");
      bytes.set(row.payload, at);
    }
    const values = new Float32Array(bytes.buffer);
    model.caches[layer].k.set(values.subarray(0, floats));
    model.caches[layer].v.set(values.subarray(floats));
  }
  model.position = checkpoint.position;
}

export async function removeKv(owner, keepSnapshot = null) {
  if (keepSnapshot) {
    await db.run("DELETE FROM qwen_kv WHERE owner = :owner AND id NOT LIKE :keep",
      { ":owner": owner, ":keep": `${keepSnapshot}/%` });
  } else {
    await db.run("DELETE FROM qwen_kv WHERE owner = :owner", { ":owner": owner });
  }
}
