import { db } from "sdk";
import { qwenKv } from "../schema.js";
import { eq } from "sdk/db";

const CHUNK = 256 * 1024;
const KV_WIDTH = 1024;
const BUDGET = 120 * 1024 * 1024;

// The answer state is the largest thing the bot stores, and the database is almost
// full of weights, so each value is kept in 16 bits instead of 32. float16 keeps a
// 10-bit mantissa (~0.05% relative error) — bfloat16's 8-bit mantissa (0.2%) was
// enough to push the continuation into gibberish, so the extra two bits matter.
const scratchFloat = new Float32Array(1);
const scratchBits = new Uint32Array(scratchFloat.buffer);

function toHalf(value) {
  scratchFloat[0] = value;
  const bits = scratchBits[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  const shifted = exponent - 127 + 15;
  if (shifted >= 31) return sign | 0x7c00;                       // overflow to infinity
  if (shifted <= 0) {
    if (shifted < -10) return sign;                              // underflow to zero
    return sign | ((((mantissa | 0x800000) >>> (1 - shifted)) + 0x1000) >>> 13);
  }
  return sign | (shifted << 10) | ((mantissa + 0x1000) >>> 13);
}

function fromHalf(half) {
  const sign = (half & 0x8000) << 16;
  const exponent = (half >>> 10) & 0x1f;
  const mantissa = half & 0x3ff;
  if (exponent === 0) scratchBits[0] = sign | (mantissa << 13);
  else if (exponent === 31) scratchBits[0] = sign | 0x7f800000 | (mantissa << 13);
  else scratchBits[0] = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  return scratchFloat[0];
}

function packHalves(source, count, out, offset) {
  for (let i = 0; i < count; i++) out[offset + i] = toHalf(source[i]);
}

function unpackHalves(halves, count, out, offset) {
  // `offset` indexes the packed buffer; the destination cache always starts at 0.
  for (let i = 0; i < count; i++) out[i] = fromHalf(halves[offset + i]);
}

export async function saveKv(owner, snapshot, model) {
  const floats = model.position * KV_WIDTH;   // values per cache (k, and v)
  const layerBytes = floats * 4;              // both caches, two bytes per value
  const required = layerBytes * model.caches.length;
  const usage = await db.get("SELECT coalesce(sum(length(payload)), 0) AS bytes FROM qwen_kv");
  if (usage.bytes + required > BUDGET)
    throw new Error("Недостаточно места для состояния ответа. Заверши или сбрось другой активный ответ.");
  for (let layer = 0; layer < model.caches.length; layer++) {
    const halves = new Uint16Array(floats * 2);
    packHalves(model.caches[layer].k, floats, halves, 0);
    packHalves(model.caches[layer].v, floats, halves, floats);
    const bytes = new Uint8Array(halves.buffer);
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
  if (checkpoint.layerBytes !== floats * 4) throw new Error("Invalid KV checkpoint");
  for (let layer = 0; layer < model.caches.length; layer++) {
    const bytes = new Uint8Array(checkpoint.layerBytes);
    for (let at = 0, part = 0; at < bytes.length; at += CHUNK, part++) {
      const row = await db.select({ payload: qwenKv.payload }).from(qwenKv)
        .where(eq(qwenKv.id, `${checkpoint.snapshot}/${layer}/${part}`)).get();
      if (!row) throw new Error("Состояние ответа не найдено. Начни новый вопрос.");
      bytes.set(row.payload, at);
    }
    const halves = new Uint16Array(bytes.buffer);
    unpackHalves(halves, floats, model.caches[layer].k, 0);
    unpackHalves(halves, floats, model.caches[layer].v, floats);
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
