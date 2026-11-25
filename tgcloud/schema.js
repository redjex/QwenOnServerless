import { table, integer, text, blob } from "sdk/db";

export const models = table("tiny_text_models", {
  userId: integer("user_id").primaryKey(),
  revision: integer("revision").notNull().default(0),
  checkpoint: text("checkpoint").notNull(),
  corpus: text("corpus").notNull(),
});

export const qwenBlocks = table("qwen_blocks", {
  id: text("id").primaryKey(),
  payload: blob("payload").notNull(),
  checksum: text("checksum").notNull(),
});

export const qwenMeta = table("qwen_meta", {
  id: text("id").primaryKey(),
  value: text("value").notNull(),
});

export const qwenGenerations = table("qwen_generations", {
  id: text("id").primaryKey(),
  state: text("state").notNull(),
  revision: integer("revision").notNull().default(0),
  busyUntil: integer("busy_until").notNull().default(0),
  autoEnabled: integer("auto_enabled").notNull().default(0),
  nextRunAt: integer("next_run_at").notNull().default(0),
  failureCount: integer("failure_count").notNull().default(0),
});

// Trimmed conversation transcript per chat, so follow-up questions keep context.
export const qwenHistory = table("qwen_history", {
  id: text("id").primaryKey(),
  messages: text("messages").notNull(),
});

export const qwenKv = table("qwen_kv", {
  id: text("id").primaryKey(),
  owner: text("owner").notNull(),
  payload: blob("payload").notNull(),
});
