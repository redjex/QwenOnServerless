import { db, api } from "sdk";

async function loadAsset(key) {
  const descriptor = await db.get("SELECT value FROM qwen_meta WHERE id = :id", { ":id": key });
  if (!descriptor) throw new Error(`Qwen asset is not installed: ${key}`);
  const info = JSON.parse(descriptor.value);
  const bytes = new Uint8Array(info.bytes);
  // Fetch several 256 KiB chunks per query to cut round trips, staying well under
  // the platform's per-response size limit.
  const BATCH = 4;
  let seen = 0;
  for (let first = 0; first < info.parts; first += BATCH) {
    const ids = Array.from({ length: Math.min(BATCH, info.parts - first) }, (_, i) => `${key}/${first + i}`);
    const params = {};
    ids.forEach((id, i) => { params[`:p${i}`] = id; });
    const rows = await db.all(
      `SELECT id, payload FROM qwen_blocks WHERE id IN (${ids.map((_, i) => `:p${i}`).join(",")})`, params);
    for (const row of rows) {
      if (!row.payload || !(row.payload instanceof Uint8Array)) throw new Error(`Missing block: ${row.id}`);
      const part = Number(row.id.slice(key.length + 1));
      bytes.set(row.payload, part * 256 * 1024);
      seen++;
    }
  }
  if (seen !== info.parts) throw new Error(`Missing block: ${key} (${seen}/${info.parts})`);
  return bytes;
}

const metaCache = new Map();

export async function loadMeta(name) {
  const cached = metaCache.get(name);
  if (cached) return cached;
  const value = JSON.parse(new TextDecoder().decode(await loadAsset(`meta:${name}`)));
  metaCache.set(name, value);
  return value;
}

export async function loadBlock(name) { return loadAsset(`block:${name}`); }

// Bundle descriptors written by the maintenance uploader: one row per bundle with
// its file_id and the blocks it contains. Empty means everything stays in SQLite.
let bundleMap = null;
const bundleCache = new Map();

async function bundleIndex() {
  if (bundleMap) return bundleMap;
  bundleMap = new Map();
  const rows = await db.all("SELECT value FROM qwen_meta WHERE id LIKE 'file:bundle-%'");
  for (const { value } of rows) {
    const info = JSON.parse(value);
    for (const block of info.blocks)
      bundleMap.set(block.name, { file: info.file_id, at: block.at, length: block.length });
  }
  return bundleMap;
}

// Prefer Telegram file storage for a block when it is available, otherwise SQLite.
export async function readBlock(name) {
  const index = await bundleIndex();
  const entry = index.get(name);
  if (!entry) return loadBlock(name);
  let bundle = bundleCache.get(entry.file);
  if (!bundle) {
    bundle = await api.getFileContent(entry.file);
    bundleCache.set(entry.file, bundle);
  }
  return bundle.subarray(entry.at, entry.at + entry.length);
}
