import { loadMeta, readBlock } from "./qwen-store.js";
import { createEngine } from "./qwen-engine.js";

// The weights and kernel are immutable and shared by every user. Keeping them in
// module scope lets a warm isolate skip re-reading them from the database or files.
let shared = null, pending = null;

export function engineLoaded() { return shared !== null; }

export async function runtime() {
  if (shared) return shared;
  if (!pending) pending = (async () => {
    const manifest = await loadMeta("manifest");
    return (shared = await createEngine({ manifest, readBytes: readBlock }));
  })();
  return pending;
}
