import { createKernel } from "./qwen-kernel.js";
import { createStore } from "./qwen-weights.js";

// Builds a reusable engine: one kernel memory holding all weights at fixed offsets.
// Splitting it from the SDK keeps the model usable by local benchmark and validation.
export async function createEngine({ manifest, readBytes, preload = true } = {}) {
  const kernel = await createKernel();
  const store = createStore(manifest, kernel, readBytes);
  if (preload) await store.preload();
  return { manifest, kernel, store };
}
