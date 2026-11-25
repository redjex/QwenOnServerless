// Run manually with tgcloud run, without deploying. No DB changes.
// A successful allocation is a lower bound, not the platform's RAM quota.
export default function ({ mib = 1, kind = "typed" } = {}) {
  if (!Number.isInteger(mib) || mib < 1 || mib > 256)
    throw new Error("mib must be an integer from 1 to 256");
  if (kind !== "typed" && kind !== "heap") throw new Error("kind must be typed or heap");
  const started = Date.now();
  const buffers = [];
  const count = 1024 * 1024 / 8;
  try {
    for (let chunk = 0; chunk < mib; chunk++) {
      const values = kind === "typed" ? new Float64Array(count) : new Array(count);
      // Write all elements so the test does not just reserve virtual address space.
      for (let i = 0; i < count; i++) values[i] = i + chunk + 0.5;
      buffers.push(values);
    }
    let checksum = 0;
    // Keep all buffers live and read them after allocation.
    for (const values of buffers)
      for (let i = 0; i < count; i += 512) checksum += values[i];
    return {
      success: true, kind, requestedPayloadMiB: mib,
      elapsedMs: Date.now() - started, checksum,
      note: "Allocation passed in this invocation. Not a measurement of total RAM or official quota.",
    };
  } catch (error) {
    return {
      success: false, kind, requestedPayloadMiB: mib, completedChunks: buffers.length,
      elapsedMs: Date.now() - started, error: String(error),
      note: "A fatal OOM or platform timeout may terminate execution without returning this object.",
    };
  }
}
