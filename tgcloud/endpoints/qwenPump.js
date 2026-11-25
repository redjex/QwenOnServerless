// Called by the local worker through the authenticated official CLI.
// It is deliberately excluded from the public deployment.
import { db } from "sdk";
import { generationStep, LEASE_MS, STALE_MS } from "../lib/qwen-job.js";
import { deliverAnswer } from "../lib/qwen-delivery.js";

// When a relay bot drives continuation, the worker only rescues jobs the relay has
// not advanced for this long (e.g. after a CPU-killed step or a lost tick).
const WATCHDOG_MS = 12000;

export default async function ({ test = false, dryRun = false } = {}) {
  const now = Date.now();
  const scope = test ? "id = 'diagnostic-job'" : "id LIKE 'user:%'";
  const select = () => db.get(
    `SELECT id, state, revision, failure_count FROM qwen_generations WHERE ${scope} AND auto_enabled = 1 AND busy_until <= :now AND next_run_at <= :now ORDER BY next_run_at, id LIMIT 1`,
    { ":now": now });
  let row = await select();
  if (!row) {
    // Reclaim leases abandoned by a CPU-killed invocation so a job never stalls
    // for the whole lease duration.
    const cleared = await db.run(
      "UPDATE qwen_generations SET busy_until = 0 WHERE busy_until > :now AND busy_until - :lease < :stale",
      { ":now": now, ":lease": LEASE_MS, ":stale": now - STALE_MS });
    if (cleared.rowsAffected) row = await select();
  }
  if (!row) return { status: "idle" };

  if (!test) {
    const relay = await db.get("SELECT value FROM qwen_meta WHERE id = 'relay_group'");
    if (relay) {
      const state = JSON.parse(row.state);
      // The relay is alive and recently advanced this job: stay out of its way.
      if (now - (state.updatedAt ?? 0) < WATCHDOG_MS) return { status: "idle", relay: true };
      // Otherwise this is a rescue: continue the stalled relay job.
    }
  }

  const previous = JSON.parse(row.state);
  if (dryRun) return { status: "pending", outputTokens: previous.outputTokens, test };
  if (previous.finished || !previous.requestId || (!test && !Number.isSafeInteger(previous.displayMessageId))) {
    await db.run("UPDATE qwen_generations SET auto_enabled = 0 WHERE id = :id AND revision = :revision",
      { ":id": row.id, ":revision": row.revision });
    return { status: "skipped" };
  }
  try {
    const state = await generationStep(row.id, {
      requestId: previous.requestId,
      publish: test ? undefined : state => deliverAnswer(Number(row.id.slice("user:".length)), state),
    });
    return { status: state.finished ? "finished" : "advanced", outputTokens: state.outputTokens, test };
  } catch (error) {
    const current = await db.get("SELECT state, revision, busy_until FROM qwen_generations WHERE id = :id", { ":id": row.id });
    if (!current || current.busy_until > Date.now()) return { status: "busy" };
    if (JSON.parse(current.state).requestId !== previous.requestId) return { status: "superseded" };
    const retrySeconds = Number(error.parameters?.retry_after);
    const rateLimited = Number.isFinite(retrySeconds) && retrySeconds > 0;
    const failures = row.failure_count + (rateLimited ? 0 : 1);
    const enabled = failures < 3;
    await db.run(
      "UPDATE qwen_generations SET failure_count = :failures, next_run_at = :next, auto_enabled = CASE WHEN :enabled = 0 THEN 0 ELSE auto_enabled END WHERE id = :id AND revision = :revision",
      { ":id": row.id, ":revision": current.revision, ":failures": failures,
        ":next": Date.now() + (rateLimited ? retrySeconds + 1 : 30) * 1000, ":enabled": enabled ? 1 : 0 });
    console.warn("Automatic generation failed:", error.description || error.message);
    return { status: enabled ? "retry" : "paused", failures };
  }
}