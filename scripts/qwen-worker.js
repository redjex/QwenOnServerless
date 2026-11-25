// This process only triggers Telegram Serverless calls. It runs no model locally.
//
// The model can only produce a bounded number of tokens per invocation (the
// ~3 s CPU budget), so answer speed is governed by how quickly we chain calls.
// While a generation advances we re-poll almost immediately; only an idle queue
// waits. Several lanes can run at once (the platform allows ~8 concurrent calls),
// which lets a few users be served in parallel.
import fs from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const cli = path.join(root, "node_modules/@tgcloud/cli/bin/tgcloud.js");
const lock = path.join(root, "models/qwen-worker.lock");
const argv = process.argv;
const test = argv.includes("--test");
const once = argv.includes("--once");
const dryRun = argv.includes("--dry-run");
const parallelIndex = argv.indexOf("--parallel");
const parallel = once ? 1 : Math.max(1, Number(parallelIndex >= 0 ? argv[parallelIndex + 1] : 0) || 1);
const IDLE_MS = Math.max(500, Number(process.env.QWEN_IDLE_MS) || 3000);
const ACTIVE_MS = Math.max(0, Number(process.env.QWEN_ACTIVE_MS) || 150);
let stopping = false, lockOwned = false, failures = 0;
const children = new Set();

function acquireLock() {
  try {
    fs.writeFileSync(lock, JSON.stringify({ pid: process.pid }), { flag: "wx" });
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    const pid = JSON.parse(fs.readFileSync(lock, "utf8")).pid;
    let alive = true;
    try { process.kill(pid, 0); }
    catch (check) { if (check.code === "ESRCH") alive = false; else throw check; }
    if (alive) throw new Error(`Qwen worker is already running (PID ${pid}).`);
    fs.unlinkSync(lock);
    fs.writeFileSync(lock, JSON.stringify({ pid: process.pid }), { flag: "wx" });
  }
  lockOwned = true;
}

function invoke() {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cli, "run", "endpoints/qwenPump", JSON.stringify({ test, dryRun })], {
      cwd: root, stdio: ["ignore", "pipe", "pipe"],
    });
    children.add(child);
    let output = "";
    child.stdout.on("data", data => { output += data; });
    child.stderr.on("data", data => { output += data; });
    const timer = setTimeout(() => child.kill(), 120000);
    child.on("error", error => { clearTimeout(timer); children.delete(child); reject(error); });
    child.on("close", code => { clearTimeout(timer); children.delete(child); resolve({ code, output }); });
  });
}

async function wait(ms) {
  const until = Date.now() + ms;
  while (!stopping && Date.now() < until)
    await new Promise(resolve => setTimeout(resolve, Math.min(200, until - Date.now())));
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => { stopping = true; for (const child of children) child.kill(); });
}

function statusOf(output) {
  const match = output.match(/status:\s*['"](\w+)['"]/);
  return match ? match[1] : null;
}

async function lane(id) {
  let wasIdle = false;
  while (!stopping) {
    const result = await invoke();
    if (stopping) break;
    const status = statusOf(result.output);
    const idle = status === "idle";
    if (!idle || !wasIdle) console.log(`[${new Date().toISOString()}][lane ${id}] ${result.output.trim()}`);
    wasIdle = idle;
    let delay;
    if (result.code !== 0) {
      const retry = result.output.match(/retry after (\d+)/i);
      if (retry) delay = Math.max(IDLE_MS, (Number(retry[1]) + 1) * 1000);
      else {
        // A crash can be a cold start or a hiccup; only a run of them is fatal.
        failures++;
        delay = Math.max(IDLE_MS, 30000);
        if (failures >= 3 * parallel) throw new Error("Repeated CLI failures. Check login and restart the worker.");
      }
    } else {
      failures = 0;
      // Re-poll at once while there is progress; back off only when the queue is empty.
      delay = idle ? IDLE_MS : status === "busy" ? 3000 : ACTIVE_MS;
    }
    if (once) {
      if (result.code !== 0) process.exitCode = 1;
      break;
    }
    await wait(delay);
  }
}

try {
  acquireLock();
  console.log(`Qwen automatic continuation started (PID ${process.pid}, lanes ${parallel}, idle ${IDLE_MS}ms). Computation stays in Telegram.`);
  await Promise.all(Array.from({ length: parallel }, (_, i) => lane(i + 1)));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
} finally {
  if (lockOwned) fs.rmSync(lock, { force: true });
}
