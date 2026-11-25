// Packs the files the continuation worker needs on its server.
// Development aid, never deployed. Usage: node scripts/build-worker-tarball.js
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const output = path.join(root, "models/qwen-worker-deploy.tar.gz");
const files = [
  "package.json", "package-lock.json", "tgcloud.jsonc", ".gitignore",
  "scripts/qwen-worker.js",
  "tgcloud/schema.js", "tgcloud/lib", "tgcloud/handlers", "tgcloud/endpoints/qwenPump.js",
];
fs.rmSync(output, { force: true });
const result = spawnSync("tar", ["-czf", output, ...files], { cwd: root, stdio: "inherit" });
if (result.status !== 0) throw new Error("tar failed");
console.log(`Wrote ${path.relative(root, output)} (${fs.statSync(output).size} bytes)`);
