// Transfer authentication through SSH stdin, using the official CLI resolver.
// The token is never printed, placed in command-line arguments, or archived.
import { spawn } from "node:child_process";
import { resolveToken } from "../node_modules/@tgcloud/cli/src/core/credentials.js";

const host = process.argv[2];
if (!/^root@[A-Za-z0-9.-]+$/.test(host ?? "")) {
  throw new Error("Usage: node scripts/deploy-worker-auth.js root@server");
}
const token = await resolveToken();
if (!/^app\d+:[A-Za-z0-9_-]+$/.test(token)) throw new Error("Invalid CLI token format");
const ssh = spawn("ssh", [
  "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
  "set -eu; umask 077; cat > /etc/qwen-worker.env; chmod 600 /etc/qwen-worker.env",
], { stdio: ["pipe", "ignore", "inherit"] });
ssh.on("error", error => { console.error(error.message); process.exitCode = 1; });
ssh.stdin.on("error", () => { process.exitCode = 1; });
ssh.stdin.end(`TGCLOUD_TOKEN=${token}\n`);
const code = await new Promise(resolve => ssh.on("close", resolve));
if (code !== 0) process.exitCode = 1;
else console.log("Worker authentication transferred securely.");
