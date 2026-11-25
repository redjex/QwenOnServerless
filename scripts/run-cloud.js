// Invoke the official CLI, respecting its explicit retry-after response.
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
const cli = fileURLToPath(new URL("../node_modules/@tgcloud/cli/bin/tgcloud.js", import.meta.url));
for (let attempt = 0; attempt < 5; attempt++) {
  const result = spawnSync(process.execPath, [cli, "run", ...process.argv.slice(2)], {
    encoding: "utf8", timeout: 180000, maxBuffer: 2 * 1024 * 1024,
  });
  process.stdout.write(result.stdout ?? "");
  process.stderr.write(result.stderr ?? "");
  if (result.status === 0) break;
  const retry = `${result.stdout}\n${result.stderr}`.match(/retry after (\d+)/i);
  if (!retry || attempt === 4) process.exit(result.status || 1);
  await new Promise(resolve => setTimeout(resolve, (Number(retry[1]) + 1) * 1000));
}
