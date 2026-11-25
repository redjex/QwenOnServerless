// Stages packed blocks and sends them to Telegram file storage through the CLI.
// Development aid, never deployed. Usage:
//   node scripts/upload-files.js [selector]
// selector matches block names, e.g. "layers.0.self_attn.k_proj" or "all".
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const root = path.resolve(import.meta.dirname, "..");
const sourceArg = process.argv.find(arg => arg.startsWith("--source="));
const sourceDir = path.join(root, sourceArg ? sourceArg.slice("--source=".length) : "models/packed");
const records = JSON.parse(fs.readFileSync(path.join(sourceDir, "uploads.json"), "utf8"));
const selector = process.argv[2] ?? "all";
const staged = path.join(root, "tgcloud/endpoints/qwenStaging.js");
const cli = path.join(root, "node_modules/@tgcloud/cli/bin/tgcloud.js");
const partDir = path.dirname(staged);
const parts = [];
function clearParts() {
  for (const file of parts) fs.rmSync(file, { force: true });
  parts.length = 0;
}

try {
  const chosen = records.filter(record => record.name && (selector === "all" || record.name.includes(selector)));
  const batches = [];
  let batch = [], batchBytes = 0;
  for (const record of chosen) {
    const size = fs.statSync(record.path).size;
    if (batch.length && batchBytes + size > 6 * 1024 * 1024) { batches.push(batch); batch = []; batchBytes = 0; }
    batch.push(record); batchBytes += size;
  }
  if (batch.length) batches.push(batch);
  for (const current of batches) {
    clearParts();
    const imports = [], entries = [];
    for (const record of current) {
      const bytes = fs.readFileSync(record.path);
      const entry = { name: record.name, length: bytes.length };
      const names = [];
      for (let offset = 0; offset < bytes.length; offset += 512 * 1024) {
        const index = parts.length;
        const partName = `qwenStagingPart${index}`;
        const file = path.join(partDir, `${partName}.js`);
        fs.writeFileSync(file, `export default ${JSON.stringify(bytes.subarray(offset, offset + 512 * 1024).toString("base64"))};\n`);
        parts.push(file);
        imports.push(`import p${index} from "./${partName}.js";`);
        names.push(`p${index}`);
      }
      entries.push(`{...${JSON.stringify(entry)},parts:[${names.join(",")}]}`);
    }
    fs.writeFileSync(staged, `${imports.join("\n")}\nexport default [${entries.join(",")}];\n`);
    let result;
    for (let attempt = 0; attempt < 6; attempt++) {
      result = spawnSync(process.execPath,
        [cli, "run", "endpoints/qwenFiles", JSON.stringify({ action: "upload" })],
        { cwd: root, encoding: "utf8", timeout: 300000, maxBuffer: 4 * 1024 * 1024 });
      process.stdout.write(result.stdout ?? "");
      process.stderr.write(result.stderr ?? "");
      if (result.status === 0) break;
      const retry = `${result.stdout}\n${result.stderr}`.match(/retry after (\d+)/i);
      if (!retry || attempt === 5) break;
      const wait = Number(retry[1]) + 1;
      console.log(`Rate limited; waiting ${wait}s before retrying`);
      await new Promise(resolve => setTimeout(resolve, wait * 1000));
    }
    if (result.status !== 0) throw new Error(`Upload failed for ${current.map(r => r.name).join(", ")}`);
  }
} finally {
  fs.writeFileSync(staged, "export default [];\n");
  clearParts();
}