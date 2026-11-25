import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const root = path.resolve(import.meta.dirname, "..");
const sourceArg = process.argv.find(arg => arg.startsWith("--source="));
const sourceDir = sourceArg ? sourceArg.slice("--source=".length) : "models/packed";
const sourceRoot = path.join(root, sourceDir);
const records = JSON.parse(fs.readFileSync(path.join(sourceRoot, "uploads.json"), "utf8"));
const selector = process.argv[2] ?? "all";
const staged = path.join(root, "tgcloud/endpoints/qwenStaging.js");
const cli = path.join(root, "node_modules/@tgcloud/cli/bin/tgcloud.js");
const markerPath = path.join(sourceRoot, "uploaded.json");
const markers = fs.existsSync(markerPath) ? JSON.parse(fs.readFileSync(markerPath, "utf8")) : {};
const chosen = records.filter(r => selector === "all" || r.name.includes(selector));
const parts = fs.readdirSync(path.dirname(staged))
  .filter(name => /^qwenStagingPart\d+\.js$/.test(name))
  .map(name => path.join(path.dirname(staged), name));
function clearParts() {
  for (const file of parts) fs.rmSync(file, { force: true });
  parts.length = 0;
}
try {
  const batches = [];
  let batch = [], batchBytes = 0;
  for (const record of chosen.filter(r => markers[r.name] !== `chunked1:${r.sha256}`)) {
    const size = fs.statSync(record.path).size;
    if (batch.length && batchBytes + size > 8 * 1024 * 1024) {
      batches.push(batch);
      batch = []; batchBytes = 0;
    }
    batch.push(record); batchBytes += size;
  }
  if (batch.length) batches.push(batch);
  for (const batch of batches) {
    clearParts();
    const imports = [], entries = [];
    for (const record of batch) {
      const bytes = fs.readFileSync(record.path);
      const entry = { ...record, length: bytes.length };
      delete entry.path;
      const names = [];
      for (let offset = 0; offset < bytes.length; offset += 512 * 1024) {
        const i = parts.length;
        const name = `qwenStagingPart${i}`;
        const file = path.join(root, `tgcloud/endpoints/${name}.js`);
        fs.writeFileSync(file, `export default ${JSON.stringify(bytes.subarray(offset, offset + 512 * 1024).toString("base64"))};\n`);
        parts.push(file);
        imports.push(`import p${i} from "./${name}.js";`);
        names.push(`p${i}`);
      }
      entries.push(`{...${JSON.stringify(entry)},parts:[${names.join(",")}]}`);
    }
    // Decode each independently: base64 padding exists at each chunk boundary.
    fs.writeFileSync(staged, `${imports.join("\n")}\nexport default [${entries.join(",")}];\n`);
    let result;
    for (let attempt = 0; attempt < 5; attempt++) {
      result = spawnSync(process.execPath, [cli, "run", "endpoints/qwenUpload"], {
        cwd: root, encoding: "utf8", timeout: 180000, maxBuffer: 1024 * 1024,
      });
      if (result.status === 0) break;
      const retry = `${result.stdout}\n${result.stderr}`.match(/retry after (\d+)/i);
      if (!retry || attempt === 4) break;
      const wait = Number(retry[1]) + 1;
      console.log(`Platform rate limit; waiting ${wait}s before retrying batch`);
      await new Promise(resolve => setTimeout(resolve, wait * 1000));
    }
    if (result.status !== 0) {
      console.error(result.stdout, result.stderr);
      throw new Error(`Upload failed: ${batch.map(r => r.name).join(", ")}`);
    }
    for (const record of batch) {
      console.log(`${record.name}: ${fs.statSync(record.path).size} bytes uploaded`);
      markers[record.name] = `chunked1:${record.sha256}`;
    }
    fs.writeFileSync(markerPath, JSON.stringify(markers, null, 2));
  }
} finally {
  fs.writeFileSync(staged, "export default [];\n");
  clearParts();
}
