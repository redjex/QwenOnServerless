import fs from "node:fs";
const bytes = fs.readFileSync(new URL("../models/qwen-kernel.wasm", import.meta.url));
fs.writeFileSync(new URL("../tgcloud/lib/qwen-wasm.js", import.meta.url),
  `// Generated from scripts/qwen-kernel.rs; rebuild instructions in QWEN.md.\nexport default ${JSON.stringify(bytes.toString("base64"))};\n`);
