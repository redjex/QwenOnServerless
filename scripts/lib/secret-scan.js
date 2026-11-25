// Shared secret scanner: everything that leaves this machine passes through here.
//
// Used by scripts/sync.js (before a local commit is sent) and by
// scripts/watch-agent.js (before a change made in the cloud sandbox is accepted).
// The rules live in one place on purpose: a new pattern must not have to be
// added twice.

import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

// First block: an actual credential. Second block: a local file holding one.
export const SECRET_PATTERNS = [
  ["токен Telegram Serverless CLI", /\bapp\d+:[A-Za-z0-9_-]{20,}/],
  ["токен Telegram-бота", /\bbot\d+:[A-Za-z0-9_-]{30,}/],
  ["ключ вида OpenAI", /\bsk-[A-Za-z0-9_-]{20,}/],
  ["токен GitHub", /\bgh[pousr]_[A-Za-z0-9]{30,}/],
  ["ключ ML AI Code", /\bmlai_sk_[A-Za-z0-9]{20,}/],
  ["ключ AWS", /\bAKIA[0-9A-Z]{16}\b/],
  ["блок приватного ключа", /-----BEGIN[A-Z ]*PRIVATE KEY-----/],
];

export const FORBIDDEN_PATHS = [
  /(^|\/)\.tgcloud\//,
  /(^|\/)node_modules\//,
  /(^|\/)\.env($|\.)/,
  /(^|\/)credentials$/,
  /qwen-?staging/i,
  /\.(pem|key|p12|pfx)$/,
];

// Не секрет, но и не для публичного репозитория: отчёты и журналы уезжают в
// открытый доступ, поэтому хэндл бота и ссылки на него скрываются целиком.
export const IDENTITY_PATTERNS = [
  ["хэндл бота", /\b[A-Za-z0-9_]{5,}_bot\b/g],
  ["ссылка t.me", /t\.me\/[A-Za-z0-9_]+/g],
];

const MAX_SCAN_BYTES = 4 * 1024 * 1024;

export function maskValue(value) {
  return value.length <= 8 ? "*".repeat(value.length) : `${value.slice(0, 6)}${"*".repeat(value.length - 6)}`;
}

// Replaces every secret-looking fragment in free text, so logs and reports can
// be written or committed safely.
export function redactText(text) {
  let result = String(text);
  for (const [label, pattern] of SECRET_PATTERNS) {
    result = result.replace(new RegExp(pattern.source, "g"), match => `${maskValue(match)}(${label})`);
  }
  for (const [label, pattern] of IDENTITY_PATTERNS) {
    result = result.replace(new RegExp(pattern.source, "g"), `<скрыто: ${label}>`);
  }
  return result;
}

// Scans existing files under `root`. `paths` are relative to `root` and may use
// either slash. Returns [path, reason] pairs, so callers can decide the verdict.
export function scanPaths(root, paths) {
  const findings = [];
  for (const path of paths) {
    const normalized = path.replace(/\\/g, "/");
    if (FORBIDDEN_PATHS.some(pattern => pattern.test(normalized))) {
      findings.push([path, "файл не должен попадать в репозиторий"]);
      continue;
    }
    const absolute = resolve(root, path);
    if (!existsSync(absolute)) continue;
    const stats = statSync(absolute);
    if (!stats.isFile() || stats.size === 0 || stats.size > MAX_SCAN_BYTES) continue;
    const bytes = readFileSync(absolute);
    if (bytes.includes(0)) continue; // двоичный файл
    const text = bytes.toString("utf8");
    for (const [label, pattern] of SECRET_PATTERNS) {
      const match = text.match(pattern);
      if (match) {
        findings.push([path, `${label}: ${maskValue(match[0])}`]);
        break;
      }
    }
  }
  return findings;
}
