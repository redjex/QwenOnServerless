// Two-way sync between this working copy and the shared GitHub remote, which is
// also the checkout the ML AI Code cloud machine works in.
//
// Flow: scan for secrets -> pull --rebase --autostash -> commit (signed) -> push.
// The script never force-pushes and never resolves conflicts on its own: on a
// conflict it stops, undoes the interrupted rebase and prints what to do next.
//
// Branch main on GitHub accepts signed commits only (repository ruleset), and
// the signing key lives on one machine, so pushes from anywhere else are
// rejected by the server. This script checks the local signing setup first and
// refuses to make a commit GitHub would not accept.
//
// Usage:
//   node scripts/sync.js                 full cycle
//   node scripts/sync.js -m "message"    own commit message
//   node scripts/sync.js --pull-only     only take remote changes
//   node scripts/sync.js --push-only     only send local changes
//   node scripts/sync.js --dry-run       report what would happen, change nothing

import { existsSync, readFileSync, statSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { hostname } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

const argv = process.argv.slice(2);
const hasFlag = name => argv.includes(name);
const option = name => (argv.indexOf(name) >= 0 ? argv[argv.indexOf(name) + 1] ?? null : null);

const dryRun = hasFlag("--dry-run");
const pullOnly = hasFlag("--pull-only");
const pushOnly = hasFlag("--push-only");
const customMessage = option("-m") ?? option("--message");
const remote = "origin";

function log(text) {
  process.stdout.write(`${text}\n`);
}

function fail(text) {
  process.stderr.write(`${text}\n`);
  process.exit(1);
}

if (pullOnly && pushOnly) fail("--pull-only и --push-only нельзя использовать вместе.");

// ---------------------------------------------------------------------------
// Guards: what must never leave this machine.
// ---------------------------------------------------------------------------

// First block: an actual credential. Second block: a local file holding one.
const SECRET_PATTERNS = [
  ["токен Telegram Serverless CLI", /\bapp\d+:[A-Za-z0-9_-]{20,}/],
  ["токен Telegram-бота", /\bbot\d+:[A-Za-z0-9_-]{30,}/],
  ["ключ вида OpenAI", /\bsk-[A-Za-z0-9_-]{20,}/],
  ["токен GitHub", /\bgh[pousr]_[A-Za-z0-9]{30,}/],
  ["ключ ML AI Code", /\bmlai_sk_[A-Za-z0-9]{20,}/],
  ["ключ AWS", /\bAKIA[0-9A-Z]{16}\b/],
  ["блок приватного ключа", /-----BEGIN[A-Z ]*PRIVATE KEY-----/],
];

const FORBIDDEN_PATHS = [
  /(^|\/)\.tgcloud\//,
  /(^|\/)node_modules\//,
  /(^|\/)\.env($|\.)/,
  /(^|\/)credentials$/,
  /qwen-?staging/i,
  /\.(pem|key|p12|pfx)$/,
];

const MAX_SCAN_BYTES = 4 * 1024 * 1024;

function mask(value) {
  return value.length <= 8 ? "*".repeat(value.length) : `${value.slice(0, 6)}${"*".repeat(value.length - 6)}`;
}

// ---------------------------------------------------------------------------
// git plumbing.
// ---------------------------------------------------------------------------

function run(command, args, options = {}) {
  const { cwd, trim = true } = options;
  const result = spawnSync(command, args, {
    cwd, encoding: "utf8", maxBuffer: 64 * 1024 * 1024,
  });
  if (result.error) fail(`Не удалось запустить ${command}: ${result.error.message}`);
  const stdout = trim ? (result.stdout ?? "").trim() : result.stdout ?? "";
  const stderr = (result.stderr ?? "").trim();
  return { status: result.status, stdout, stderr, output: `${stdout}\n${stderr}` };
}

const probe = run("git", ["rev-parse", "--show-toplevel"]);
if (probe.status !== 0) fail("Это не git-репозиторий. Запустите скрипт внутри проекта.");
const root = probe.stdout;

function git(args, options = {}) {
  const result = run("git", args, { cwd: root, ...options });
  const { check = true } = options;
  if (check && result.status !== 0) {
    fail(`git ${args.join(" ")} завершился с ошибкой:\n${result.output.trim()}`);
  }
  return result;
}

const gitPath = relative => {
  const raw = git(["rev-parse", "--git-path", relative], { check: false }).stdout;
  return raw ? (isAbsolute(raw) ? raw : resolve(root, raw)) : resolve(root, ".git", relative);
};

// ---------------------------------------------------------------------------
// Preflight.
// ---------------------------------------------------------------------------

const remotes = git(["remote"]).stdout.split("\n").filter(Boolean);
if (!remotes.includes(remote)) {
  fail([
    `Удалённый репозиторий «${remote}» не настроен.`,
    "Создайте приватный репозиторий и подключите его один раз:",
    "  gh repo create <owner>/<name> --private --source=. --remote=origin",
  ].join("\n"));
}

const remoteUrl = git(["remote", "get-url", remote]).stdout;
const branch = git(["branch", "--show-current"]).stdout;
if (!branch) fail("Отсоединённый HEAD: сначала переключитесь на ветку (git switch -c <branch>).");

if (["rebase-merge", "rebase-apply"].some(name => existsSync(gitPath(name)))) {
  fail("В репозитории незавершённый rebase. Завершите его (git rebase --continue) или отмените (git rebase --abort).");
}
if (existsSync(gitPath("MERGE_HEAD"))) {
  fail("В репозитории незавершённое слияние. Завершите его (git commit) или отмените (git merge --abort).");
}

const status = git(["status", "--porcelain", "-z", "--untracked-files=all"], { trim: false }).stdout;
const changed = [];
{
  const entries = status.split("\0").filter(Boolean);
  for (let index = 0; index < entries.length; index++) {
    const code = entries[index].slice(0, 2);
    changed.push(entries[index].slice(3));
    if (code.startsWith("R") || code.startsWith("C")) index++; // пропускаем старое имя в записи переименования
  }
}

function scanForSecrets(paths) {
  const findings = [];
  for (const path of paths) {
    const normalized = path.replace(/\\/g, "/");
    const forbidden = FORBIDDEN_PATHS.find(pattern => pattern.test(normalized));
    if (forbidden) {
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
        findings.push([path, `${label}: ${mask(match[0])}`]);
        break;
      }
    }
  }
  return findings;
}

const findings = scanForSecrets(changed);
if (findings.length > 0) {
  fail([
    "Проверка остановлена: в коммит попали бы секреты. Ничего не отправлено.",
    ...findings.map(([path, why]) => `  ${path} — ${why}`),
    "Удалите их из файлов или добавьте в .gitignore и запустите синхронизацию снова.",
  ].join("\n"));
}

log(`Репозиторий: ${root}`);
log(`Ветка: ${branch}  ->  ${remoteUrl}`);
log(`Изменённых файлов: ${changed.length}`);

// ---------------------------------------------------------------------------
// Pull.
// ---------------------------------------------------------------------------

const remoteBranch = git(["ls-remote", "--exit-code", "--heads", remote, branch], { check: false }).status === 0;
let pulled = 0;

if (pushOnly) {
  log("Шаг «забрать» пропущен (--push-only).");
} else if (!remoteBranch) {
  log(`Ветки ${branch} на ${remote} пока нет: это первый push.`);
} else {
  if (dryRun) {
    log("--dry-run: пропускаю pull (посмотреть, что придёт: git fetch && git log HEAD..@{u}).");
  } else {
    const before = git(["rev-parse", "HEAD"]).stdout;
    const result = git(["pull", "--rebase", "--autostash", remote, branch], { check: false });
    if (result.status !== 0) {
      if (["rebase-merge", "rebase-apply"].some(name => existsSync(gitPath(name)))) {
        git(["rebase", "--abort"], { check: false });
      }
      fail([
        "Не удалось применить изменения с удалённой стороны: конфликт.",
        "Локальные правки возвращены на место, ничего не потеряно.",
        "",
        "Разрешите конфликт вручную:",
        `  git pull --rebase --autostash ${remote} ${branch}`,
        "  # правите файлы, затем: git add <файлы> && git rebase --continue",
        "  npm run sync",
        "",
        result.output.trim(),
      ].join("\n"));
    }
    pulled = Number(git(["rev-list", "--count", `${before}..HEAD`]).stdout) || 0;
    log(`Забрано коммитов: ${pulled}`);
  }
}

// ---------------------------------------------------------------------------
// Commit and push.
// ---------------------------------------------------------------------------

if (pullOnly) {
  log("Шаг «отправить» пропущен (--pull-only).");
} else {
  // main принимает только подписанные коммиты, поэтому без настроенной подписи
  // сервер отклонит push. Проверяем заранее, чтобы сообщение было понятным.
  if (changed.length > 0 && !dryRun) {
    const format = git(["config", "--get", "gpg.format"], { check: false }).stdout;
    const signingKey = git(["config", "--get", "user.signingkey"], { check: false }).stdout;
    const signing = git(["config", "--get", "commit.gpgsign"], { check: false }).stdout === "true";
    if (format !== "ssh" || !signingKey || !signing) {
      fail([
        "Подпись коммитов не настроена, а ветка main принимает только подписанные коммиты.",
        "Ничего не закоммичено. Включите подпись один раз в этом репозитории:",
        "  git config --local gpg.format ssh",
        '  git config --local user.signingkey "$HOME/.ssh/id_ed25519.pub"',
        "  git config --local commit.gpgsign true",
        "и добавьте этот же ключ на GitHub: Settings -> SSH and GPG keys ->",
        "New SSH key -> Key type: Signing Key.",
      ].join("\n"));
    }
  }

  if (changed.length > 0) {
    const stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
    const message = customMessage ?? `sync: ${stamp} (${hostname()})`;
    if (dryRun) {
      log(`--dry-run: закоммитил бы ${changed.length} файл(ов) как «${message}».`);
    } else {
      git(["add", "-A", "--", "."]);
      git(["commit", "-m", message]);
      log(`Коммит: ${message}`);
      // %G? == N значит подписи нет: такой коммит GitHub отклонит на push.
      const signature = git(["log", "-1", "--format=%G?"]).stdout;
      if (signature === "N") {
        fail("Коммит вышел без подписи, его отклонят на GitHub. Проверьте настройку подписи выше и запустите синхронизацию снова.");
      }
      log(`Подпись коммита: ${signature === "G" ? "проверена" : `статус ${signature}`}`);
    }
  } else {
    log("Локальных изменений нет, коммит не нужен.");
  }

  // Локальные коммиты могли существовать до этого запуска и просто переехать поверх
  // удалённых при rebase, поэтому смотрим на расхождение, а не на список изменений.
  let ahead = 0;
  if (remoteBranch) {
    ahead = Number(git(["rev-list", "--count", `${remote}/${branch}..HEAD`], { check: false }).stdout) || 0;
  }

  if (dryRun) {
    if (remoteBranch) log(`--dry-run: отправил бы ${ahead} коммит(ов) в ${remote}/${branch}.`);
    else log(`--dry-run: создал бы ветку ${branch} на ${remote}.`);
  } else if (!remoteBranch || ahead > 0) {
    const push = git(["push", "-u", remote, branch], { check: false });
    if (push.status !== 0) {
      fail([
        "Push отклонён (на удалённой стороне появились новые коммиты).",
        "Запустите npm run sync снова: скрипт сначала заберёт изменения и повторит отправку.",
        "",
        push.output.trim(),
      ].join("\n"));
    }
    log(`Отправлено в ${remote}/${branch}.`);
  } else {
    log("Отправлять нечего.");
  }
}

log("Готово. На облачной машине ML AI Code сделайте pull той же ветки (или попросите агента запушить изменения пул-реквестом).");
