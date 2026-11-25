// Watches this repository for changes made by the ML AI Code cloud machine (a
// new branch, a new push, a pull request), verifies them and hands the verdict
// back to the sandbox through the only channel both sides share: GitHub.
//
// What "a signal" means here: the head of a branch on origin other than main.
// Each head that was not verified before is checked in a throwaway worktree, so
// the owner's working copy is never touched.
//
// Checks:
//   secret-scan           changed files against the shared secret patterns
//   npm-test              npm test in the throwaway worktree
//   demo                  npm run demo (model smoke run) in the worktree
//   telegram-serverless   real calls in the deployed Telegram environment
//                         (runtimeCheck init -> train -> read -> cleanup)
//
// The verdict goes back as .mlai/verification.json committed onto that branch
// and, when the branch has a pull request, as a comment on the pull request.
//
// Usage:
//   node scripts/watch-agent.js                    all checks, one pass
//   node scripts/watch-agent.js --local-only       skip the Telegram stage
//   node scripts/watch-agent.js --telegram-only    only the Telegram stage
//   node scripts/watch-agent.js --watch 60         check every 60 seconds
//   node scripts/watch-agent.js --branch agent/x   only this branch
//   node scripts/watch-agent.js --list             show candidates, verify nothing
//   node scripts/watch-agent.js --no-report-commit verify, but do not push the report
//   node scripts/watch-agent.js --dry-run          do not push or comment

import { appendFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { redactText, scanPaths } from "./lib/secret-scan.js";

const argv = process.argv.slice(2);
const hasFlag = name => argv.includes(name);
const option = name => (argv.indexOf(name) >= 0 ? argv[argv.indexOf(name) + 1] ?? null : null);

const dryRun = hasFlag("--dry-run");
const listOnly = hasFlag("--list");
const localOnly = hasFlag("--local-only");
const telegramOnly = hasFlag("--telegram-only");
const commitReport = !hasFlag("--no-report-commit");
const onlyBranch = option("--branch");
const watchSeconds = hasFlag("--watch") ? Number(option("--watch") ?? 60) || 60 : null;
const endpoint = option("--endpoint") ?? "runtimeCheck";
const customInput = option("--input");

const remote = "origin";
const mainBranch = "main";
const REPORT_PATH = ".mlai/verification.json";
const REPORT_SUBJECT = "verify: report for";
const STATE_FILE = ".agent-watch.json";
const LOG_FILE = ".agent-watch.log";
const MAX_CHECK_OUTPUT = 3000;

function log(text) {
  process.stdout.write(`${text}\n`);
}

function fail(text) {
  process.stderr.write(`${text}\n`);
  process.exit(1);
}

if (telegramOnly && localOnly) fail("--local-only и --telegram-only нельзя использовать вместе.");

function parseInputOption() {
  if (!customInput) return null;
  try {
    return JSON.parse(customInput);
  } catch (error) {
    fail(`--input должен быть JSON: ${error.message}`);
  }
}

const customInputValue = parseInputOption();

// ---------------------------------------------------------------------------
// git plumbing.
// ---------------------------------------------------------------------------

function run(command, args, options = {}) {
  const { cwd, timeout = 300_000, trim = true, shell = false } = options;
  const result = spawnSync(command, args, {
    cwd, shell, encoding: "utf8", timeout, maxBuffer: 64 * 1024 * 1024,
  });
  const stdout = trim ? (result.stdout ?? "").trim() : result.stdout ?? "";
  const stderr = (result.stderr ?? "").trim();
  const timeoutNote = result.error?.code === "ETIMEDOUT" ? `команда превысила ${timeout} мс` : (result.error?.message ?? "");
  return {
    status: result.status, stdout, stderr, error: result.error,
    output: [stdout, stderr, timeoutNote].filter(Boolean).join("\n"),
  };
}

const probe = run("git", ["rev-parse", "--show-toplevel"]);
if (probe.status !== 0) fail("Это не git-репозиторий. Запустите скрипт внутри проекта.");
const root = probe.stdout;

function git(args, options = {}) {
  const { check = true, ...rest } = options;
  const result = run("git", args, { cwd: root, ...rest });
  if (check && result.status !== 0) {
    fail(`git ${args.join(" ")} завершился с ошибкой:\n${result.output}`);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Local state: which heads were already verified.
// ---------------------------------------------------------------------------

const statePath = join(root, STATE_FILE);
const logPath = join(root, LOG_FILE);
const state = existsSync(statePath) ? JSON.parse(readFileSync(statePath, "utf8")) : { schema: 1, verified: {} };
state.verified ??= {};

function saveState() {
  state.updated_at = new Date().toISOString();
  writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`);
}

function appendLog(entry) {
  appendFileSync(logPath, `${JSON.stringify(entry)}\n`);
}

// ---------------------------------------------------------------------------
// Output sanitising: logs and reports are written to a public repository, so
// absolute local paths and anything secret-shaped must not survive.
// ---------------------------------------------------------------------------

function sanitize(text) {
  let result = redactText(text ?? "");
  for (const path of [root, tmpdir()]) {
    result = result.split(path).join(".");
    result = result.split(path.replace(/\\/g, "/")).join(".");
  }
  const lines = result.split("\n").map(line => line.trimEnd());
  while (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  const joined = lines.join("\n");
  return joined.length > MAX_CHECK_OUTPUT ? `${joined.slice(0, MAX_CHECK_OUTPUT)}\n… (обрезано)` : joined;
}

// ---------------------------------------------------------------------------
// Candidates: branch heads on origin, except main itself.
// ---------------------------------------------------------------------------

function fetchRemote() {
  const result = git(["fetch", "--prune", "--quiet", remote], { check: false, timeout: 120_000 });
  if (result.status !== 0) fail(`Не удалось получить состояние ${remote}:\n${result.output}`);
}

const isOurReport = subject => (subject ?? "").startsWith(REPORT_SUBJECT);

function collectCandidates() {
  // Символические ссылки (origin/HEAD) пропускаем: их цель — уже обычная ветка.
  const refs = git([
    "for-each-ref", "--format=%(refname)|%(objectname)|%(symref)|%(contents:subject)", `refs/remotes/${remote}`,
  ]).stdout.split("\n").filter(Boolean);

  const prefix = `refs/remotes/${remote}/`;
  const candidates = [];
  for (const line of refs) {
    const [ref, sha, symref, ...subjectParts] = line.split("|");
    if (symref || !ref.startsWith(prefix)) continue;
    const branch = ref.slice(prefix.length);
    const subject = subjectParts.join("|");
    if (branch === "HEAD" || branch === mainBranch) continue;
    if (onlyBranch && branch !== onlyBranch) continue;
    candidates.push({ branch, sha, subject, ours: isOurReport(subject) });
  }
  return candidates;
}

function pullRequestsByBranch() {
  const result = run("gh", [
    "pr", "list", "--state", "open", "--limit", "50",
    "--json", "number,headRefName,headRefOid,title,url",
  ], { cwd: root, timeout: 60_000 });
  if (result.status !== 0 || !result.stdout) return new Map();
  try {
    return new Map(JSON.parse(result.stdout).map(pr => [pr.headRefName, pr]));
  } catch {
    return new Map();
  }
}

// ---------------------------------------------------------------------------
// Throwaway worktree: checks run on the branch itself, never in the owner's
// working copy.
// ---------------------------------------------------------------------------

function inWorktree(sha, label, work) {
  const path = join(tmpdir(), `qwen-${label}-${Date.now()}`);
  const added = git(["worktree", "add", "--detach", path, sha], { check: false, timeout: 120_000 });
  try {
    if (added.status !== 0) return { ok: false, output: added.output };
    return { ok: true, value: work(path) };
  } finally {
    git(["worktree", "remove", "--force", path], { check: false });
    git(["worktree", "prune"], { check: false });
    if (existsSync(path)) rmSync(path, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// Checks.
// ---------------------------------------------------------------------------

function changedFiles(worktree, sha) {
  const base = git(["merge-base", `refs/remotes/${remote}/${mainBranch}`, sha], { check: false }).stdout;
  const result = base
    ? git(["diff", "--name-only", base, sha], { check: false })
    : git(["ls-tree", "-r", "--name-only", sha], { check: false });
  return result.stdout.split("\n").map(line => line.trim()).filter(Boolean)
    .filter(path => existsSync(resolve(worktree, path)));
}

function runNpm(args, cwd) {
  // npm на Windows — это npm.cmd, который Node отказывается запускать без shell;
  // аргументы здесь всегда наши литералы, поэтому строка команды безопасна.
  const result = run(["npm", ...args].join(" "), [], { cwd, shell: true });
  return { ok: result.status === 0, output: sanitize(result.output) };
}

function demoMetrics(output) {
  const match = output.match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    const parsed = JSON.parse(match[0]);
    return {
      steps: parsed.steps,
      firstTrainingLoss: parsed.firstTrainingLoss,
      lastTrainingLoss: parsed.lastTrainingLoss,
      sampleBytes: parsed.sample ? parsed.sample.length : null,
    };
  } catch {
    return null;
  }
}

// CLI печатает "> запрос", затем "< {json}  время" и служебные строки; JSON ищем
// после последней строки-ответа, иначе разбор зацепит эхо запроса.
function extractCliJson(output) {
  const lines = output.split("\n");
  const index = lines.map(line => line.trimStart().startsWith("< ")).lastIndexOf(true);
  if (index < 0) return null;
  const candidate = lines.slice(index).join("\n");
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    return JSON.parse(candidate.slice(start, end + 1));
  } catch {
    return null;
  }
}

// Real calls in the deployed Telegram environment. This validates the
// environment and the deployed code, not the branch under review: a branch is
// never published to the live bot from here.
function telegramChecks() {
  const call = input => run(
    "node", ["scripts/run-cloud.js", `endpoints/${endpoint}`, JSON.stringify(input)],
    { cwd: root, timeout: 300_000 },
  );
  const sequence = endpoint === "runtimeCheck" && !customInputValue
    ? [{ action: "init" }, { action: "train" }, { action: "read" }]
    : [customInputValue ?? {}];

  const steps = [];
  let ok = true;
  let metrics = null;
  try {
    for (const input of sequence) {
      const result = call(input);
      const parsed = extractCliJson(result.output);
      const stepOk = result.status === 0 && parsed !== null && parsed.error === undefined;
      steps.push({ action: input.action ?? "call", ok: stepOk, detail: sanitize(result.output).slice(0, 700) });
      if (!stepOk) ok = false;
      if (parsed && (input.action === "train" || input.action === "read" || input.action === undefined)) {
        metrics = {
          steps: parsed.steps,
          // Хэндл бота в отчёт не переносим: отчёт уезжает в публичный репозиторий.
          botReachable: Boolean(parsed.botUsername),
          meanLoss: parsed.progress?.meanLoss ?? null,
          checkpointCharacters: parsed.checkpointCharacters,
          sample: parsed.sample ? sanitize(parsed.sample) : null,
        };
      }
    }
  } catch (error) {
    ok = false;
    steps.push({ action: "call", ok: false, detail: sanitize(error.message) });
  } finally {
    // Служебная строка с отрицательным ID не должна остаться в базе.
    if (endpoint === "runtimeCheck") {
      const cleanup = call({ action: "cleanup" });
      const done = cleanup.status === 0;
      if (!done) {
        ok = false;
        steps.push({ action: "cleanup", ok: false, detail: sanitize(cleanup.output).slice(0, 400) });
      }
    }
  }
  return { ok, steps, metrics };
}

// ---------------------------------------------------------------------------
// Verification of one branch head.
// ---------------------------------------------------------------------------

function verify(candidate) {
  const { branch, sha, subject } = candidate;
  const short = sha.slice(0, 8);
  const started = Date.now();
  log(`--- ${branch} @ ${short}${subject ? ` «${subject}»` : ""}`);

  const checks = [];
  const prepared = inWorktree(sha, `verify-${short}`, worktree => {
    if (!telegramOnly) {
      const paths = changedFiles(worktree, sha);
      const findings = scanPaths(worktree, paths);
      checks.push({
        name: "secret-scan",
        ok: findings.length === 0,
        detail: findings.length === 0
          ? `проверено файлов: ${paths.length}`
          : findings.map(([path, why]) => `${path} — ${why}`).join("\n"),
      });

      const tests = runNpm(["test"], worktree);
      checks.push({ name: "npm-test", ok: tests.ok, detail: tests.output });

      const demo = runNpm(["run", "demo"], worktree);
      const metrics = demo.ok ? demoMetrics(demo.output) : null;
      checks.push({ name: "demo", ok: demo.ok, detail: demo.output, ...(metrics ? { metrics } : {}) });
    }

    if (!localOnly) {
      const telegram = telegramChecks();
      checks.push({
        name: "telegram-serverless",
        ok: telegram.ok,
        ...(telegram.metrics ? { metrics: telegram.metrics } : {}),
        detail: telegram.steps.map(step => `${step.action}: ${step.ok ? "ок" : "сбой"}\n${step.detail}`).join("\n\n"),
      });
    }
    return checks;
  });

  if (!prepared.ok) {
    checks.push({ name: "worktree", ok: false, detail: sanitize(prepared.output) });
  }

  const verdict = checks.length > 0 && checks.every(check => check.ok) ? "pass" : "fail";
  const failed = checks.filter(check => !check.ok).map(check => check.name);
  const report = {
    schema: 1,
    ref: branch,
    sha,
    short_sha: short,
    checked_at: new Date().toISOString(),
    duration_ms: Date.now() - started,
    verdict,
    checks,
    notes: [
      "Проверки secret-scan, npm-test и demo выполнены во временной копии именно этой ветки.",
      "Проверка telegram-serverless выполнена в развёрнутой версии кода в Telegram Serverless, а не в этой ветке: ветка в живого бота не публиковалась.",
      "Вердикт формируется на машине владельца; в main попадает только подписанный код с неё.",
    ],
  };

  // Отчёт уходит коммитом в ту же ветку: агент читает его после git pull.
  if (commitReport && !dryRun) {
    const pushed = inWorktree(sha, `report-${short}`, worktree => {
      mkdirSync(join(worktree, ".mlai"), { recursive: true });
      writeFileSync(join(worktree, REPORT_PATH), `${JSON.stringify(report, null, 2)}\n`);
      git(["add", "-f", "--", REPORT_PATH], { cwd: worktree });
      const committed = git(["commit", "-m", `${REPORT_SUBJECT} ${short} (${verdict})`], { cwd: worktree, check: false });
      if (committed.status !== 0) return null;
      const push = git(["push", remote, `HEAD:refs/heads/${branch}`], { cwd: worktree, check: false, timeout: 120_000 });
      if (push.status !== 0) {
        log(`  отчёт не попал в ветку: ${sanitize(push.output).split("\n").filter(Boolean).slice(-1)[0] ?? "push отклонён"}`);
        return null;
      }
      return git(["rev-parse", "HEAD"], { cwd: worktree }).stdout;
    });
    report.report_sha = pushed.ok ? pushed.value : null;
    if (report.report_sha) log(`  отчёт отправлен в ${branch} (${report.report_sha.slice(0, 8)})`);
  }

  const summary = verdict === "pass"
    ? `все проверки пройдены (${checks.map(check => check.name).join(", ")})`
    : `сбой: ${failed.join(", ")}`;

  log(`  вердикт: ${verdict} — ${summary}`);
  for (const check of checks) {
    log(`    ${check.ok ? "ок  " : "сбой"} ${check.name}${check.metrics ? ` ${JSON.stringify(check.metrics)}` : ""}`);
  }

  appendLog({ at: report.checked_at, ref: branch, sha, verdict, failed, report_sha: report.report_sha ?? null });
  return { branch, sha, subject, verdict, checks, summary, report_sha: report.report_sha ?? null };
}

// ---------------------------------------------------------------------------
// Feedback to the sandbox: a comment on the pull request, when there is one.
// ---------------------------------------------------------------------------

function reportBody(result, report) {
  const lines = [
    `## Проверка с машины владельца: **${result.verdict === "pass" ? "пройдено" : "не пройдено"}**`,
    "",
    `Ветка \`${result.branch}\`, коммит \`${result.sha.slice(0, 8)}\`${result.subject ? `, «${result.subject}»` : ""}.`,
    "",
    "| Проверка | Результат |",
    "| --- | --- |",
    ...report.checks.map(check => `| ${check.name} | ${check.ok ? "ок" : "сбой"} |`),
    "",
  ];
  for (const check of report.checks.filter(item => !item.ok)) {
    lines.push(`<details><summary>${check.name}: что не так</summary>`, "", "```", check.detail.slice(0, 1500), "```", "", "</details>", "");
  }
  lines.push(
    "Полный отчёт: `.mlai/verification.json` в этой ветке (`git pull` и откройте файл).",
    "",
    "> `npm-test` и `demo` проверены на этой ветке, `telegram-serverless` — в развёрнутой версии в Telegram. " +
    "В `main` попадает только подписанный код с машины владельца.",
  );
  return lines.join("\n");
}

function commentOnPullRequest(pullRequest, result, report) {
  if (!pullRequest) {
    log(`  PR для ${result.branch} не найден: вердикт только в .mlai/verification.json ветки.`);
    return;
  }
  if (dryRun) {
    log(`--dry-run: комментарий в PR #${pullRequest.number} не отправлен.`);
    return;
  }
  const bodyPath = join(tmpdir(), `verify-report-${result.sha.slice(0, 8)}.md`);
  writeFileSync(bodyPath, reportBody(result, report));
  const commented = run("gh", ["pr", "comment", String(pullRequest.number), "--body-file", bodyPath], { cwd: root, timeout: 60_000 });
  if (commented.status !== 0) {
    log(`  комментарий в PR не отправлен: ${sanitize(commented.output).split("\n").filter(Boolean).slice(-1)[0] ?? ""}`);
  } else {
    log(`  комментарий отправлен в PR #${pullRequest.number}`);
  }
  rmSync(bodyPath, { force: true });
}

// ---------------------------------------------------------------------------
// One pass over every candidate.
// ---------------------------------------------------------------------------

function pass() {
  fetchRemote();
  const candidates = collectCandidates();

  if (candidates.length === 0) {
    log("Новых ветвей от облачной машины нет.");
    saveState();
    return { checked: 0, failed: 0 };
  }

  if (listOnly) {
    for (const candidate of candidates) {
      const known = state.verified[candidate.sha];
      const mark = candidate.ours ? " [наш отчёт]" : known ? ` [уже проверен: ${known.verdict}]` : " [новый]";
      log(`${candidate.branch} @ ${candidate.sha.slice(0, 8)}${mark}`);
    }
    return { checked: 0, failed: 0 };
  }

  const pullRequests = pullRequestsByBranch();
  let checked = 0;
  let failed = 0;

  for (const candidate of candidates) {
    const short = candidate.sha.slice(0, 8);

    if (candidate.ours) {
      log(`- ${candidate.branch} @ ${short}: наш отчёт, пропускаю.`);
      state.verified[candidate.sha] = { at: new Date().toISOString(), verdict: "report", branch: candidate.branch };
      saveState();
      continue;
    }
    if (state.verified[candidate.sha]?.verdict) {
      log(`- ${candidate.branch} @ ${short}: уже проверен, пропускаю.`);
      continue;
    }

    const result = verify(candidate);
    checked += 1;
    if (result.verdict === "fail") failed += 1;

    state.verified[candidate.sha] = {
      at: new Date().toISOString(), verdict: result.verdict, branch: candidate.branch,
      summary: result.summary, report_sha: result.report_sha ?? null,
    };
    // Отчёт в ветке — это новый head: помечаем проверенным, чтобы следующая
    // итерация не проверяла собственный отчёт.
    if (result.report_sha) {
      state.verified[result.report_sha] = { at: new Date().toISOString(), verdict: "report", branch: candidate.branch };
    }
    saveState();

    commentOnPullRequest(pullRequests.get(candidate.branch), result, { checks: result.checks });
  }

  saveState();
  return { checked, failed };
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

log(`Репозиторий: ${root}`);
log(`Проверки: ${telegramOnly ? "только Telegram Serverless" : localOnly ? "только локальные" : "локальные + Telegram Serverless"}${customInputValue ? `, свой input для ${endpoint}` : ""}`);
log(`Отчёт в ветку: ${commitReport ? "да" : "нет"}${dryRun ? ", режим --dry-run (без отправки отчёта и комментариев)" : ""}`);

if (watchSeconds) {
  log(`Режим наблюдения: каждые ${watchSeconds} с, остановка по Ctrl+C.`);
  for (;;) {
    try {
      const result = pass();
      log(`Цикл завершён: проверено ${result.checked}, сбоев ${result.failed}, ${new Date().toISOString()}`);
    } catch (error) {
      log(`Цикл прерван ошибкой: ${error.message}`);
    }
    await new Promise(wait => setTimeout(wait, watchSeconds * 1000));
  }
} else {
  const result = pass();
  log(`Готово: проверено ${result.checked}, сбоев ${result.failed}.`);
  process.exitCode = result.failed > 0 ? 1 : 0;
}
