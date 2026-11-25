"""Show confirmed Qwen upload progress by bytes; no Telegram credentials needed."""
import argparse
import json
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--watch", action="store_true", help="Refresh until upload finishes")
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[1] / "models" / "packed"
records = json.loads((root / "uploads.json").read_text(encoding="utf-8"))
sizes = {r["name"]: pathlib.Path(r["path"]).stat().st_size for r in records}
total = sum(sizes.values())

while True:
    try:
        markers = json.loads((root / "uploaded.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        if args.watch:
            time.sleep(0.2)
            continue
        raise SystemExit("Маркер загрузки сейчас недоступен, повтори команду.")
    confirmed = [r for r in records if markers.get(r["name"]) == f"chunked1:{r['sha256']}"]
    done = sum(sizes[r["name"]] for r in confirmed)
    percent = 100 * done / total if total else 100
    filled = int(40 * done / total) if total else 40
    bar = "#" * filled + "-" * (40 - filled)
    line = (
        f"Загрузка [{bar}] {percent:6.2f}% | осталось {100 - percent:.2f}%"
        f" | {done / 1048576:.2f}/{total / 1048576:.2f} МиБ"
        f" | {len(confirmed)}/{len(records)} файлов"
    )
    print(("\r" if args.watch else "") + line, end="" if args.watch else "\n", flush=True)
    if not args.watch or done == total:
        if args.watch:
            print()
        break
    time.sleep(1)
