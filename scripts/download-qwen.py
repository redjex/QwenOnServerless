"""Download official artifacts at a pinned revision, without touching credentials."""
import json
import pathlib
import urllib.request

root = pathlib.Path(__file__).resolve().parents[1] / "models"
source = root / "source.json"
if source.exists():
    revision = json.loads(source.read_text())["revision"]
else:
    with urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3-0.6B") as response:
        revision = json.load(response)["sha"]
manifest = {"repository": "Qwen/Qwen3-0.6B", "revision": revision}
source.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "LICENSE"):
    target = root / name
    if target.exists():
        print(f"Already present: {name}", flush=True)
        continue
    url = f"https://huggingface.co/Qwen/Qwen3-0.6B/resolve/{revision}/{name}"
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while data := response.read(4 * 1024 * 1024):
            output.write(data)
    temporary.replace(target)
    print(f"{name}: {target.stat().st_size} bytes", flush=True)
