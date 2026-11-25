"""Prepare int8 embeddings and block-int4 layers from official BF16 tensors."""
import json
import pathlib
import struct
import hashlib
import numpy as np
from safetensors import safe_open

root = pathlib.Path(__file__).resolve().parents[1] / "models"
packed = root / "packed"
packed.mkdir(exist_ok=True)
manifest = {
    "format": 1, "quantization": "int8-embeddings-int4-block32-transformer",
    "source": json.loads((root / "source.json").read_text()),
    "config": json.loads((root / "config.json").read_text()),
    "embeddingChunkRows": 4096, "blocks": {}, "norms": {},
}
uploads = []

def write_block(name, matrix):
    rows, cols = matrix.shape
    if name.startswith("embed."):
        scales = np.maximum(np.abs(matrix).max(axis=1) / 127.0, 1e-12).astype("<f4")
        q = np.clip(np.rint(matrix / scales[:, None]), -127, 127).astype(np.int8)
        data = struct.pack("<II", rows, cols) + scales.tobytes() + q.tobytes()
    else:
        groups = matrix.reshape(-1, 32)
        scales = np.maximum(np.abs(groups).max(axis=1) / 7.0, 1e-12).astype("<f4")
        q = (np.clip(np.rint(groups / scales[:, None]), -7, 7) + 8).astype(np.uint8).ravel()
        packed_q = q[0::2] | (q[1::2] << 4)
        data = struct.pack("<III", rows, cols, 4) + scales.tobytes() + packed_q.tobytes()
    path = packed / (name + ".bin")
    path.write_bytes(data)
    manifest["blocks"][name] = {"rows": rows, "cols": cols, "bytes": len(data)}
    uploads.append({"kind": "block", "name": name, "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest()})

with safe_open(root / "model.safetensors", framework="pt", device="cpu") as weights:
    for name in weights.keys():
        if name == "lm_head.weight":
            # The official checkpoint stores both copies despite tied embeddings.
            a, b = weights.get_slice(name), weights.get_slice("model.embed_tokens.weight")
            for start in range(0, a.get_shape()[0], 4096):
                if not np.array_equal(a[start:start+4096].float().numpy(), b[start:start+4096].float().numpy()):
                    raise ValueError("Output weights are not tied to the embeddings")
            continue
        if name.endswith("norm.weight"):
            manifest["norms"][name] = weights.get_tensor(name).float().numpy().tolist()
            continue
        if name == "model.embed_tokens.weight":
            tensor = weights.get_slice(name)
            for offset in range(0, tensor.get_shape()[0], 4096):
                write_block(f"embed.{offset // 4096}", tensor[offset:offset+4096].float().numpy())
        else:
            matrix = weights.get_tensor(name).float().numpy()
            if matrix.ndim != 2:
                raise ValueError(f"Unsupported tensor {name} {matrix.shape}")
            write_block(name, matrix)
        print(name, flush=True)

tokenizer = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
vocab = tokenizer["model"]["vocab"]
tokens = [""] * (max(vocab.values()) + 1)
for token, index in vocab.items():
    tokens[index] = token
token_data = {
    "tokens": tokens, "merges": tokenizer["model"]["merges"],
    "added": tokenizer["added_tokens"],
}
for name, value in (("manifest", manifest), ("tokenizer", token_data)):
    path = packed / (name + ".json")
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(data)
    uploads.append({"kind": "meta", "name": name, "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest()})
(packed / "uploads.json").write_text(json.dumps(uploads, indent=2), encoding="utf-8")
print(json.dumps({"blocks": len(manifest["blocks"]),
                  "weightBytes": sum(b["bytes"] for b in manifest["blocks"].values()),
                  "tokenizerBytes": (packed / "tokenizer.json").stat().st_size}))
