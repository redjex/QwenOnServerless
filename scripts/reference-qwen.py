"""Independent Transformers/PyTorch reference for the quantized JS/WASM engine."""
import os
os.environ["USE_TF"] = "0"
import json
import pathlib
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

root = pathlib.Path(__file__).resolve().parents[1] / "models"
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
texts = ["Привет! Как дела?", "Hello, world!", "Сегодня 2026 год.\n", "Привет!"]
vectors = [{"text": text, "ids": tokenizer.encode(text),
            "chat": tokenizer.apply_chat_template([{"role": "user", "content": text}],
                        tokenize=True, add_generation_prompt=True, enable_thinking=False)} for text in texts]
print("Loading reference", flush=True)
model = Qwen3ForCausalLM.from_pretrained(root, local_files_only=True,
    dtype=torch.float32, attn_implementation="eager").eval()
with torch.no_grad():
    for name, weight in model.named_parameters():
        if weight.ndim == 2:
            for start in range(0, weight.shape[0], 4096):
                part = weight[start:start+4096]
                if name == "model.embed_tokens.weight":
                    scale = part.abs().amax(dim=1, keepdim=True).div(127).clamp(min=1e-12)
                    part.copy_(part.div(scale).round().clamp(-127,127).mul(scale))
                else:
                    groups = part.reshape(-1, 32)
                    scale = groups.abs().amax(dim=1, keepdim=True).div(7).clamp(min=1e-12)
                    groups.copy_(groups.div(scale).round().clamp(-7,7).mul(scale))
    print("Running quantized reference", flush=True)
    output = model(torch.tensor([[198, 198]]), output_hidden_states=True)
    layer = output.hidden_states[1][0].float().tolist()
    top = torch.topk(output.logits[0, -1], 10)
    reference = {"tokenizer": vectors, "layer0": layer,
                 "diagnosticTop": {"ids": top.indices.tolist(), "logits": top.values.tolist()}}
    prompt = vectors[-1]["chat"]
    output = model(torch.tensor([prompt]))
    top = torch.topk(output.logits[0, -1], 10)
    reference["chatTop"] = {"ids": top.indices.tolist(), "logits": top.values.tolist()}
    generated = model.generate(torch.tensor([prompt]), max_new_tokens=24,
        do_sample=False, repetition_penalty=1.08, eos_token_id=151645, pad_token_id=151643)
    reference["chatSample"] = tokenizer.decode(generated[0, len(prompt):], skip_special_tokens=True)
(root / "reference.json").write_text(json.dumps(reference, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"diagnosticTop": reference["diagnosticTop"], "chatTop": reference["chatTop"],
                  "sample": reference["chatSample"]}, ensure_ascii=True), flush=True)
