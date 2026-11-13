"""Accuracy cost of restoring a prefix's KV from the FP8 CPU cache.

Teacher-forced on real text (wikitext-103 test): run a prefix through HF
transformers, pass its KV through the engine's own fp8_quantize/fp8_dequantize
(one scale per token and KV head, as a CPU-cache round trip does), then score
the next tokens against the untouched BF16 KV. Reports perplexity on both, the
KL divergence of the next-token distributions, top-1 agreement, and how many
greedy tokens match before the first divergence. As a noise floor, the same
metrics compare BF16 KV prefilled in 4096-token chunks (as the engine does)
against one-shot prefill: numerically different but equally valid BF16.

Run: python -m bench.fp8_kv_quality --model DIR --text PARQUET --out FILE.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_fp8_functions():
    path = Path(__file__).resolve().parents[1] / "nanovllm_upstream/nanovllm/engine/cpu_prefix_cache.py"
    # Only the two quantization functions are needed; skip the engine package.
    source = path.read_text()
    namespace = {"torch": torch}
    start = source.index("# Largest finite float8_e4m3fn magnitude.")
    end = source.index("@dataclass(frozen=True)")
    exec(compile(source[start:end], str(path), "exec"), namespace)
    return namespace["fp8_quantize"], namespace["fp8_dequantize"]


def round_trip(cache, quantize, dequantize):
    """Replace every layer's K/V with its FP8 round trip, in place."""
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):  # [batch, KV head, token, head dim]
            payload = torch.empty_like(tensor, dtype=torch.float8_e4m3fn)
            scale = torch.empty((*tensor.shape[:-1], 1), dtype=tensor.dtype, device=tensor.device)
            quantize(tensor, payload, scale)
            dequantize(payload, scale, tensor)


@torch.inference_mode()
def evaluate(model, ids, prefix_len, score_len, greedy_len, quantize, dequantize):
    from transformers import DynamicCache
    prefix = ids[:, :prefix_len]
    target = ids[:, prefix_len:prefix_len + score_len + 1]
    results = {}
    logits = {}
    for name in ("bf16", "fp8", "chunked"):
        cache = DynamicCache()
        if name == "chunked":
            for start in range(0, prefix_len, 4096):
                model(prefix[:, start:start + 4096], past_key_values=cache, use_cache=True)
        else:
            model(prefix, past_key_values=cache, use_cache=True)
        if name == "fp8":
            round_trip(cache, quantize, dequantize)
        snapshot = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        out = model(target[:, :-1], past_key_values=cache, use_cache=True).logits.float()
        logits[name] = out
        nll = F.cross_entropy(out[0], target[0, 1:])
        results[f"ppl_{name}"] = nll.exp().item()
        # Greedy continuation from the (possibly quantized) prefix.
        for layer, (k, v) in zip(cache.layers, snapshot):
            layer.keys, layer.values = k, v
        cache.crop(prefix_len)
        generated, token = [], ids[:, prefix_len:prefix_len + 1]
        for _ in range(greedy_len):
            step = model(token, past_key_values=cache, use_cache=True).logits[:, -1]
            token = step.argmax(-1, keepdim=True)
            generated.append(token.item())
        results[f"greedy_{name}"] = generated
    p = logits["bf16"].log_softmax(-1)
    a = results.pop("greedy_bf16")
    for name, tag in (("fp8", ""), ("chunked", "_chunked")):
        q = logits[name].log_softmax(-1)
        results["kl_mean" + tag] = F.kl_div(q, p, log_target=True, reduction="none").sum(-1).mean().item()
        results["top1_agreement" + tag] = (p.argmax(-1) == q.argmax(-1)).float().mean().item()
        b = results.pop(f"greedy_{name}")
        results["greedy_match_prefix" + tag] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True, help="wikitext parquet file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix-lens", default="2048,8192,16384,32768")
    parser.add_argument("--score-len", type=int, default=512)
    parser.add_argument("--greedy-len", type=int, default=128)
    parser.add_argument("--windows", type=int, default=4, help="Text windows per prefix length")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer
    quantize, dequantize = load_fp8_functions()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    text = "".join(pq.read_table(args.text)["text"].to_pylist())
    ids = torch.tensor([tokenizer(text)["input_ids"]], device="cuda")
    rows = []
    for prefix_len in map(int, args.prefix_lens.split(",")):
        span = prefix_len + args.score_len + 1
        stride = max(1, (ids.shape[1] - span) // args.windows)
        for w in range(args.windows):
            window = ids[:, w * stride:w * stride + span]
            row = {"prefix_len": prefix_len, "window": w,
                   **evaluate(model, window, prefix_len, args.score_len, args.greedy_len, quantize, dequantize)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                    "text_tokens": ids.shape[1], "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
