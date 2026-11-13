"""Check the engine's YaRN rotary tables against transformers' Qwen3 implementation.

With --text, also score real text after a long prefix (transformers, BF16) with
YaRN versus plain RoPE extrapolation, to show long-context outputs are sane.

Run on a CUDA box: python -m bench.check_yarn --model DIR [--factor 4] [--text PARQUET]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--text", type=Path, help="wikitext parquet for the perplexity check")
    parser.add_argument("--prefix-lens", default="16384,32768,49152,65024")
    parser.add_argument("--score-len", type=int, default=512)
    args = parser.parse_args()
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    path = Path(__file__).resolve().parents[1] / "nanovllm_upstream/nanovllm/layers/rotary_embedding.py"
    spec = importlib.util.spec_from_file_location("rotary_embedding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = AutoConfig.from_pretrained(args.model)
    original = 32768
    config.rope_scaling = {"rope_type": "yarn", "factor": args.factor,
                           "original_max_position_embeddings": original}
    config.max_position_embeddings = int(original * args.factor)
    reference = Qwen3RotaryEmbedding(config)
    positions = torch.tensor([[0, 1, 1000, 32767, 40960, 65535, config.max_position_embeddings - 1]])
    x = torch.zeros(1, positions.shape[1], config.head_dim)
    cos_ref, sin_ref = reference(x, positions)
    ours = module.RotaryEmbedding(config.head_dim, config.head_dim, config.max_position_embeddings,
                                  config.rope_theta, (args.factor, original))
    cos, sin = ours.cos_sin_cache[positions[0], 0].chunk(2, dim=-1)
    half = config.head_dim // 2  # transformers repeats each frequency for both halves.
    error = max((cos - cos_ref[0, :, :half]).abs().max().item(),
                (sin - sin_ref[0, :, :half]).abs().max().item(),
                (cos_ref[0, :, :half] - cos_ref[0, :, half:]).abs().max().item())
    print(f"max |engine - transformers| over cos/sin tables: {error:.2e}")
    if args.text:
        perplexity(args, config)
    return 0 if error < 1e-4 else 1


@torch.inference_mode()
def perplexity(args, yarn_config):
    import pyarrow.parquet as pq
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = torch.tensor([tokenizer("".join(pq.read_table(args.text)["text"].to_pylist()))["input_ids"]],
                       device="cuda")
    plain = AutoConfig.from_pretrained(args.model)
    plain.max_position_embeddings = yarn_config.max_position_embeddings
    for name, config in (("plain RoPE", plain), ("YaRN", yarn_config)):
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").cuda().eval()
        for prefix in map(int, args.prefix_lens.split(",")):
            nll = []
            for start in (0, ids.shape[1] // 2):  # Two text windows.
                window = ids[:, start:start + prefix + args.score_len + 1]
                logits = model(window[:, :-1], logits_to_keep=args.score_len).logits.float()
                nll.append(F.cross_entropy(logits[0], window[0, -args.score_len:]).item())
            print(f"{name:10s} prefix {prefix:6d}: perplexity of next {args.score_len} tokens "
                  f"{torch.tensor(nll).mean().exp().item():8.2f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
