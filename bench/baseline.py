"""One-GPU fixed-batch diagnostic with synchronized prefill/decode wall timings.

Synthetic prompts have fresh prefixes across trials. Per-step synchronization
is intentional; this is not a production-serving or uninstrumented throughput run.
"""

import argparse
import atexit
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

from bench.inventory import collect, save_json


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def measure_batch(llm, prompts, sampling_params, synchronize):
    """Measure the real engine's step API; also usable with a scripted CPU runner."""
    synchronize()
    start = time.perf_counter()
    request_ids = [llm.add_request(prompt, sampling_params) for prompt in prompts]
    outputs = {request_id: [] for request_id in request_ids}
    steps = []
    while not llm.is_finished():
        synchronize()
        before = time.perf_counter()
        result = llm.step_with_events()
        synchronize()
        duration = time.perf_counter() - before
        if not result.num_tokens:
            raise RuntimeError("Engine returned an empty step")
        steps.append({"phase": "prefill" if result.num_tokens > 0 else "decode",
                      "tokens": abs(result.num_tokens), "wall_s": duration})
        for event in result.token_events:
            tokens = outputs[event.seq_id]
            if event.output_index != len(tokens):
                raise RuntimeError("Missing, duplicate, or reordered output token")
            tokens.append(event.token_id)
    total_s = time.perf_counter() - start
    if any(len(tokens) != sampling_params.max_tokens for tokens in outputs.values()):
        raise RuntimeError("Output token count did not match the fixed budget")
    prefill = [step for step in steps if step["phase"] == "prefill"]
    decode = [step for step in steps if step["phase"] == "decode"]
    prompt_tokens = sum(map(len, prompts))
    if sum(step["tokens"] for step in prefill) != prompt_tokens:
        raise RuntimeError("Unexpected prefix-cache hit or prefill recomputation; trial is not comparable")
    expected_decode_tokens = len(prompts) * (sampling_params.max_tokens - 1)
    if sum(step["tokens"] for step in decode) != expected_decode_tokens:
        raise RuntimeError("Unexpected decode work or preemption")
    if any(step["tokens"] != len(prompts) for step in decode):
        raise RuntimeError("Decode did not preserve the requested fixed batch")
    prefill_s = sum(step["wall_s"] for step in prefill)
    decode_s = sum(step["wall_s"] for step in decode)
    return {
        "prefill_wall_s": prefill_s, "prefill_tokens_per_s": prompt_tokens / prefill_s,
        "decode_wall_s": decode_s,
        "decode_tokens_per_s": expected_decode_tokens / decode_s if decode else None,
        "mean_decode_step_ms": 1000 * decode_s / len(decode) if decode else None,
        "total_wall_s": total_s, "steps": steps,
        "prompt_token_ids": prompts, "output_token_ids": list(outputs.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local pinned Qwen3-0.6B checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--prompt-tokens", type=positive_int, default=128)
    parser.add_argument("--prefill-budget", type=positive_int, default=4096,
                        help="Maximum prompt tokens per engine step; bounds large-batch prefill workspaces")
    parser.add_argument("--output-tokens", type=positive_int, default=32)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--cuda-graphs", action="store_true", help="Enable the engine's GPU graph path; eager is the default")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if not (args.model / "config.json").is_file():
        parser.error("--model must be a local checkpoint directory containing config.json")
    model_config = json.loads((args.model / "config.json").read_text())
    if model_config.get("model_type") != "qwen3":
        parser.error("This baseline supports the engine's Qwen3 architecture only")
    max_len = args.prompt_tokens + args.output_tokens
    if max_len > model_config["max_position_embeddings"]:
        parser.error("Prompt plus output exceeds the checkpoint context limit")
    if args.out.exists():
        parser.error("--out already exists; choose a new run directory")

    # Always use this repository's engine, even if a stock editable install exists.
    import importlib.util
    engine_root = Path(__file__).resolve().parents[1] / "nanovllm_upstream" / "nanovllm"
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required. Run this command in the RTX sandbox; use bench.inventory on the Mac.")
    package_spec = importlib.util.spec_from_file_location(
        "nanovllm", engine_root / "__init__.py", submodule_search_locations=[str(engine_root)])
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["nanovllm"] = package
    package_spec.loader.exec_module(package)
    from nanovllm import LLM, SamplingParams

    args.out.mkdir(parents=True)
    manifest = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "inventory": collect(), "model_config": model_config,
                "engine_source": str(engine_root),
                "engine_sha256": {str(p.relative_to(engine_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(engine_root.rglob("*.py"))},
                "limitations": ["Synthetic fixed-batch diagnostic; no natural-language quality claim.",
                                 "Synchronized host step times include scheduling, sampling, and transfer overhead.",
                                 "Prefill includes first output token; decode counts the remaining N-1 tokens.",
                                 "Not a sustained online service or a GPU-kernel-only throughput measurement."]}
    save_json(args.out / "manifest.json", manifest)
    llm = LLM(str(args.model.resolve()), tensor_parallel_size=1,
              enforce_eager=not args.cuda_graphs, max_num_seqs=args.batch_size,
              max_num_batched_tokens=min(args.prefill_budget, args.batch_size * args.prompt_tokens),
              max_model_len=max_len, gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        block_size = llm.scheduler.block_manager.block_size
        required_blocks = args.batch_size * ((max_len - 1 + block_size - 1) // block_size)
        available_blocks = len(llm.scheduler.block_manager.blocks)
        save_json(args.out / "capacity.json", {"required_kv_blocks": required_blocks,
                  "available_kv_blocks": available_blocks, "block_size": block_size})
        if required_blocks > available_blocks:
            raise RuntimeError("Insufficient KV blocks for a fixed-batch trial; reduce batch/context or increase the declared GPU budget")
        params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=args.output_tokens)
        rng = random.Random(args.seed)
        measured = []
        for trial in range(args.warmup + args.repeats):
            prompts = [[rng.randrange(model_config["vocab_size"]) for _ in range(args.prompt_tokens)]
                       for _ in range(args.batch_size)]
            torch.cuda.reset_peak_memory_stats()
            result = measure_batch(llm, prompts, params, torch.cuda.synchronize)
            result.update(trial=trial, warmup=trial < args.warmup,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save_json(args.out / f"trial-{trial:03d}.json", result)
            if not result["warmup"]:
                measured.append(result)
        keys = ("prefill_tokens_per_s", "decode_tokens_per_s", "mean_decode_step_ms", "total_wall_s")
        summary = {"measured_repeats": len(measured), "median": {
            key: statistics.median(row[key] for row in measured) if measured[0][key] is not None else None
            for key in keys}}
        save_json(args.out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        # LLMEngine also registers exit at construction; avoid calling it twice.
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
