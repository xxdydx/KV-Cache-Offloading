"""Online mixed-length workload for comparing prefill policies on one GPU.

Run each policy in a fresh process with identical workload/resource arguments.
Synchronized host timestamps include scheduling, KV copies and model execution.
"""

import argparse
import atexit
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

from bench.baseline import positive_int
from bench.inventory import collect, save_json
from bench.metrics import inter_token_latencies, percentile


@dataclass(frozen=True)
class Request:
    name: str
    group: str
    arrival_s: float
    prompt: list[int]
    output_tokens: int


def make_workload(*, vocab_size, seed=42, initial_requests=4, initial_prompt=128,
                  initial_output=128, pairs=4, short_prompt=64, long_prompt=4096,
                  output_tokens=32, start_delay_s=0.02, arrival_gap_s=0.005):
    """Fixed wall-clock arrivals, independent of scheduler decisions or outcomes."""
    rng = random.Random(seed)
    requests = []

    def add(name, group, arrival, length, output):
        requests.append(Request(name, group, arrival,
                                [rng.randrange(vocab_size) for _ in range(length)], output))

    for i in range(initial_requests):
        add(f"initial-{i}", "initial", 0.0, initial_prompt, initial_output)
    for i in range(pairs):
        add(f"long-{i}", "long", start_delay_s + 2 * i * arrival_gap_s, long_prompt, output_tokens)
        add(f"short-{i}", "short", start_delay_s + (2 * i + 1) * arrival_gap_s,
            short_prompt, output_tokens)
    return requests


def distribution_ms(values):
    return {f"p{q}": 1000 * percentile(values, q) if values else None for q in (50, 95, 99)}


def run_workload(llm, requests, sampling_params, synchronize, *, clock=time.perf_counter,
                 sleep=time.sleep, timeout_s=120, trace=None):
    """Measure real event APIs; injectable clock/runner permit deterministic tests.

    Arrivals crossed during a model step are submitted at the next boundary.
    Their latency still starts at their ORIGINAL arrival, counting that delay.
    trace, when supplied, retains completed records even if the run fails.
    """
    if not requests or len({item.name for item in requests}) != len(requests):
        raise ValueError("workload must contain requests with unique names")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    for item in requests:
        if not math.isfinite(item.arrival_s) or item.arrival_s < 0 or not item.prompt or item.output_tokens < 1:
            raise ValueError("invalid arrival, prompt, or output budget")
    if not llm.is_finished():
        raise ValueError("engine must be idle before a measured workload")
    pending = sorted(requests, key=lambda item: item.arrival_s)
    by_id, records, steps = {}, {}, []
    trace = trace if trace is not None else []
    synchronize()
    start = clock()
    next_request = 0
    while next_request < len(pending) or not llm.is_finished():
        now = clock() - start
        if now > timeout_s:
            raise TimeoutError("workload exceeded timeout; unfinished work is a failed run")
        while next_request < len(pending) and pending[next_request].arrival_s <= now:
            item = pending[next_request]
            seq_id = llm.add_request(item.prompt, sampling_params(item.output_tokens))
            if seq_id in by_id:
                raise RuntimeError("engine reused a request ID")
            by_id[seq_id] = item.name
            records[item.name] = {
                "group": item.group, "arrival_s": item.arrival_s,
                "submitted_s": clock() - start, "first_scheduled_s": None,
                "token_times_s": [], "token_ids": [], "finished": False,
            }
            trace.append({"kind": "arrival", "name": item.name, "seq_id": seq_id,
                          "arrival_s": item.arrival_s, "submitted_s": records[item.name]["submitted_s"]})
            next_request += 1
        if llm.is_finished():
            sleep(min(0.005, max(0, pending[next_request].arrival_s - (clock() - start))))
            continue
        before = clock() - start
        result = llm.step_with_events()
        synchronize()
        after = clock() - start
        if after > timeout_s:
            raise TimeoutError("workload exceeded timeout; timed-out runs are not performance results")
        if not result.num_tokens:
            raise RuntimeError("engine returned an empty step")
        decision = llm.scheduler.last_schedule
        step = {"kind": "step", **asdict(decision), "start_s": before, "end_s": after,
                "requests": [by_id[seq_id] for seq_id in decision.seq_ids]}
        steps.append(step)
        trace.append(step)
        for name in step["requests"]:
            if records[name]["first_scheduled_s"] is None:
                records[name]["first_scheduled_s"] = before
        for event in result.token_events:
            record = records[by_id[event.seq_id]]
            if record["finished"] or event.output_index != len(record["token_ids"]):
                raise RuntimeError("missing, duplicate, or reordered output token")
            record["token_ids"].append(event.token_id)
            record["token_times_s"].append(after)
            record["finished"] = event.finished
            trace.append({"kind": "token", "name": by_id[event.seq_id],
                          "output_index": event.output_index, "token_id": event.token_id,
                          "finished": event.finished, "time_s": after})
        for seq_id, token_ids in result.outputs:
            record = records[by_id[seq_id]]
            if not record["finished"] or token_ids != record["token_ids"]:
                raise RuntimeError("completed output disagrees with token events")

    wall_s = clock() - start
    for item in requests:
        record = records[item.name]
        if not record["finished"] or len(record["token_ids"]) != item.output_tokens:
            raise RuntimeError("engine dropped work or did not honor the fixed output budget")
        record["ttft_s"] = record["token_times_s"][0] - item.arrival_s
        record["completion_s"] = record["token_times_s"][-1] - item.arrival_s
        record["prefill_wait_s"] = record["first_scheduled_s"] - item.arrival_s

    def summarize(rows):
        gaps = [gap for row in rows for gap in inter_token_latencies(row["token_times_s"])]
        return {"requests": len(rows), "ttft_ms": distribution_ms([row["ttft_s"] for row in rows]),
                "inter_token_ms": distribution_ms(gaps),
                "completion_ms": distribution_ms([row["completion_s"] for row in rows]),
                "prefill_wait_ms": distribution_ms([row["prefill_wait_s"] for row in rows])}

    contended = [step for step in steps if step["phase"] == "prefill" and step["decode_requests"]]
    output_tokens = sum(item.output_tokens for item in requests)
    summary = {
        **summarize(list(records.values())), "wall_s": wall_s,
        "completed_requests_per_s": len(requests) / wall_s,
        "output_tokens_per_s": output_tokens / wall_s,
        "processed_prefill_tokens": sum(step["scheduled_tokens"] for step in steps if step["phase"] == "prefill"),
        "contended_prefill_steps": len(contended),
        "max_contended_prefill_step_ms": max((1000 * (step["end_s"] - step["start_s"])
                                             for step in contended), default=None),
        "groups": {group: summarize([row for row in records.values() if row["group"] == group])
                   for group in sorted({item.group for item in requests})},
    }
    return {"summary": summary, "requests": records, "steps": steps}


def nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New run directory")
    parser.add_argument("--policy", choices=("legacy", "fixed", "decode_aware"), default="legacy")
    parser.add_argument("--chunk-size", type=positive_int, default=512)
    parser.add_argument("--min-chunk-size", type=positive_int, default=32)
    parser.add_argument("--prefill-budget", type=positive_int, default=4096)
    parser.add_argument("--max-num-seqs", type=positive_int, default=32)
    parser.add_argument("--initial-requests", type=positive_int, default=4)
    parser.add_argument("--initial-prompt", type=positive_int, default=128)
    parser.add_argument("--initial-output", type=positive_int, default=128)
    parser.add_argument("--pairs", type=positive_int, default=4)
    parser.add_argument("--short-prompt", type=positive_int, default=64)
    parser.add_argument("--long-prompt", type=positive_int, default=4096)
    parser.add_argument("--output-tokens", type=positive_int, default=32)
    parser.add_argument("--start-delay-ms", type=nonnegative_float, default=20)
    parser.add_argument("--arrival-gap-ms", type=nonnegative_float, default=5)
    parser.add_argument("--timeout-s", type=positive_int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--cpu-cache-gib", type=nonnegative_float, default=0,
                        help="Optional automatic CPU prefix cache, same budget for policy comparisons")
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    if args.min_chunk_size > args.chunk_size:
        parser.error("--min-chunk-size must not exceed --chunk-size")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.out.exists():
        parser.error("--out exists; choose a new directory")
    if not (args.model / "config.json").is_file():
        parser.error("--model must be a local pinned Qwen3 checkpoint")
    model_config = json.loads((args.model / "config.json").read_text())
    if model_config.get("model_type") != "qwen3":
        parser.error("the vendored engine supports Qwen3 for this experiment")
    max_len = max(args.initial_prompt + args.initial_output,
                  args.long_prompt + args.output_tokens, args.short_prompt + args.output_tokens)
    if max_len > model_config["max_position_embeddings"]:
        parser.error("prompt plus output exceeds checkpoint context limit")
    requests = make_workload(
        vocab_size=model_config["vocab_size"], seed=args.seed,
        initial_requests=args.initial_requests, initial_prompt=args.initial_prompt,
        initial_output=args.initial_output, pairs=args.pairs, short_prompt=args.short_prompt,
        long_prompt=args.long_prompt, output_tokens=args.output_tokens,
        start_delay_s=args.start_delay_ms / 1000, arrival_gap_s=args.arrival_gap_ms / 1000)

    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required; the Mac can run the scheduler regression tests")
    root = Path(__file__).resolve().parents[1] / "nanovllm_upstream" / "nanovllm"
    spec = importlib.util.spec_from_file_location("nanovllm", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["nanovllm"] = package
    spec.loader.exec_module(package)
    from nanovllm import LLM, SamplingParams

    args.out.mkdir(parents=True)
    workload = [asdict(item) for item in requests]
    save_json(args.out / "workload.json", workload)
    save_json(args.out / "manifest.json", {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "workload_sha256": hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest(),
        "inventory": collect(), "model_config": model_config,
        "engine_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(root.rglob("*.py"))},
        "limitations": ["Synthetic mixed-length requests, not multi-turn agent replay or a cache-benefit benchmark.",
                         "Synchronized host step times include scheduler, copy, sampling and execution overhead.",
                         "Arrival timestamps count delay until the next engine boundary; no arrival is reset on submission.",
                         "Chunk token budgets are heuristics, not wall-clock latency guarantees.",
                         "Use multiple fresh-process repetitions; verify contended_prefill_steps is nonzero."]})
    llm = LLM(str(args.model.resolve()), tensor_parallel_size=1,
              enforce_eager=not args.cuda_graphs, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.prefill_budget, max_model_len=max_len,
              gpu_memory_utilization=args.gpu_memory_utilization, prefill_policy=args.policy,
              prefill_chunk_size=args.chunk_size, prefill_min_chunk_size=args.min_chunk_size)
    trace = []
    try:
        manager = llm.scheduler.block_manager
        # Keep this first scheduling experiment below worst-case KV pressure.
        # Cache/memory-pressure experiments are separate ablations.
        required = sum((len(item.prompt) + item.output_tokens - 2 + manager.block_size)
                       // manager.block_size for item in requests)
        save_json(args.out / "capacity.json", {"worst_case_required_blocks": required,
                                              "available_blocks": len(manager.blocks)})
        if required > len(manager.blocks):
            raise RuntimeError("workload exceeds worst-case KV capacity; reduce requests/context or raise GPU budget")
        # Warm both execution paths before timing; use a separate deterministic prefix.
        llm.generate([[1, 2, 3]], SamplingParams(temperature=0, ignore_eos=True, max_tokens=2), use_tqdm=False)
        if args.cpu_cache_gib:
            llm.create_cpu_prefix_cache(int(args.cpu_cache_gib * 1024**3), automatic=True)
        torch.cuda.reset_peak_memory_stats()
        result = run_workload(llm, requests,
                              lambda count: SamplingParams(temperature=0, ignore_eos=True, max_tokens=count),
                              torch.cuda.synchronize, timeout_s=args.timeout_s, trace=trace)
        result["summary"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save_json(args.out / "result.json", result)
        print(json.dumps(result["summary"], indent=2))
    except Exception as error:
        save_json(args.out / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        save_json(args.out / "trace.json", trace)
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
