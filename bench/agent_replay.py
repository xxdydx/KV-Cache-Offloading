"""Deterministic multi-turn agent replay on one GPU: real inference, synthetic tool waits.

Each agent repeatedly: arrives/resumes, runs a turn to completion, waits a fixed
synthetic duration (standing in for a tool call), then resumes with its prior
context plus new tool-result tokens appended. This is the A2 replay harness
from spec.md: it measures whether CPU-preserved KV helps completed-trajectory
throughput and latency relative to the existing GPU-only prefix cache and a
forced cold-recompute control. It is not the prefill-scheduling benchmark.
"""

import argparse
import atexit
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from bench.baseline import positive_int
from bench.inventory import collect, save_json
from bench.metrics import inter_token_latencies, percentile


@dataclass(frozen=True)
class TurnSpec:
    agent: str
    turn_index: int
    tool_result_tokens: list[int]  # Appended before this turn; empty for turn 0.
    output_tokens: int
    tool_wait_s: float  # Synthetic wait after this turn finishes, before the next arrives.


def make_agent_workload(*, vocab_size, seed=42, num_agents=2, turns_per_agent=2,
                        initial_prompt=256, tool_result_tokens=32, output_tokens=64,
                        tool_wait_s=1.0, stagger_s=0.0):
    """Independent per-agent turn sequences; arrival/resume timing is computed by the runner."""
    rng = random.Random(seed)
    agents = {}
    for a in range(num_agents):
        name = f"agent-{a}"
        turns = []
        for t in range(turns_per_agent):
            length = initial_prompt if t == 0 else tool_result_tokens
            tokens = [rng.randrange(vocab_size) for _ in range(length)]
            turns.append(TurnSpec(name, t, tokens, output_tokens, tool_wait_s))
        agents[name] = {"turns": turns, "arrival_s": a * stagger_s}
    return agents


def distribution_ms(values):
    return {f"p{q}": 1000 * percentile(values, q) if values else None for q in (50, 95, 99)}


def install_cache_hit_tracking(manager) -> dict:
    """Record each request's genuine GPU/CPU prefix-cache hit, in tokens, by seq_id.

    seq.num_cached_tokens is not that signal on its own: scheduler.postprocess
    does `seq.num_cached_tokens += seq.num_scheduled_tokens` every prefill step,
    so by the time a turn finishes it always equals the full prompt length,
    hit or not. BlockManager.allocate's num_cached_blocks argument is the real
    hit count, so wrap it (instance-level; vendored engine file stays untouched).
    """
    hits: dict[int, int] = {}
    original_allocate = manager.allocate

    def wrapped(seq, num_cached_blocks):
        hits[seq.seq_id] = num_cached_blocks * manager.block_size
        return original_allocate(seq, num_cached_blocks)

    manager.allocate = wrapped
    return hits


def _force_recompute(llm, manager, vocab_size, max_request_tokens, rng):
    """Flood every currently free GPU block with filler so finished agents' prefixes recompute.

    Diagnostic only: this deliberately removes idle prefix reuse between turns.
    It is not the headline GPU-only baseline. Best-effort: concurrent admissions
    by other agents between this call and the next turn's arrival can still
    leave a few blocks un-recycled; it is not a guaranteed-cold control. Each
    filler request is capped below max_request_tokens (the model's context
    limit); a free pool larger than that is flooded with several requests.
    """
    remaining = (len(manager.blocks) - len(manager.used_block_ids)) * manager.block_size - 1
    while remaining > 0:
        size = min(remaining, max_request_tokens)
        llm.add_request([rng.randrange(vocab_size) for _ in range(size)],
                        _sampling_params_cls(temperature=0, ignore_eos=True, max_tokens=1))
        remaining -= size


# Set once in main() after nanovllm is dynamically loaded; evict_recompute needs
# its own SamplingParams instance independent of the caller's sampling_params factory.
_sampling_params_cls = None


def _bind_sampling_params(cls):
    global _sampling_params_cls
    _sampling_params_cls = cls


def run_replay(llm, agents, sampling_params, synchronize, cache_hits, *, clock=time.perf_counter,
               sleep=time.sleep, timeout_s=120, policy="gpu_only", trace=None,
               vocab_size=None, max_model_len=None, seed=0):
    """Drive every agent's turns to completion with real engine steps.

    policy in {"gpu_only", "cpu_backed", "evict_recompute"}; cpu_backed assumes
    the caller already attached an automatic CPU prefix cache. evict_recompute
    injects filler requests between turns (see _force_recompute). cache_hits
    is the dict from install_cache_hit_tracking, keyed by seq_id.
    """
    if policy not in ("gpu_only", "cpu_backed", "evict_recompute"):
        raise ValueError("unknown policy")
    if policy == "evict_recompute" and not (vocab_size and max_model_len):
        raise ValueError("evict_recompute requires vocab_size and max_model_len for filler requests")
    if not llm.is_finished():
        raise ValueError("engine must be idle before a measured replay")
    filler_rng = random.Random(seed)

    state = {}
    for name, spec in agents.items():
        state[name] = {
            "turns": spec["turns"], "next_turn": 0, "context": [],
            "ready_s": spec["arrival_s"], "admitted_seq_id": None,
            "admit_prompt_len": None, "admit_s": None, "token_times": [],
            "records": [], "done": False, "completed_s": None,
        }
    by_id = {}
    steps = []
    trace = trace if trace is not None else []
    manager = llm.scheduler.block_manager
    synchronize()
    start = clock()

    while not all(st["done"] for st in state.values()):
        now = clock() - start
        if now > timeout_s:
            raise TimeoutError("replay exceeded timeout; unfinished work is a failed run")
        admitted_any = False
        for name, st in state.items():
            if st["done"] or st["admitted_seq_id"] is not None or st["ready_s"] > now:
                continue
            turn = st["turns"][st["next_turn"]]
            prompt = st["context"] + turn.tool_result_tokens
            if not prompt:
                raise ValueError("turn produced an empty prompt")
            seq_id = llm.add_request(prompt, sampling_params(turn.output_tokens))
            if seq_id in by_id:
                raise RuntimeError("engine reused a request ID")
            by_id[seq_id] = name
            st.update(admitted_seq_id=seq_id, admit_prompt_len=len(prompt), admit_s=now,
                      token_times=[])
            trace.append({"kind": "admit", "agent": name, "turn": st["next_turn"],
                         "seq_id": seq_id, "ready_s": st["ready_s"], "admit_s": now,
                         "prompt_len": len(prompt)})
            admitted_any = True
        if llm.is_finished() and not admitted_any:
            pending = [st["ready_s"] for st in state.values()
                      if not st["done"] and st["admitted_seq_id"] is None]
            if not pending:
                break
            sleep(min(0.005, max(0, min(pending) - (clock() - start))))
            continue
        before = clock() - start
        result = llm.step_with_events()
        synchronize()
        after = clock() - start
        if after > timeout_s:
            raise TimeoutError("replay exceeded timeout; timed-out runs are not performance results")
        decision = llm.scheduler.last_schedule
        step = {"kind": "step", **asdict(decision), "start_s": before, "end_s": after}
        steps.append(step)
        trace.append(step)
        completed_in_outputs = {seq_id for seq_id, _ in result.outputs}
        for event in result.token_events:
            name = by_id.get(event.seq_id)
            if name is None:
                continue  # Untracked background request (e.g. evict_recompute filler).
            st = state[name]
            if event.output_index != len(st["token_times"]):
                raise RuntimeError("missing, duplicate, or reordered output token")
            st["token_times"].append(after)
            if event.finished:
                if event.seq_id not in completed_in_outputs:
                    raise RuntimeError("finished token without a matching completed output")
                output_ids = next(ids for sid, ids in result.outputs if sid == event.seq_id)
                turn = st["turns"][st["next_turn"]]
                record = {
                    "agent": name, "turn": st["next_turn"], "ready_s": st["ready_s"],
                    "admit_s": st["admit_s"], "resume_latency_s": st["admit_s"] - st["ready_s"],
                    "ttft_s": st["token_times"][0] - st["ready_s"],
                    "turn_latency_s": after - st["ready_s"],
                    "prompt_len": st["admit_prompt_len"],
                    "num_cached_tokens": cache_hits.get(st["admitted_seq_id"], 0),
                    "token_times_s": list(st["token_times"]),
                }
                st["records"].append(record)
                trace.append({"kind": "turn_complete", **{k: v for k, v in record.items()
                                                          if k != "token_times_s"}})
                st["context"] = st["context"] + turn.tool_result_tokens + output_ids
                st["next_turn"] += 1
                st["admitted_seq_id"] = None
                if st["next_turn"] >= len(st["turns"]):
                    st["done"] = True
                    st["completed_s"] = after
                else:
                    st["ready_s"] = after + turn.tool_wait_s
                    if policy == "evict_recompute":
                        _force_recompute(llm, manager, vocab_size, max_model_len - 1, filler_rng)

    wall_s = clock() - start
    all_turns = [record for st in state.values() for record in st["records"]]
    trajectory_s = [st["completed_s"] - agents[name]["arrival_s"] for name, st in state.items()]

    reused = sum(record["num_cached_tokens"] for record in all_turns)
    total_prompt = sum(record["prompt_len"] for record in all_turns)
    summary = {
        "wall_s": wall_s,
        "completed_trajectories": sum(1 for st in state.values() if st["done"]),
        "completed_trajectories_per_min": 60 * sum(1 for st in state.values() if st["done"]) / wall_s,
        "trajectory_latency_ms": distribution_ms(trajectory_s),
        "turn_ttft_ms": distribution_ms([r["ttft_s"] for r in all_turns]),
        "turn_resume_latency_ms": distribution_ms([r["resume_latency_s"] for r in all_turns]),
        "turn_latency_ms": distribution_ms([r["turn_latency_s"] for r in all_turns]),
        "inter_token_ms": distribution_ms([gap for r in all_turns
                                           for gap in inter_token_latencies(r["token_times_s"])]),
        "prompt_tokens_total": total_prompt,
        "prompt_tokens_reused": reused,
        "prompt_tokens_recomputed_or_new": total_prompt - reused,
        "policy": policy,
    }
    return {"summary": summary, "turns": all_turns, "steps": steps}


def nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New run directory")
    parser.add_argument("--policy", choices=("gpu_only", "cpu_backed", "evict_recompute"),
                        default="gpu_only")
    parser.add_argument("--cpu-cache-gib", type=nonnegative_float, default=8,
                        help="Automatic CPU prefix cache budget when --policy=cpu_backed")
    parser.add_argument("--cpu-cache-save-on-release", action="store_true",
                        help="Save idle prefix blocks when released instead of when recycled")
    parser.add_argument("--cpu-cache-prefetch", action="store_true",
                        help="Restore a blocked waiting turn's CPU hits into spare GPU blocks in the background")
    parser.add_argument("--no-cpu-cache-preallocate", dest="cpu_cache_preallocate",
                        action="store_false",
                        help="Pin CPU cache memory lazily during the run instead of before it")
    parser.add_argument("--num-agents", type=positive_int, default=2)
    parser.add_argument("--turns-per-agent", type=positive_int, default=2)
    parser.add_argument("--initial-prompt", type=positive_int, default=256)
    parser.add_argument("--tool-result-tokens", type=positive_int, default=32)
    parser.add_argument("--output-tokens", type=positive_int, default=64)
    parser.add_argument("--tool-wait-s", type=nonnegative_float, default=1.0)
    parser.add_argument("--stagger-s", type=nonnegative_float, default=0.05)
    parser.add_argument("--max-num-seqs", type=positive_int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=4096)
    parser.add_argument("--max-model-len", type=positive_int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--timeout-s", type=positive_int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Generate exactly --output-tokens per turn so every policy does identical work")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.out.exists():
        parser.error("--out exists; choose a new directory")
    if not (args.model / "config.json").is_file():
        parser.error("--model must be a local pinned Qwen3 checkpoint")
    model_config = json.loads((args.model / "config.json").read_text())
    if model_config.get("model_type") != "qwen3":
        parser.error("the vendored engine supports Qwen3 for this experiment")

    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required; the Mac can run unit tests only")
    root = Path(__file__).resolve().parents[1] / "nanovllm_upstream" / "nanovllm"
    spec = importlib.util.spec_from_file_location("nanovllm", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["nanovllm"] = package
    spec.loader.exec_module(package)
    from nanovllm import LLM, SamplingParams
    _bind_sampling_params(SamplingParams)

    agents = make_agent_workload(
        vocab_size=model_config["vocab_size"], seed=args.seed, num_agents=args.num_agents,
        turns_per_agent=args.turns_per_agent, initial_prompt=args.initial_prompt,
        tool_result_tokens=args.tool_result_tokens, output_tokens=args.output_tokens,
        tool_wait_s=args.tool_wait_s, stagger_s=args.stagger_s)
    workload_dump = {name: {"arrival_s": spec_["arrival_s"],
                            "turns": [asdict(turn) for turn in spec_["turns"]]}
                     for name, spec_ in agents.items()}
    workload_hash = hashlib.sha256(json.dumps(workload_dump, sort_keys=True).encode()).hexdigest()

    args.out.mkdir(parents=True)
    save_json(args.out / "workload.json", workload_dump)
    save_json(args.out / "manifest.json", {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "workload_sha256": workload_hash, "inventory": collect(), "model_config": model_config,
        "limitations": [
            "Synthetic tool waits; no external tool or agent framework executes.",
            "evict_recompute uses best-effort filler flooding between turns, not a guaranteed-cold control.",
            "Greedy sampling with real KV computation; no fixed-output replay here, unlike prefill_scheduling.py.",
            "num_cached_tokens is the block_manager.allocate num_cached_blocks argument at admission,"
            " not seq.num_cached_tokens (which accumulates to the full prompt length regardless of hits).",
            "Single host/GPU, TP=1; not a steady-state capacity or multi-GPU result.",
            "Per-step timing synchronizes the compute stream only; CPU-cache copies on their own"
            " stream may still be in flight when a step's tokens are recorded.",
        ]})
    llm = LLM(str(args.model.resolve()), tensor_parallel_size=1,
              enforce_eager=not args.cuda_graphs, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        llm.generate([[1, 2, 3]], SamplingParams(temperature=0, ignore_eos=True, max_tokens=2), use_tqdm=False)
        cpu_cache = None
        if args.policy == "cpu_backed":
            cpu_cache = llm.create_cpu_prefix_cache(
                int(args.cpu_cache_gib * 1024**3), automatic=True,
                save_on_release=args.cpu_cache_save_on_release,
                preallocate=args.cpu_cache_preallocate, prefetch=args.cpu_cache_prefetch)
        cache_hits = install_cache_hit_tracking(llm.scheduler.block_manager)
        torch.cuda.reset_peak_memory_stats()
        trace = []
        result = run_replay(llm, agents,
                            lambda count: SamplingParams(temperature=0, max_tokens=count,
                                                         ignore_eos=args.ignore_eos),
                            # Token timing waits for compute only; a device-wide sync
                            # would also wait for background CPU-cache copies.
                            torch.cuda.current_stream().synchronize, cache_hits,
                            timeout_s=args.timeout_s,
                            policy=args.policy, trace=trace,
                            vocab_size=model_config["vocab_size"],
                            max_model_len=llm.model_runner.config.max_model_len, seed=args.seed + 1)
        result["summary"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if cpu_cache is not None:
            result["summary"]["cpu_cache"] = {**cpu_cache.stats, "bytes_used": cpu_cache.bytes_used,
                                              "reserved_bytes": cpu_cache.reserved_bytes,
                                              "prefetched_blocks": cpu_cache.prefetched_blocks}
        save_json(args.out / "result.json", result)
        save_json(args.out / "trace.json", trace)
        print(json.dumps(result["summary"], indent=2))
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
