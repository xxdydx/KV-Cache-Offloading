"""The agent-replay workload on vLLM, for comparison with the nano-vllm CPU tier.

Replays the same seeded workload as bench.agent_replay (make_agent_workload)
through vLLM's AsyncLLM: each agent arrives, runs a turn, waits its synthetic
tool time, then resumes with prior context, outputs, and new tool-result tokens.
Policies: vllm_gpu_only (GPU prefix cache only) and vllm_cpu_offload (vLLM's
native OffloadingConnector, which stores evicted KV blocks in CPU memory).
Writes result.json in agent_replay's format so bench.summarize_agent_replay
reads both engines; there is no step trace, so prefill/decode times are 0.

Run in a venv with vLLM: python -m bench.vllm_replay --model DIR --out DIR
     --policy vllm_cpu_offload --num-gpu-blocks N --cpu-cache-gib G [workload flags]
"""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from bench.agent_replay import TurnSpec, distribution_ms, make_agent_workload, nonnegative_float
from bench.baseline import positive_int
from bench.inventory import collect, save_json
from bench.metrics import inter_token_latencies


async def run_agent(engine, name, spec, sampling, start, records, clock=time.perf_counter):
    from vllm.inputs import TokensPrompt
    context = []
    ready = start + spec["arrival_s"]
    for turn in spec["turns"]:
        await asyncio.sleep(max(0.0, ready - clock()))
        prompt = context + turn.tool_result_tokens
        token_times, output_ids, cached = [], [], None
        async for out in engine.generate(TokensPrompt(prompt_token_ids=prompt),
                                         sampling(turn.output_tokens), f"{name}-{turn.turn_index}"):
            now = clock()
            new = out.outputs[0].token_ids
            token_times += [now] * len(new)
            output_ids += new
            cached = out.num_cached_tokens
        after = clock()
        if len(output_ids) != turn.output_tokens:
            raise RuntimeError(f"{name} turn {turn.turn_index}: {len(output_ids)} tokens, "
                               f"expected {turn.output_tokens}")
        records.append({
            "agent": name, "turn": turn.turn_index, "ready_s": ready - start,
            "prompt_len": len(prompt), "num_cached_tokens": cached or 0,
            # vLLM admits requests internally; resume latency is not observable here.
            "resume_latency_s": 0.0,
            "ttft_s": token_times[0] - ready, "turn_latency_s": after - ready,
            "token_times_s": [t - start for t in token_times], "output_ids": output_ids,
        })
        context = prompt + output_ids
        ready = after + turn.tool_wait_s
    return after - start - spec["arrival_s"]


async def replay(engine, agents, sampling):
    start = time.perf_counter()
    records = []
    trajectory_s = await asyncio.gather(*(run_agent(engine, name, spec, sampling, start, records)
                                          for name, spec in agents.items()))
    wall_s = time.perf_counter() - start
    records.sort(key=lambda r: (r["agent"], r["turn"]))
    reused = sum(r["num_cached_tokens"] for r in records)
    total_prompt = sum(r["prompt_len"] for r in records)
    summary = {
        "wall_s": wall_s,
        "completed_trajectories": len(agents),
        "completed_trajectories_per_min": 60 * len(agents) / wall_s,
        "trajectory_latency_ms": distribution_ms(trajectory_s),
        "turn_ttft_ms": distribution_ms([r["ttft_s"] for r in records]),
        "turn_resume_latency_ms": distribution_ms([r["resume_latency_s"] for r in records]),
        "turn_latency_ms": distribution_ms([r["turn_latency_s"] for r in records]),
        "inter_token_ms": distribution_ms([gap for r in records
                                           for gap in inter_token_latencies(r["token_times_s"])]),
        "prompt_tokens_total": total_prompt,
        "prompt_tokens_reused": reused,
        "prompt_tokens_recomputed_or_new": total_prompt - reused,
    }
    return {"summary": summary, "turns": records, "steps": []}


def engine_args(args, token_bytes):
    from vllm import AsyncEngineArgs
    extra = {}
    if args.policy == "vllm_cpu_offload":
        from vllm.config import KVTransferConfig
        num_cpu_blocks = int(args.cpu_cache_gib * 1024**3) // (token_bytes * args.block_size)
        extra["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both",
            kv_connector_extra_config={"num_cpu_blocks": num_cpu_blocks, "block_size": args.block_size})
    return AsyncEngineArgs(
        model=str(args.model), skip_tokenizer_init=True, enable_prefix_caching=True,
        block_size=args.block_size, num_gpu_blocks_override=args.num_gpu_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=not args.cuda_graphs, seed=0, **extra)


async def main_async(args):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    model_config = json.loads((args.model / "config.json").read_text())
    head_dim = model_config.get("head_dim") or model_config["hidden_size"] // model_config["num_attention_heads"]
    # K and V, every layer, BF16: KV bytes per token.
    token_bytes = 2 * model_config["num_hidden_layers"] * model_config["num_key_value_heads"] * head_dim * 2
    agents = make_agent_workload(
        vocab_size=model_config["vocab_size"], seed=args.seed, num_agents=args.num_agents,
        turns_per_agent=args.turns_per_agent, initial_prompt=args.initial_prompt,
        tool_result_tokens=args.tool_result_tokens, output_tokens=args.output_tokens,
        tool_wait_s=args.tool_wait_s, stagger_s=args.stagger_s)
    workload_dump = {name: {"arrival_s": spec["arrival_s"], "turns": [asdict(t) for t in spec["turns"]]}
                     for name, spec in agents.items()}
    args.out.mkdir(parents=True)
    save_json(args.out / "workload.json", workload_dump)
    save_json(args.out / "manifest.json", {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "workload_sha256": hashlib.sha256(json.dumps(workload_dump, sort_keys=True).encode()).hexdigest(),
        "inventory": collect(), "model_config": model_config})

    engine = AsyncLLM.from_engine_args(engine_args(args, token_bytes))
    try:
        def sampling(count):
            return SamplingParams(temperature=0, max_tokens=count, ignore_eos=args.ignore_eos,
                                  detokenize=False, output_kind=RequestOutputKind.DELTA)
        # Warm-up outside the measurement, as agent_replay does.
        await replay(engine, {"warm": {"arrival_s": 0.0, "turns": [TurnSpec("warm", 0, [1, 2, 3], 2, 0.0)]}},
                     sampling)
        result = await replay(engine, agents, sampling)
        result["summary"]["policy"] = args.policy
        save_json(args.out / "result.json", result)
        print(json.dumps(result["summary"], indent=2))
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New run directory")
    parser.add_argument("--policy", choices=("vllm_gpu_only", "vllm_cpu_offload"), required=True)
    parser.add_argument("--num-gpu-blocks", type=positive_int, required=True,
                        help="GPU KV pool in --block-size blocks (match the nano-vllm pool)")
    parser.add_argument("--block-size", type=positive_int, default=16)
    parser.add_argument("--cpu-cache-gib", type=nonnegative_float, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
