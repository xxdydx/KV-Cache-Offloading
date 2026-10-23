"""Greedy continuation correctness and naive KV export/import measurements.

python -m bench.kv_handoff --model /path/to/Qwen3-8B --out results/new-run
Run on a CUDA host with nano-vllm dependencies installed. Select a physical GPU
using CUDA_VISIBLE_DEVICES. Timings include synchronous host staging, without TCP.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nanovllm_upstream"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.kv_handoff import export_kv, import_kv, verify_imported_kv
from nanovllm.sampling_params import SamplingParams
from bench.runtime_info import memory_info


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


def run_request(engine, prompt, output_tokens, after=None):
    manager = engine.scheduler.block_manager
    assert engine.is_finished() and not manager.used_block_ids
    manager.hash_to_block_id.clear()  # Force fresh prefill; do not reuse prior runs.
    engine.add_request(prompt, SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True))
    seq = engine.scheduler.waiting[-1]
    origin = time.perf_counter()
    token_times = []
    transfer = None
    while not engine.is_finished():
        count = seq.num_completion_tokens
        _, processed_tokens = engine.step()
        if transfer is not None:
            assert processed_tokens < 0, "Resume unexpectedly recomputed prefill"
        if seq.num_completion_tokens > count:
            token_times.append(time.perf_counter() - origin)
        if after is not None and seq.num_completion_tokens == after and transfer is None:
            source = seq
            source_blocks = source.block_table.copy()
            engine.scheduler.running.remove(source)
            snapshot, export_ms = timed(lambda: export_kv(engine.model_runner, source))
            seq, import_ms = timed(lambda: import_kv(engine.model_runner, manager, snapshot))
            destination_blocks = seq.block_table.copy()
            assert set(source_blocks).isdisjoint(destination_blocks), "Source KV was reused"
            verified, verify_ms = timed(lambda: verify_imported_kv(engine.model_runner, seq, snapshot))
            assert verified, "Imported KV differs"
            # Make accidental reads of source storage fail visibly, then release it.
            indices = torch.tensor(source_blocks, dtype=torch.long, device=engine.model_runner.kv_cache.device)
            engine.model_runner.kv_cache.index_fill_(2, indices, float("nan"))
            manager.deallocate(source)
            engine.scheduler.running.append(seq)
            transfer = dict(cached_tokens=snapshot.num_cached_tokens,
                            payload_bytes=snapshot.payload_bytes, export_ms=export_ms,
                            import_ms=import_ms, copy_total_ms=export_ms + import_ms,
                            validation_readback_ms=verify_ms,
                            source_blocks=source_blocks, destination_blocks=destination_blocks,
                            disjoint_blocks=True, kv_exact_match=True, source_poisoned=True,
                            resume_decode_only=True, memory=memory_info())
            del snapshot
    assert len(seq.completion_token_ids) == output_tokens
    assert len(token_times) == output_tokens and not manager.used_block_ids
    return dict(token_ids=seq.completion_token_ids, token_times_s=token_times,
                total_with_validation_s=time.perf_counter()-origin, transfer=transfer)


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(), hostname=platform.node(),
                    args=vars(args), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    memory=memory_info(),
                    versions={p:importlib.metadata.version(p) for p in ["torch", "triton", "transformers", "flash-attn", "safetensors"]},
                    gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"],text=True),
                    config=json.loads(Path(args.model,"config.json").read_text()),
                    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                        ROOT/"nanovllm_upstream/nanovllm/engine/kv_handoff.py",
                        ROOT/"nanovllm_upstream/nanovllm/engine/block_manager.py",
                        ROOT/"nanovllm_upstream/nanovllm/layers/sampler.py", Path(__file__)]},
                    limitations=["Single GPU, one model instance, independent physical blocks",
                                 "Copy timings include CPU staging and exclude TCP",
                                 "Token timestamps include extra KV validation and source poisoning",
                                 "Correctness baseline is uninterrupted nano-vllm generation, not stock vLLM"])
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    engine = LLMEngine(args.model, enforce_eager=True, max_num_seqs=1,
                       max_num_batched_tokens=args.prefill_chunk,
                       max_model_len=max(args.lengths)+args.output_tokens+256,
                       gpu_memory_utilization=args.memory_fraction,
                       distributed_init_method=f"tcp://127.0.0.1:{args.init_port}")
    rng = random.Random(args.seed)
    text = "\n".join(f"Record {i}: the measured value is {rng.randrange(10000000)}. Explain the result clearly and check the arithmetic." for i in range(max(args.lengths)))
    pool = engine.tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                   max_length=max(args.lengths))
    prompts = {n:pool[:n] for n in args.lengths}
    (out/"prompts.json").write_text(json.dumps(prompts))
    print("WARMUP", flush=True)
    run_request(engine,prompts[max(args.lengths)],args.output_tokens)
    records = []
    baselines = {}
    for length in args.lengths:
        for repeat in range(args.repeats):
            result = run_request(engine,prompts[length],args.output_tokens)
            if length in baselines:
                assert result["token_ids"] == baselines[length], "Greedy baseline changed across repeats"
            baselines[length] = result["token_ids"]
            record = dict(mode="baseline",prompt_tokens=length,repeat=repeat,**result)
            records.append(record)
            with (out/"requests.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            print(f"BASELINE L={length} repeat={repeat} tokens={len(result['token_ids'])}",flush=True)
    jobs = [(length,after,repeat) for length in args.lengths for after in args.handoff_after for repeat in range(args.repeats)]
    rng.shuffle(jobs)
    for length,after,repeat in jobs:
        result = run_request(engine,prompts[length],args.output_tokens,after)
        expected = baselines[length]
        matched = result["token_ids"] == expected
        record = dict(mode="handoff",prompt_tokens=length,handoff_after=after,repeat=repeat,
                      exact_token_match=matched,**result)
        records.append(record)
        with (out/"requests.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        if not matched:
            index = next(i for i,(a,b) in enumerate(zip(expected,result["token_ids"])) if a!=b)
            raise AssertionError(f"First differing output token {index}, L={length}, handoff_after={after}")
        t=result["transfer"]
        print(f"PASS L={length} after={after} repeat={repeat} bytes={t['payload_bytes']} "
              f"export={t['export_ms']:.2f}ms import={t['import_ms']:.2f}ms",flush=True)
    summary = dict(passed=True,baseline_requests=len(args.lengths)*args.repeats,
                   handoff_requests=len(jobs),output_tokens_per_request=args.output_tokens,
                   prompt_lengths=args.lengths,handoff_after=args.handoff_after,
                   exact_token_matches=len(jobs),source_blocks_poisoned=True,
                   all_destination_blocks_disjoint=True,
                   all_imported_payloads_bitwise_equal=True, memory=memory_info())
    (out/"validation.json").write_text(json.dumps(summary,indent=2))
    print("VALIDATION",json.dumps(summary),flush=True)
    # LLMEngine owns its atexit cleanup.


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--lengths",nargs="+",type=int,default=[255,256,257,512,2048,8192])
    parser.add_argument("--handoff-after",nargs="+",type=int,default=[1,17])
    parser.add_argument("--output-tokens",type=int,default=64)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--seed",type=int,default=20261023)
    parser.add_argument("--prefill-chunk",type=int,default=2048)
    parser.add_argument("--memory-fraction",type=float,default=.6)
    parser.add_argument("--init-port",type=int,default=2333)
    args=parser.parse_args()
    if any(n < 1 for n in args.lengths) or any(not 0 < n < args.output_tokens for n in args.handoff_after):
        parser.error("prompts must be nonempty and handoffs must precede the last output token")
    main(args)
