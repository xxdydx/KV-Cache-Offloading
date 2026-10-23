"""Finite closed-loop mixed-load diagnostic, with raw client token timestamps.

80% 512-token prompts, 20% 8K prompts; fixed outputs; identical saved inputs.
This is a small capacity/latency probe, not a production p99 estimate or an
open-loop Poisson trace. Client-side semaphore waiting is outside request TTFT.
"""

import argparse
import asyncio
import importlib.metadata
import json
import random
from pathlib import Path

import aiohttp

from bench.serve_probe import Probe, pct
from bench.runtime_info import memory_info


async def run(args):
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        probe = Probe(args,session)
        metadata = dict(args=vars(args), checkpoint=str(Path(args.checkpoint).resolve()),
                        model_config=json.loads(Path(args.checkpoint,"config.json").read_text()),
                        versions={p:importlib.metadata.version(p) for p in ["vllm","torch","aiohttp"]},
                        memory=memory_info(), models=await probe.json("/v1/models"),
                        limitations=["Finite closed-loop workload, at most 8 requests in flight",
                                     "Client timings include API/CPU scheduling overhead",
                                     "Small diagnostic; no production p99 or matched-throughput comparison"])
        (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
        rng = random.Random(args.seed)
        text = "\n".join(f"Record {i}: value {rng.randrange(10000000)}. Explain the observation and check the result." for i in range(4000))
        pool = (await probe.json("/tokenize",dict(model=args.model,prompt=text,add_special_tokens=False)))["tokens"]
        lengths = [512]*(args.requests*4//5)+[8192]*(args.requests//5)
        rng.shuffle(lengths)
        trace = []
        for i,n in enumerate(lengths):
            offset = rng.randrange(len(pool)-n)
            trace.append(dict(index=i,prompt=pool[offset:offset+n],length=n))
        (out/"trace.json").write_text(json.dumps(trace))
        (out/"metrics_before.txt").write_text(await probe.metrics())
        await asyncio.gather(*(probe.request(trace[i%len(trace)]["prompt"][:512],64,"warmup","warmup",512,-1)
                               for i in range(args.concurrency)))
        trials = []
        sampler = asyncio.create_task(probe.sample_metrics())
        try:
            for repeat in range(args.repeats):
                semaphore = asyncio.Semaphore(args.concurrency)
                async def request(item):
                    async with semaphore:
                        return await probe.request(item["prompt"],args.output_tokens,"mixed",
                                                   "short" if item["length"]==512 else "long",
                                                   item["length"],repeat)
                start = probe.now()
                results = await asyncio.gather(*(request(item) for item in trace))
                end = probe.now()
                elapsed = end-start
                gaps = [(b["t"]-a["t"])*1000 for r in results for a,b in zip(r["events"],r["events"][1:])]
                bundled = [r["request_id"] for r in results if any(len(e["token_ids"])!=1 for e in r["events"])]
                ttfts = [(r["events"][0]["t"]-r["sent"])*1000 for r in results]
                trial = dict(repeat=repeat,window_start_s=start,window_end_s=end,elapsed_s=elapsed,requests=len(results),
                             output_tokens=len(results)*args.output_tokens,
                             output_tokens_per_s=len(results)*args.output_tokens/elapsed,
                             ttft_p50_ms=pct(ttfts,50),ttft_p99_ms=pct(ttfts,99),
                             pooled_itl_p99_ms=pct(gaps,99) if not bundled else None,
                             max_itl_ms=max(gaps) if not bundled else None,
                             requests_with_bundled_events=bundled)
                trials.append(trial)
                (out/"trials.json").write_text(json.dumps(trials,indent=2))
                print("MIXED",json.dumps(trial),flush=True)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler,return_exceptions=True)
            (out/"scheduler_samples.json").write_text(json.dumps(probe.samples))
            (out/"metrics_after.txt").write_text(await probe.metrics())
            (out/"memory_after.json").write_text(json.dumps(memory_info(),indent=2))
        probe.save_summary()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",default="http://127.0.0.1:8002")
    p.add_argument("--model",default="qwen3-8b")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--requests",type=int,default=20)
    p.add_argument("--concurrency",type=int,default=8)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--output-tokens",type=int,default=128)
    p.add_argument("--seed",type=int,default=20261023)
    args = p.parse_args()
    if args.requests<5 or args.requests%5 or min(args.concurrency,args.repeats,args.output_tokens)<1:
        p.error("requests must be a positive multiple of five; other counts must be positive")
    asyncio.run(run(args))
