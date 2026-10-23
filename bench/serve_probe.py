"""Probe an existing vLLM server without changing its configuration.

Run inside the GPU host: python serve_probe.py --out /tmp/serve-probe
Requires aiohttp. All timing is measured on one loopback client clock.
"""

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

try:
    from bench.runtime_info import memory_info
except ModuleNotFoundError:  # Also support python /path/to/bench/serve_probe.py.
    from runtime_info import memory_info


def pct(values, q):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


class Probe:
    def __init__(self, args, session):
        self.args, self.session = args, session
        self.origin = time.perf_counter()
        self.records = []
        self.samples = []
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)

    def now(self):
        return time.perf_counter() - self.origin

    async def json(self, path, payload=None):
        method = self.session.get if payload is None else self.session.post
        async with method(self.args.url + path, **({} if payload is None else {"json": payload})) as response:
            response.raise_for_status()
            return await response.json()

    async def metrics(self):
        async with self.session.get(self.args.url + "/metrics") as response:
            response.raise_for_status()
            return await response.text()

    async def sample_metrics(self):
        while True:
            content = await self.metrics()
            chosen = {}
            for line in content.splitlines():
                if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
                    name, value = line.rsplit(" ", 1)
                    chosen[name] = float(value)
            self.samples.append({"t": self.now(), **chosen})
            await asyncio.sleep(0.05)

    async def request(self, prompt, output, scenario, role, length, repeat, ready=None):
        record = dict(request_id=uuid.uuid4().hex, scenario=scenario, role=role,
                      prompt_length=length, repeat=repeat, requested_output=output,
                      prompt_token_ids=prompt, sent=self.now(), events=[], usage=None)
        payload = dict(model=self.args.model, prompt=prompt, max_tokens=output,
                       min_tokens=output, ignore_eos=True, temperature=0,
                       stream=True, stream_options={"include_usage": True},
                       return_token_ids=True, add_special_tokens=False,
                       cache_salt=uuid.uuid4().hex + uuid.uuid4().hex)
        async with self.session.post(self.args.url + "/v1/completions", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
            # aiohttp's line iterator handles arbitrary TCP fragmentation.
            async for raw in response.content:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event.get("usage"):
                    record["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    ids = choice.get("token_ids") or []
                    if ids:
                        record["events"].append({"t": self.now(), "token_ids": ids})
                        if ready and sum(len(e["token_ids"]) for e in record["events"]) >= 16:
                            ready.set()
        record["done"] = self.now()
        ids = [i for event in record["events"] for i in event["token_ids"]]
        if len(ids) != output:
            raise RuntimeError(f"Expected {output} tokens, received {len(ids)}: {record}")
        if not record["usage"] or record["usage"]["prompt_tokens"] != len(prompt):
            raise RuntimeError(f"Unexpected prompt token accounting: {record['usage']}")
        if record["usage"]["completion_tokens"] != output:
            raise RuntimeError(f"Unexpected completion token accounting: {record['usage']}")
        details = record["usage"].get("prompt_tokens_details") or {}
        if details.get("cached_tokens", 0):
            raise RuntimeError(f"Unexpected prefix cache hit: {details}")
        self.records.append(record)
        with (self.out / "requests.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if ready:
            ready.set()  # Unblocks the driver even if a very short request finishes.
        times = [e["t"] for e in record["events"]]
        print(f"{scenario:18s} {role:8s} L={length:5d} repeat={repeat} "
              f"TTFT={(times[0]-record['sent'])*1000:.1f}ms "
              f"total={record['done']-record['sent']:.2f}s", flush=True)
        return record

    def save_summary(self):
        rows = []
        for record in self.records:
            times = [e["t"] for e in record["events"]]
            single = all(len(e["token_ids"]) == 1 for e in record["events"])
            gaps = [1000 * (b-a) for a, b in zip(times, times[1:])]
            rows.append({k: record[k] for k in ("request_id", "scenario", "role", "prompt_length", "repeat")})
            rows[-1].update(ttft_ms=1000*(times[0]-record["sent"]),
                            mean_tpot_ms=1000*(times[-1]-times[0])/(record["requested_output"]-1),
                            p99_itl_ms=pct(gaps, 99) if single else float("nan"),
                            max_itl_ms=max(gaps) if single else float("nan"),
                            total_s=record["done"]-record["sent"],
                            output_tokens=record["requested_output"], stream_events=len(times))
        with (self.out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    async def run(self):
        metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                        hostname=platform.node(), python=platform.python_version(),
                        args=vars(self.args), models=await self.json("/v1/models"),
                        versions={p: importlib.metadata.version(p) for p in ("torch", "vllm", "transformers", "aiohttp")},
                        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.sm,clocks.mem,power.limit", "--format=csv"], text=True),
                        server_processes=subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True),
                        limitations=["Interpret concurrency from server_processes and scheduler samples",
                                     "Model/dtype/quantization are determined by the captured server command",
                                     "Client-observed loopback token timings include API overhead",
                                     "GPU clocks are recorded but not locked"],
                        description=self.args.description,
                        memory=memory_info(),
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if self.args.checkpoint:
            checkpoint = Path(self.args.checkpoint)
            metadata["checkpoint"] = str(checkpoint.resolve())
            metadata["model_config"] = json.loads((checkpoint / "config.json").read_text())
        # Keep only the server command, excluding other clients' process details.
        metadata["server_processes"] = [line for line in metadata["server_processes"].splitlines() if "vllm serve" in line]
        (self.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (self.out / "metrics_before.txt").write_text(await self.metrics())
        seed = random.Random(self.args.seed)
        text = "\n".join(f"Record {i}: value {seed.randrange(10000000)}. The system measures latency and processes independent requests." for i in range(4000))
        pool = (await self.json("/tokenize", dict(model=self.args.model, prompt=text, add_special_tokens=False)))["tokens"]
        if len(pool) < max(self.args.lengths) + 1024:
            raise RuntimeError("Prompt pool is too short")
        def prompt(length):
            offset = seed.randrange(len(pool)-length)
            return pool[offset:offset+length]
        await self.request(prompt(max(self.args.lengths)), 32, "warmup", "warmup", max(self.args.lengths), -1)
        await self.request(prompt(512), 128, "warmup", "warmup", 512, -1)
        sampler = asyncio.create_task(self.sample_metrics())
        try:
            jobs = [(length, repeat) for repeat in range(self.args.repeats) for length in self.args.lengths]
            seed.shuffle(jobs)
            for length, repeat in jobs:
                await self.request(prompt(length), 128, "solo", "target", length, repeat)
            if not self.args.solo_only:
                for length, repeat in jobs:
                    blocker = asyncio.create_task(self.request(prompt(length), 128, "queued_short", "blocker", length, repeat))
                    await asyncio.sleep(0.02)
                    probe = asyncio.create_task(self.request(prompt(512), 64, "queued_short", "probe", length, repeat))
                    await asyncio.gather(blocker, probe)
                for length, repeat in jobs:
                    decoder_prompt = prompt(512)
                    if self.args.paired_decoder_baselines:
                        await self.request(decoder_prompt, 128, "decoder_control", "decoder", length, repeat)
                    ready = asyncio.Event()
                    decoder = asyncio.create_task(self.request(decoder_prompt, 128, "inject_long", "decoder", length, repeat, ready))
                    # Race completion against readiness so an HTTP failure cannot deadlock.
                    waiter = asyncio.create_task(ready.wait())
                    await asyncio.wait([decoder, waiter], return_when=asyncio.FIRST_COMPLETED)
                    if decoder.done():
                        await decoder
                    waiter.cancel()
                    injected = asyncio.create_task(self.request(prompt(length), 32, "inject_long", "injected", length, repeat))
                    await asyncio.gather(decoder, injected)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            (self.out / "scheduler_samples.json").write_text(json.dumps(self.samples))
            (self.out / "metrics_after.txt").write_text(await self.metrics())
            (self.out / "memory_after.json").write_text(json.dumps(memory_info(), indent=2))
        self.save_summary()
        print(f"Saved {len(self.records)} requests to {self.out}", flush=True)


async def main(args):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        await Probe(args, session).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="local")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048, 8192, 16384])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--solo-only", action="store_true")
    parser.add_argument("--paired-decoder-baselines", action="store_true",
                        help="Measure the same decoder input alone before each injection pair")
    parser.add_argument("--checkpoint", help="Local checkpoint directory to record configuration/revision")
    parser.add_argument("--description", default="See server command for model and cache configuration")
    args = parser.parse_args()
    if Path(args.out, "requests.jsonl").exists():
        parser.error("Choose a new output directory; requests.jsonl already exists")
    asyncio.run(main(args))
