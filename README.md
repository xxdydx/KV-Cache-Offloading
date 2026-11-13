# agent-kv-offload

**A CPU KV-cache tier for long-horizon LLM agents, built on top of
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).**

Agents are overwhelmingly I/O bound— they call the model, wait for a tool, then call it again, as the context keeps increasing. During the wait, their KV cache sits idle on the GPU. Once the agents' combined context outgrows GPU memory, a serving engine evicts that KV and recomputes the whole prefix on the next turn. 

This project keeps the idle KV in
CPU RAM instead: it saves blocks as their GPU slots are reclaimed, stores them
in FP8, and restores them in the background while a returning agent waits for
GPU space.

On Qwen3-8B with one RTX 6000 Ada, this gives **+58–71% completed agent
trajectories per minute** over GPU-only prefix caching. It **matches or slightly
beats vLLM's native CPU offloading** on the same workload.

## Results

All numbers are means over 3 seeds, with paired 95% confidence intervals. Both
engines get the same 81,920-token GPU KV pool and 24 GiB of CPU cache, and run
the same seeded workload with fixed output lengths.

### Qwen3-8B vs vLLM 0.11

| Workload | Engine / policy | Trajectories/min | vs own GPU-only | Median TTFT |
| --- | --- | ---: | ---: | ---: |
| 4 agents × 32K, 3 turns | vLLM, GPU-only | 1.72 | | 28.2 s |
| | vLLM + `OffloadingConnector` | 2.88 | +67.4% ± 0.9 | 9.4 s |
| | **this repo, CPU tier + FP8** | **2.99** | **+71.2% ± 1.0** | **8.4 s** |
| 4 agents, 12 turns, 4K → 35K | vLLM, GPU-only | 0.93 | | 2.21 s |
| | vLLM + `OffloadingConnector` | 1.46 | +56.7% ± 2.1 | 1.87 s |
| | **this repo, CPU tier + FP8** | **1.49** | **+58.1% ± 1.0** | **1.56 s** |

Compared directly with vLLM's offloading, this repo is +3.9% ± 1.3 and
+2.5% ± 1.4 faster, winning all 3 seeds in both workloads. The two engines'
GPU-only baselines are within 1.6% of each other.

### Performance improvements for long-horizon agents

Median time to first token per turn on the 12-turn workload (Qwen3-8B):

| Turn | Prompt tokens | GPU-only | CPU tier | Prefix reused, GPU-only → CPU tier |
| ---: | ---: | ---: | ---: | ---: |
| 0–6 | 4K–21K | 0.7–1.7 s | within ±7% | the same |
| 7 | 23.8K | 4.4 s | 1.3 s (−71%) | 59% → 88% |
| 8 | 26.6K | 13.4 s | 1.1 s (−92%) | 20% → 89% |
| 11 | 35.1K | 29.6 s | 9.8 s (−67%) | 8% → 92% |

While every agent fits on the GPU, there is not much improvement with KV offloading. Once the agents outgrow
it, GPU-only loses its prefix reuse and recomputes every turn from scratch. The
CPU tier keeps about 90% of each prompt reusable.

### When it helps (Qwen3-0.6B sweep)

| Workload | Throughput vs GPU-only | Median TTFT |
| --- | ---: | ---: |
| 2 agents × 64K* | +46.8% ± 14.1 | −64% |
| 4 agents × 32K | +37.5% ± 2.4 | −56% |
| 8 agents × 16K | +23.3% ± 3.0 | −37% |
| 4 agents × 16K | +12.6% ± 1.4 | −23% |
| 12 agents × 2K (short context) | +1.4% ± 5.0 | flat |
| 1 agent × 64K (no memory pressure) | +0.2% ± 0.3 | flat |

\*Run beyond the model's trained positions using RoPE extrapolation, so this
row measures speed only.

The benefit grows with context length, memory pressure and model size. The
model size effect comes from recompute getting more expensive while a restore
costs about the same per token. When there is no pressure, the tier doesn't contribute any positive/negative change.

## How it works

```text
turn ends ──► KV blocks released on GPU (still hashed, still reusable)
                │
   GPU slot reclaimed ──► block saved to pinned CPU memory (BF16 or FP8)
                │
   next turn arrives, blocked on GPU space ──► CPU hits prefetched into spare GPU blocks
                │
   admitted ──► GPU hits (incl. prefetched) + CPU restores; only new tokens are prefilled
```

The tier is implemented in the engine (scheduler, block manager and copy
path):

- **Save on release.** A block is copied out only when its GPU slot is about to
  be reused.
- **Prefetch.** When a returning agent's request is blocked waiting for GPU
  space, its prompt is already known, so the engine hashes it block by block
  and restores the CPU hits into spare GPU blocks in the background. On
  admission those are ordinary GPU hits and need no copy. Prefetch never takes
  blocks from running requests, and it keeps one block per running request
  for decode growth. It may recycle other agents' idle cached blocks, which are
  saved to CPU first, so they move down a tier rather than being lost. It runs
  only for the blocked request at the head of the queue, never speculatively.
- **FP8 storage.** The e4m3 payload has one scale per token and KV head. That
  halves PCIe bytes and doubles CPU capacity. On Wikitext, perplexity changes
  by less than 0.3% at 2K–32K context.
- **Split copy streams.** Saves and restores run on separate CUDA streams,
  since PCIe is full-duplex: CUDA events enforce save → restore → slot-reuse ordering.
- **Hybrid admission.** The engine recomputes the head of a prefix while
  restoring the tail in parallel. The split minimises max(recompute, restore)
  using costs measured on the machine. This removes the loss on short prefixes,
  where recomputing is cheaper than copying.
- **Optional extras.** MRU eviction keeps whole released prefixes instead of
  losing their heads under LRU. A client hint (`llm.prefetch_prefix(tokens)`)
  lets an agent framework start the restore during the tool wait, before the
  request arrives, since it knows the next prompt starts with the agent's
  previous context. Hints are ignored while any request is waiting for
  admission. Both help only short contexts.

### Correctness

The copy path is asynchronous, so ordering bugs could silently corrupt KV. The
test suite injects slow copies on each CUDA stream and poisons the CPU slabs.
For each of the three cross-stream waits, a negative control shows that
removing the wait makes its test fail. Other tests check that restores are
byte-exact and that the FP8 error stays within its bound. They also cover
split admission, rollback on copy failure, and preemption, on both a scripted
CPU model and a real GPU.

## Usage

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-8B", max_model_len=40960)
cache = llm.create_cpu_prefix_cache(
    24 * 1024**3,          # CPU budget in bytes
    automatic=True,        # save/restore inside the allocator; no manual calls
    save_on_release=True,  # copy out only when a GPU slot is reclaimed
    prefetch=True,         # restore a blocked request's CPU hits into spare GPU blocks
    split_streams=True,    # separate save and restore CUDA streams
    fp8=True,              # FP8 CPU storage
)
# Use llm as usual; cache.stats reports saved, restored and prefetched blocks.
```

### Reproducing the benchmarks

```bash
# This repo's engine: GPU-only baseline vs the CPU tier
python -m bench.agent_replay --model /path/to/Qwen3-8B --out runs/cpu \
  --policy cpu_backed --cpu-cache-save-on-release --cpu-cache-prefetch \
  --cpu-cache-split-streams --cpu-cache-fp8 --cpu-cache-gib 24 \
  --num-kvcache-blocks 320 --gpu-memory-utilization 0.9 --cuda-graphs --ignore-eos \
  --num-agents 4 --turns-per-agent 12 --initial-prompt 4096 \
  --tool-result-tokens 2560 --output-tokens 256 --max-model-len 40960 --seed 7

# vLLM on the identical workload (separate venv with vllm==0.11.0)
python -m bench.vllm_replay --model /path/to/Qwen3-8B --out runs/vllm \
  --policy vllm_cpu_offload --num-gpu-blocks 5120 --cpu-cache-gib 24 \
  --gpu-memory-utilization 0.9 --cuda-graphs --ignore-eos \
  --num-agents 4 --turns-per-agent 12 --initial-prompt 4096 \
  --tool-result-tokens 2560 --output-tokens 256 --max-model-len 40960 --seed 7

# Paired comparison across seeds, and the per-turn view
python -m bench.summarize_agent_replay runs/ --baseline vllm_cpu_offload
python -m bench.summarize_turns runs/ --config <config>
```

For a GPU-only baseline, use `--policy gpu_only` or `--policy vllm_gpu_only`.
Name run directories `<config>-<policy>-s<seed>-r<rep>` so the summarizers can
pair them. Tests run on CPU, and the CUDA cases run when a GPU is present:

```bash
python -m pytest -q bench
```

## Limitations

- One GPU (RTX 6000 Ada), Qwen3-0.6B and 8B, TP=1, with synthetic tool waits
  and token IDs. These are not production traces.
- 3 seeds per cell. The seeds mostly change token IDs, so the runs are repeats
  of one workload rather than independent workloads.
- vLLM was tested at version 0.11.0 only, and its offloading API is marked
  experimental. LMCache and SGLang's HiCache were not tested.
- At 0.6B, p95 inter-token latency rises about 1–2 ms at 32K, likely from
  copies contending with decode. This hasn't been profiled.
- On these workloads, decode and the cold first turn dominate the remaining
  time, which caps how far any CPU tier can pull ahead.

## Repository layout

```text
nanovllm_upstream/nanovllm/engine/
  cpu_prefix_cache.py   CPU block store, copy streams, FP8, prefetch
  block_manager.py      Allocation, eviction (LRU/MRU), hybrid split planning
  scheduler.py          Admission, split-head prefill, restore completion
bench/
  agent_replay.py       Multi-turn agent workload on this engine
  vllm_replay.py        The same workload on vLLM AsyncLLM
  summarize_*.py        Paired-seed comparisons and per-turn tables
  fp8_kv_quality.py     FP8 KV accuracy against BF16 (Hugging Face, teacher-forced)
  test_*.py             Engine and benchmark tests
```

The engine is a modified fork of nano-vllm (MIT). Provenance and the original
license are in [nanovllm_upstream/UPSTREAM.md](nanovllm_upstream/UPSTREAM.md).
