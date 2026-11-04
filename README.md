# Agent KV cache management on one GPU

**Question: when should a serving engine move an inactive agent's KV cache to
CPU RAM so other agents can use the GPU?** Extend the existing nano-vllm engine
and compare GPU retention, eviction/recomputation, and CPU preservation across
multi-turn workloads.

Weights, prefill, and active decode attention stay on the GPU. CPU memory stores
reusable state while an agent waits for a tool. Full-block CPU save/restore and
opt-in automatic preservation/restoration are implemented. Agent replay and
policy/performance evaluation remain to be implemented. Opt-in prefill scheduling
policies also interleave decode with bounded prefill chunks. The detailed implementation
plan and acceptance gates are in `spec.md` (locally gitignored).

## What the project does

Agents repeatedly call a model, wait for tools, append results, and call the
model again. Their contexts grow, and keeping every inactive context on the GPU
can limit how many agents the system can serve.

```text
GPU: generate a turn → retain its valid KV prefix
                              ↓
Tool wait: keep KV on GPU / preserve on CPU / evict it
                              ↓
Tool result: restore or recompute prefix → prefill new input → GPU generation
```

Preserving CPU state may reduce recomputation and free GPU cache blocks, but
transfers and restoration can increase latency. Measure the conditions where
that tradeoff helps, including cases where GPU retention or recomputation wins.
CPU attention is no longer the implementation target.

## CPU block cache

Nano-vllm hashes full KV blocks by token prefix. Completion releases request
references but leaves reusable block data and hash metadata until another
allocation recycles those blocks. Matching later requests can reuse them. The
default blocks contain 256 tokens; partial tails are recomputed, and admission
leaves the last prompt block for processing. Basic cross-turn reuse therefore
does not require a new cache handle.

Enable automatic preservation and restoration once on an initialized TP=1 engine:

```python
cpu_cache = llm.create_cpu_prefix_cache(max_bytes=8 * 1024**3, automatic=True)
```

The allocator saves valid idle blocks before recycling their GPU slots. Request
admission checks GPU and CPU prefixes, protects every GPU hit, and restores CPU
hits into blocks owned by the new request. No manual save/restore calls are needed.
Copies are synchronous. Active requests still need enough GPU KV space.

The CPU store uses a fixed payload budget. Once full, it keeps existing entries
and skips additional saves; GPU eviction and recomputation continue normally.
It does not implement CPU LRU eviction yet. `cpu_cache.stats` reports cumulative
`saved_blocks`, `restored_blocks`, and `budget_skips`. `close()` disables the
hooks and releases the CPU store; engine shutdown also closes it. Unexpected
copy errors propagate with admission reservations/queues rolled back.

### Manual operations

Without `automatic=True`, the cache remains manually managed.
Save a published full block before its GPU slot is reused, then restore it by
prefix hash. Each block includes both K/V and every model layer. For an already
initialized TP=1 `llm`, this synthetic example saves the first full block:

```python
from nanovllm import SamplingParams

manager = llm.scheduler.block_manager
prompt_ids = [1] * (manager.block_size + 1)
llm.generate([prompt_ids], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
key = manager.compute_hash(prompt_ids[:manager.block_size])

cpu_cache = llm.create_cpu_prefix_cache(max_bytes=64 * 1024**2)
cpu_cache.save_block(key)
# Other requests may now recycle the original GPU block.
cpu_cache.restore_block(key)
cpu_cache.discard(key)  # Release the CPU copy when no longer needed.
```

Call between engine steps on the owning thread. Restoring an existing GPU hit
is a no-op; a restored entry remains normally recyclable. The CPU copy is kept
until `discard`, `clear`, or engine shutdown. The byte budget covers KV tensors,
not Python metadata; exceeding it raises `MemoryError` without evicting entries.
Copy failures release reservations and do not publish incomplete GPU state.
Tests in [test_cpu_prefix_cache.py](bench/test_cpu_prefix_cache.py) and
[test_automatic_cpu_cache.py](bench/test_automatic_cpu_cache.py) force block
recycling, check exact bytes, and verify reduced next-turn prefill. The CUDA
case runs when a device is available; CPU tests use a scripted model runner.

Read these files in order, under `nanovllm_upstream/nanovllm/`:

1. [`engine/llm_engine.py`](nanovllm_upstream/nanovllm/engine/llm_engine.py): requests, token events, existing handoff boundaries.
2. [`engine/scheduler.py`](nanovllm_upstream/nanovllm/engine/scheduler.py): scheduling, completion, and release.
3. [`engine/block_manager.py`](nanovllm_upstream/nanovllm/engine/block_manager.py): block references, reuse, and allocation.
4. [`engine/kv_handoff.py`](nanovllm_upstream/nanovllm/engine/kv_handoff.py): valid KV export/import and snapshot validation.
5. [`engine/model_runner.py`](nanovllm_upstream/nanovllm/engine/model_runner.py): cache layout and GPU execution.

The next steps are real-GPU validation, two-agent replay, and policy/performance evaluation. The
implementation belongs in the engine and a minimal replay harness; no external
agent framework or second GPU is needed.

## Prefill scheduling

A long prompt can delay requests already generating tokens. The existing engine
chunks prefill at `max_num_batched_tokens` but prioritizes waiting prefills.
Three policies are now selectable at engine construction:

| `prefill_policy` | Behavior |
| --- | --- |
| `legacy` (default) | Existing prefill priority and hard batch-token limit |
| `fixed` | Fixed prefill chunk budget; give decode a batch between prefill batches |
| `decode_aware` | Same interleaving; shrink the chunk budget as runnable decode count grows |

```python
from nanovllm import LLM

llm = LLM(
    "/path/to/pinned/Qwen3-0.6B",
    prefill_policy="decode_aware",
    prefill_chunk_size=512,
    prefill_min_chunk_size=32,
    max_num_batched_tokens=4096,
)
```

The adaptive budget is `max(min_chunk, chunk_size // (1 + runnable_decoders))`,
capped by `max_num_batched_tokens`. Both opt-in policies alternate **separate**
prefill and decode batches when both phases can progress; they do not combine
phases in one kernel launch. Without runnable decoders, prefills run consecutively.
Decode batches rotate when there are more resident requests than batch slots.
Under KV pressure, partial-prefill reservations can be reclaimed for decode
growth; that request may need to recompute its prefix. Long requests retain FIFO
prefill order and progress when capacity permits. `scheduler.last_schedule`
records the phase, chosen budget, decode load, tokens, and request IDs.

These are token-budget heuristics, not wall-clock latency guarantees. Context
length, batch size, and synchronous CPU-cache transfers still affect step time.
Tests cover chunk limits, interleaving, progress, memory pressure, output events,
CPU-cache integration, and copy-failure retries. GPU latency results are pending.

Run the mixed short/long arrival benchmark in a fresh GPU process for each policy:

```bash
CUDA_VISIBLE_DEVICES=0 python -m bench.prefill_scheduling \
  --model /path/to/pinned/Qwen3-0.6B \
  --policy decode_aware --chunk-size 512 --min-chunk-size 32 \
  --out results/prefill-decode-aware-01
```

Repeat with `--policy legacy` and `--policy fixed`, using new output directories
and identical workload/resource arguments. Sweep chunk size and repeat each
configuration before drawing tail-latency conclusions. The harness records the
seeded workload/hash, hardware/source manifest, raw scheduling/token events,
per-group TTFT and inter-token percentiles, completion/wait latency, and achieved
throughput. Arrival delay during a running step counts toward latency. Verify
`contended_prefill_steps` is nonzero; otherwise increase initial output length or
reduce the arrival delay. All requests must complete within the timeout.

This scheduling benchmark checks worst-case KV capacity up front to isolate
scheduling from memory-pressure preemption. Its synthetic independent requests
do not measure multi-turn agent cache benefits. CPU-cache and agent-replay
experiments remain separate evaluations.

## Starting files

```text
nanovllm_upstream/       Existing engine, upstream docs, license, provenance
bench/inventory.py      Hardware and visible sandbox-limit manifest
bench/baseline.py       GPU-only phase diagnostic and fixed-batch controls
bench/workload_sweep.py GPU load / KV memory diagnostic (not an agent replay)
bench/prefill_scheduling.py Mixed-length arrivals and scheduling-policy metrics
bench/metrics.py        Reusable token-latency calculations
bench/test_*.py         Engine and measurement regression tests
spec.md                Active plan plus collapsed historical specifications
results/               Preserved evidence, source snapshots, and new runs
```

The standalone CPU-attention probe and its tests were removed after a verified
backup. Recovery paths and hashes are in
[the scope-change manifest](results/2026-11-04-agent-kv-scope/manifest.json).
The earlier multi-GPU drivers/cost model were already archived. Keep engine
handoff APIs and their tests: their ownership and transfer checks are useful
building blocks for the new cache manager.

## Hardware and existing evidence

The completed November 4 check used an RTX 6000 Ada (49,140 MiB reported VRAM),
Ryzen 9 9900X (12 physical cores / 24 logical CPUs), and **49 GiB effective cgroup
RAM**, even though the host reported about 123 GiB. That task and the earlier
home RTX 4000 task are stopped; recheck any new allocation. The Mac is for
editing, CPU tests, analysis, and backups.

Qwen3-0.6B BF16 inference and transfers ran successfully. PyTorch GPU timelines
work; hardware-counter profiling was denied with `ERR_NVGPUCTRPERM`. The prior
[feasibility report](results/2026-11-04-office-rtx-feasibility/report.md) contains
GPU baselines, the slower CPU-attention result, raw data, and source snapshots.
It does not demonstrate a benefit from agent cache management. Restore its
source snapshot into a separate directory to reproduce removed experiments.

On the next GPU host, first save inventory and run a short GPU diagnostic using
the pinned checkpoint. Prepare a compatible CUDA environment using the
[recorded dependency setup](results/2026-11-04-office-rtx-feasibility/reproduce.md).
From the repository root, in that environment:

```bash
python -m bench.inventory --out results/agent-kv-inventory-01.json

CUDA_VISIBLE_DEVICES=0 python -m bench.baseline \
  --model /path/to/pinned/Qwen3-0.6B \
  --out results/agent-kv-gpu-baseline-01 \
  --batch-size 1 --prompt-tokens 128 --output-tokens 32
```

Use a new output path per run. These are diagnostic tools. Agent replay and
cross-turn cache behavior remain the work to implement.

## Evaluation and local checks

Use the existing GPU-only prefix cache as the main baseline. Forced eviction /
recomputation is an explicit diagnostic control, not a substitute for that
baseline. Compare policies on identical growing contexts, tool waits, arrivals,
model/precision, and resource budgets. Primary metrics are completed trajectories
per minute, end-to-end agent latency, turn/resume latency, recomputed input,
bytes moved, and live/free cache blocks. Count queueing and all failures. Include
low-pressure and short-wait cases; a throughput gain is a hypothesis.

Run existing CPU regressions from the root in an environment with PyTorch,
Transformers, xxhash, NumPy, tqdm, and pytest:

```bash
python3 -m pytest -q bench
```

These tests cover existing engine and measurement behavior. New cross-turn
retention, CPU-tier policy, and agent-harness checks must be added during
implementation. See [upstream provenance](nanovllm_upstream/UPSTREAM.md).
`spec.md` and new `results/` stay locally ignored; package them explicitly for
sharing. Original source and all historical results remain attributed.
