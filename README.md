# Agent KV cache management on one GPU

**Question: when should a serving engine move an inactive agent's KV cache to
CPU RAM so other agents can use the GPU?** Extend the existing nano-vllm engine
and compare GPU retention, eviction/recomputation, and CPU preservation across
multi-turn workloads.

Weights, prefill, and active decode attention stay on the GPU. CPU memory stores
reusable state while an agent waits for a tool. The new CPU-backed cache tier
and agent replay harness are **not implemented yet**. The detailed implementation
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

## First implementation task

**Measure the prefix caching that already exists.** Run a completed turn, submit
an extended prompt, and count reused versus newly prefilled tokens. Then create
GPU-block pressure and observe when that reuse disappears.

Nano-vllm hashes full KV blocks by token prefix. Completion releases request
references but leaves reusable block data and hash metadata until another
allocation recycles those blocks. Matching later requests can reuse them. The
default blocks contain 256 tokens; partial tails are recomputed, and admission
leaves the last prompt block for processing. Basic cross-turn reuse therefore
does not require a new cache handle.

The extension is a CPU-backed tier that preserves selected reusable prefixes
before GPU recycling destroys them, then restores them for later turns. Start
with full blocks and recompute tails. Explicit session pinning and partial-tail
reuse are optional later features. The existing detach/resume API is useful
transfer machinery, but it handles an unfinished request and cannot simply
export an already-finished request's released blocks.

Read these files in order, under `nanovllm_upstream/nanovllm/`:

1. [`engine/llm_engine.py`](nanovllm_upstream/nanovllm/engine/llm_engine.py): requests, token events, existing handoff boundaries.
2. [`engine/scheduler.py`](nanovllm_upstream/nanovllm/engine/scheduler.py): scheduling, completion, and release.
3. [`engine/block_manager.py`](nanovllm_upstream/nanovllm/engine/block_manager.py): block references, reuse, and allocation.
4. [`engine/kv_handoff.py`](nanovllm_upstream/nanovllm/engine/kv_handoff.py): valid KV export/import and snapshot validation.
5. [`engine/model_runner.py`](nanovllm_upstream/nanovllm/engine/model_runner.py): cache layout and GPU execution.

The next gates are synchronous CPU preservation, two-agent replay with the three
policies, pressure/wait-duration sweeps, and one profiled optimization. The
implementation belongs in the engine and a minimal replay harness; no external
agent framework or second GPU is needed.

## Starting files

```text
nanovllm_upstream/       Existing engine, upstream docs, license, provenance
bench/inventory.py      Hardware and visible sandbox-limit manifest
bench/baseline.py       GPU-only phase diagnostic and fixed-batch controls
bench/workload_sweep.py GPU load / KV memory diagnostic (not an agent replay)
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
