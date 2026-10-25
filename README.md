# disaggregated-inference

Adding prefill/decode-disaggregated serving to nano-vllm, optimizing KV transfers,
and comparing the result against colocated nano-vllm and vLLM.

The modified engine is included directly in `nanovllm_upstream/` as ordinary
tracked source. This is one repository: a normal clone includes both the engine
and the experiment code, with no submodule setup.

Based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm),
starting at commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`. See
[upstream provenance and local changes](nanovllm_upstream/UPSTREAM.md).
The original [MIT license](nanovllm_upstream/LICENSE) is retained.

## Tracking changes from stock nano-vllm
```bash
# List engine changes since the stock snapshot, oldest first.
git log --reverse --oneline nanovllm-stock..HEAD -- nanovllm_upstream/

# See the complete engine source difference from stock.
git diff nanovllm-stock HEAD -- nanovllm_upstream/nanovllm/
```

## Per-step token events

With the modified engine installed and a local model checkpoint, requests can
be tracked by their process-local sequence IDs and consumed one step at a time:

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model")
request_id = llm.add_request("Explain KV caching.", SamplingParams(temperature=0))
while not llm.is_finished():
    result = llm.step_with_events()
    for event in result.token_events:
        print(event.seq_id, event.output_index, event.token_id, event.finished)
```

Output indices are zero-based within each completion. Intermediate prefill
chunks emit no token events; the final prefill chunk emits the first output
token. The final token has `finished=True` when EOS or the output limit ends
the request. `result.outputs` contains completed token sequences, and
`result.num_tokens` retains the existing positive-prefill/negative-decode work
count. Existing `step()` and `generate()` callers keep their return formats.
Each call to either step method advances inference by one batch; token events
report only the new output from the `step_with_events()` call that produced it.

## Engine handoff methods

Each worker uses its own process and a TP=1 engine. Both workers must load the
same checkpoint and use matching KV layout, dtype, and block size.

| Method | Behavior |
| --- | --- |
| `detach_request(seq_id)` | Export a CPU `KVSnapshot` and stop scheduling that request, retaining its source blocks |
| `resume_request(snapshot)` | Import KV, wait for CUDA copy completion, admit the request for decode, and return its destination-local ID |
| `release_detached_request(seq_id)` | Release retained source blocks; an independently completed CPU snapshot remains valid |
| `reattach_request(seq_id)` | Resume scheduling the original source sequence with the same ID and existing KV |

Detach between engine steps after an unfinished request has generated at least
one output token. Queued, partially prefilled, finished, and already detached
requests cannot be detached. Failed export leaves the request scheduled;
failed import does not admit a request and releases newly allocated blocks.
Call these methods from the engine-owning thread between inference steps.

The minimal benchmark releases source blocks as soon as CPU export completes:

```text
Source:       snapshot = source.detach_request(source_id)
Source:       source.release_detached_request(source_id)
Transport:    deliver snapshot to the destination
Destination:  destination_id = destination.resume_request(snapshot)
Transport:    acknowledge destination readiness to the source
```

The destination continues output indices and the total output-token budget
from the snapshot. Tokens generated before handoff are included in completed
outputs but are not replayed as new token events. Source and destination IDs
are local identifiers; a coordinator must track their relationship.

Reattach only when the destination is known not to have accepted the request;
reattaching after an uncertain delivery can execute the request twice. Release
and reattach consume the detached handle and reject repeated calls. Engine
shutdown also releases retained source blocks. `is_finished()` describes the
local scheduling queues, so it can be true while detached blocks remain owned.

These methods provide the engine boundary. The two-request prototype below
adds worker loops around them. The serial benchmark remains the reference.

## Minimal concurrent handoff probe

This learning prototype runs two persistent TP=1 workers on one CUDA host.
GPU 0 prefills each request and sends its first token plus a KV snapshot;
GPU 1 imports the snapshot and streams the remaining output tokens. A socket
thread receives CPU data while the destination's owning thread advances decode.
Only the owning thread calls the engine. The engine source is unchanged by
this prototype.

Start with a small local Qwen3 checkpoint in the nano-vllm CUDA environment:

```sh
python -m bench.disagg_probe --model /path/to/Qwen3-0.6B \
  --out /tmp/disagg-smoke --gpus 0 1 --lengths 255 257 --output-tokens 32

# Larger diagnostic after the smoke test:
python -m bench.disagg_probe --model /path/to/Qwen3-8B \
  --out /tmp/disagg-8b --gpus 0 1 --lengths 512 8192 --output-tokens 128
```

Use a fresh output directory per run. Inputs are seeded synthetic token IDs;
both requests arrive together after worker startup. BF16, eager execution,
greedy sampling, and `ignore_eos=True` are fixed. Prompts are limited to 8K and
outputs to 2–256 tokens. One transfer is in flight at a time. Before prefill,
the destination grants capacity for the request's eventual full KV growth,
waiting for existing decodes to finish if needed. This can intentionally
serialize requests when the pool is too small.

The source releases its GPU blocks after completed CPU export, then drops its
CPU payload after synchronous send completes. The receiver drops its payload
after completed import. The default 3 GiB payload budget covers both host
copies; cgroup memory preflight also reserves headroom and accounts for inactive
file cache. These estimates do not guarantee allocation success or bound all
process/OS memory. Socket timeout, import error, cancellation via interrupt,
or worker failure aborts the trial and terminates workers. There is no retry,
source fallback, reconnect, or transparent crash recovery.

Outputs include `inputs.json`, `manifest.json`, raw `events.jsonl`,
`summary.json`, and a Chrome/Perfetto `timeline.json`. Token indices retain the
handoff gap in ITL. Summary latency includes source admission waiting; throughput
includes drain. `passed` checks output accounting and successful block cleanup;
it does not establish token equality with an uninterrupted GPU baseline.
Failed/aborted trials remain recorded and return a nonzero exit status.

Host step overlap is only a candidate for GPU overlap. Local CUDA durations
and NVTX ranges are included; use a GPU profiler before declaring the M1 GPU
concurrency gate passed. Per-step CUDA events/synchronization add instrumentation
overhead. This is a two-request mechanism diagnostic, not a sustained-capacity
or competitive performance result. No GPU run of this prototype has been
validated in the current CPU-only development environment.

For learning, read in this order:

1. [Message framing and validation](bench/disagg_protocol.py): `OFFER → PERMIT → TRANSFER → READY`, then `STOP`.
2. [Prefill loop](bench/disagg_probe.py): `source_loop()` calls detach, release, and send.
3. [Decode loop](bench/disagg_probe.py): `decode_loop()` owns the engine; `receive_loop()` owns the socket.
4. [CPU tests](bench/test_disagg_probe.py): bit preservation, token continuity, capacity, and failures.

The benchmark driver represents submission with `SUBMIT` events. Token and
completion events travel over process pipes, not the raw KV socket. An extra
`PERMIT` exchange bounds destination/host allocations; `READY` confirms import.

```sh
python3 -m pytest -q bench/test_disagg_probe.py bench/test_engine_events.py \
  bench/test_engine_handoff.py bench/test_kv_wire.py bench/test_metrics.py \
  bench/test_sampler.py costmodel/test_model.py
```

## Research Question

**Question:** when does shipping KV state between machines beat keeping a request on one GPU, and how far can a purpose-built transfer path cut that cost?

- Model: Qwen3-8B (BF16, ~144 KB KV/token)
- Development pairing: two RTX 6000 Ada GPUs (48 GB each) on one host

## Layout

| Path | Contents |
| --- | --- |
| `nanovllm_upstream/` | Vendored nano-vllm engine, including local handoff and sampling changes |
| `microbench/` | Week-1 network and hardware microbenchmarks (iperf3, ping-pong, PCIe, matmul, power) |
| `costmodel/` | Analytical disaggregation cost model and tests |
| `results/` | Experiment data and figures; new artifacts are currently gitignored |
| `bench/` | Scripts that regenerate every figure from `results/` |

