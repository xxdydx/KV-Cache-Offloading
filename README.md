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
| `release_detached_request(seq_id)` | Release the retained source blocks after destination acceptance, or when abandoning the request |
| `reattach_request(seq_id)` | Resume scheduling the original source sequence with the same ID and existing KV |

Detach between engine steps after an unfinished request has generated at least
one output token. Queued, partially prefilled, finished, and already detached
requests cannot be detached. Failed export leaves the request scheduled;
failed import does not admit a request and releases newly allocated blocks.
Call these methods from the engine-owning thread between inference steps.

The handoff order across the two worker processes is:

```text
Source:       snapshot = source.detach_request(source_id)
Transport:    deliver snapshot to the destination
Destination:  destination_id = destination.resume_request(snapshot)
Transport:    acknowledge destination readiness to the source
Source:       source.release_detached_request(source_id)
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

These methods provide the engine boundary. Persistent worker loops, TCP
readiness messages, bounded admission, and a streaming coordinator remain
separate implementation steps. The serial benchmark remains the reference.

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

