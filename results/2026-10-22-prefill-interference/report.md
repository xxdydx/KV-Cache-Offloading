# Serialized versus concurrent serving

The same Qwen3.5-9B GPTQ INT4 model ran on one RTX 6000 Ada with FP8 KV and eager execution. The independent experiment server changed max_num_seqs from 1 to 8 and port from 8001 to 8002; all other launch arguments were preserved.

Each result is the median of three repetitions. Only two benchmark requests overlap; eight is the configured capacity, not measured batch size.

| Long prompt | Decoder max gap, seqs=1 (ms) | Decoder max gap, seqs=8 (ms) | Short probe TTFT, seqs=1 (ms) | Short probe TTFT, seqs=8 (ms) |
|---:|---:|---:|---:|---:|
| 512 | 15.8 | 47.4 | 1711.1 | 131.1 |
| 2,048 | 14.2 | 142.9 | 1919.7 | 307.8 |
| 8,192 | 14.5 | 157.3 | 2882.6 | 378.0 |
| 16,384 | 16.7 | 171.7 | 4152.6 | 387.2 |

The injection test adds the long request after the existing decoder emits 16 of its 128 tokens. The queued-short test submits a 512-token/64-output-token probe 20 ms after a long-prompt/128-output-token request.

The concurrent server's startup log confirms enable_chunked_prefill=True. The pauses occur with chunking enabled; this experiment does not compare chunking on versus off.

The maximum token gap exposes an individual freeze that p99 can miss when it occurs in fewer than 1% of gaps. This is a small controlled experiment, not a production p99 estimate.

These runs measure single-GPU scheduling behavior. They do not establish disaggregation speedup or a network break-even point. The project spec uses BF16 Qwen3-8B; its KV-size formula should not be applied directly to this Qwen3.5 model.

The original service was restored after the concurrent run; see restoration.json. Raw token IDs/timestamps, scheduler samples, metadata, and counter-validation files live in the two input run directories.

## Regenerate

```sh
python3 -m bench.compare_serve_probes results/2026-10-22-rtx6000ada-qwen35-9b results/2026-10-22-rtx6000ada-qwen35-9b-concurrent results/2026-10-22-prefill-interference
```
