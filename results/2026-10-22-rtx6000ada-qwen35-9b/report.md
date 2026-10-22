# Serving experiment: max_num_seqs=1

RTX 6000 Ada (48 GB); InfiniCloud/Qwen3.5-9B-GPTQ-INT4-G128; FP8 KV cache; eager execution; max_num_seqs=1.

Each point has 3 repetitions. Table values are medians; figures show the observed min–max range. No prefix reuse (unique cache salt per request).

| Prompt tokens | Solo TTFT (ms) | Solo mean TPOT (ms) | Solo p99 token gap (ms) | Queued short TTFT (ms) | Active decoder p99 gap (ms) |
|---:|---:|---:|---:|---:|---:|
| 512 | 65.9 | 12.7 | 14.7 | 1711.1 | 14.4 |
| 2,048 | 249.5 | 12.7 | 14.2 | 1919.7 | 14.0 |
| 8,192 | 1017.8 | 12.8 | 14.7 | 2882.6 | 14.4 |
| 16,384 | 2138.9 | 12.9 | 14.9 | 4152.6 | 16.2 |

## Interpretation and limits

- The queued-short test submits a 512-token/64-output-token probe 20 ms after a long-prompt/128-output-token request. Its TTFT includes the blocker's prefill AND remaining decode; this is not an isolated prefill measurement.
- The injection test starts a 512-token/128-output-token decoder, then submits the long prompt after the decoder emits 16 tokens. The injected request generates 32 tokens.
- max_num_seqs=1 serializes requests. This run cannot establish concurrent prefill/decode interference.
- This run does not measure network transfer cost or disaggregation speedup. A separate chunking ablation is needed to establish chunking benefits.
- p99 token gap is computed within each request, then summarized by its median across repeats. It is not p99 of request-mean TPOT, nor a statistically stable production tail estimate.
- Client timings include local HTTP/streaming overhead. Requests used greedy sampling, ignored EOS, and returned exactly the requested number of output tokens. 1 request(s) had multi-token streaming events; their per-token gap metrics are omitted (NaN). All plotted token-gap measurements require single-token events.
- Existing prefix caching is enabled but a unique salt prevents reuse across our requests. Warmup used the largest prompt and a separate 128-token decode.
- The existing service remains available to other clients, so exclusive load is not guaranteed. Scheduler samples and cumulative metrics are included for inspection.
- Validation: 62 requests, 5,920 output tokens, all requested counts matched, and no reported prefix-cache hits. Server counters match the probe exactly. Sampled maximum running/waiting: 1/1.
- Qwen3.5-9B with quantized weights and FP8 KV differs from the spec's BF16 Qwen3-8B; do not fit that model's KV-byte assumptions to these results.

## Reproduce

On the GPU host with the same running server:

```sh
python serve_probe.py --out /tmp/new-probe --lengths 512 2048 8192 16384 --repeats 3
```

Copy the result directory back, then regenerate figures locally:

```sh
python3 -m bench.plot_serve_probe results/<run-directory>
```

Dependencies: aiohttp for the probe; matplotlib for plotting. Exact server command, versions, model revision, and GPU metadata are recorded in metadata.json.
