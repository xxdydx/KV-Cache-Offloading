# disaggregated-inference

Prefill/decode-disaggregated LLM serving across mismatched GPUs in separate machines, with the KV cache shipped over TCP.

**Question:** when does shipping KV state between machines beat keeping a request on one GPU, and how far can a purpose-built transfer path cut that cost?

- Model: Qwen3-8B (BF16, ~144 KB KV/token)
- Main pairing: RTX 5090 (prefill) → RTX 6000 Ada (decode)

## Layout

| Path | Contents |
| --- | --- |
| `microbench/` | Week-1 network and hardware microbenchmarks (iperf3, ping-pong, PCIe, matmul, power) |
| `results/` | Raw per-run CSVs/JSON with environment metadata (committed) |
| `bench/` | Scripts that regenerate every figure from `results/` |

_Status: week 1 — network + baselines._
