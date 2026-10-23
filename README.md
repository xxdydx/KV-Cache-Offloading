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

