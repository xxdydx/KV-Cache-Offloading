# Upstream provenance

- Original project: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- Imported base commit: [`bb823b3e06983d71485a8e1f23715ebd87d98ef8`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8)
- Original license and copyright: retained in [LICENSE](LICENSE).
- Original project documentation: retained in [README.md](README.md).

## Repository layout

This directory is ordinary source tracked by the parent
`disaggregated-inference` repository. It is not a submodule or a separate Git
repository. The directory name is historical: its engine includes local
modifications and should not be treated as an unmodified upstream baseline.

The upstream GitHub repository and this repository have independent histories;
vendoring the source does not create GitHub's formal fork relationship.

## Local changes at import

- `nanovllm/sampling_params.py` and `nanovllm/layers/sampler.py`: greedy decoding
  support, including temperature-zero and mixed-temperature batches.
- `nanovllm/config.py` and `nanovllm/engine/model_runner.py`: configurable
  distributed initialization so independent workers can coexist.
- `nanovllm/engine/block_manager.py`: allocation support for resumed sequences.
- `nanovllm/engine/kv_handoff.py`: BF16 KV export/import and byte-exact
  validation for TP=1 workers.

The parent repository contains the handoff benchmarks, tests, cost model, and
performance experiments. These changes establish handoff correctness; they do
not yet constitute a concurrent disaggregated serving system.

## Updating or benchmarking upstream

Use a separate checkout of the pinned upstream commit for an unmodified
baseline. Record any additional compatibility patch explicitly. Review upstream
updates as normal source changes, preserve local modifications and attribution,
and update the recorded base commit when deliberately adopting a newer version.
