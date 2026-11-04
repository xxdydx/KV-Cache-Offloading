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

## Stock baseline and change history

The annotated tag **`nanovllm-stock`** points to commit `7e7fa46`, where this
directory exactly matches the complete upstream tree at the imported base
commit. That snapshot contains no local engine patches or `UPSTREAM.md`.
The directory's Git tree ID is `5ec2b4b8034b26f5cc9af73c13d520a2aa8774f0`,
identical to the upstream tree, including file contents, paths, and modes.

The existing additions are recorded after that baseline as separate commits:

| Commit | Engine change |
| --- | --- |
| `fa580a1` | Greedy and mixed-temperature sampling |
| `1ff54d3` | Independent worker initialization endpoints |
| `a76bfc2` | KV export/import, byte validation, and resumed-sequence allocation |

This sequence preserves the earlier project history rather than rewriting it.
The original vendoring commit already contained these additions; the stock
snapshot and following commits establish a clean comparison point without
changing the final engine implementation. Make future engine changes in
focused commits on top of this sequence, and keep the baseline tag fixed.

```bash
git log --reverse --oneline nanovllm-stock..HEAD -- nanovllm_upstream/
git diff --stat nanovllm-stock HEAD -- nanovllm_upstream/
git diff nanovllm-stock HEAD -- nanovllm_upstream/nanovllm/
```

The tag identifies a stock **engine**, not a stock version of the surrounding
project. The retained handoff benchmarks and greedy tests require subsequent
extension commits and are not expected to run against the stock engine.

## Local changes at import

- `nanovllm/sampling_params.py` and `nanovllm/layers/sampler.py`: greedy decoding
  support, including temperature-zero and mixed-temperature batches.
- `nanovllm/config.py` and `nanovllm/engine/model_runner.py`: configurable
  distributed initialization so independent workers can coexist.
- `nanovllm/engine/block_manager.py`: allocation support for resumed sequences.
- `nanovllm/engine/kv_handoff.py`: BF16 KV export/import and byte-exact
  validation for TP=1 workers.

This project extends nano-vllm's GPU prefix cache with CPU storage for inactive
agent contexts. The CPU cache tier is not yet implemented.

## Updating or benchmarking upstream

Use a separate worktree at `nanovllm-stock` for an unmodified engine baseline:

```bash
git worktree add --detach ../disaggregated-inference-stock nanovllm-stock
```

Run the stock engine's own entry points under that worktree's
`nanovllm_upstream/`, with an explicit environment and model configuration.
Record any additional compatibility patch explicitly. Review upstream updates
as normal source changes, preserve local modifications and attribution, and
update the recorded base commit when deliberately adopting a newer version.
