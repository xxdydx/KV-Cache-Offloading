"""Per-turn-index view of an agent-replay matrix: does the CPU tier's benefit grow with the horizon?

For each config and turn index, averages over agents and seeds: prompt length,
the share of the prompt served from a prefix cache, median time to first
token, and mean turn latency per policy, with each policy's TTFT change
relative to the baseline policy at the same turn.

Run: python3 -m bench.summarize_turns RESULTS_DIR --config horizon [--baseline gpu_only] [--out FILE]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from bench.summarize_agent_replay import NAME


def load_turns(directory: Path, config: str):
    """{policy: {turn index: [turn record, ...]}} over all seeds and agents."""
    turns = defaultdict(lambda: defaultdict(list))
    for run in sorted(directory.iterdir()):
        match = NAME.match(run.name)
        if not match or match["config"] != config or not (run / "result.json").is_file():
            continue
        for record in json.loads((run / "result.json").read_text())["turns"]:
            turns[match["policy"]][record["turn"]].append(record)
    return turns


def summarize_turns(turns, baseline="gpu_only"):
    rows = []
    for index in sorted({i for by_turn in turns.values() for i in by_turn}):
        row = {"turn": index}
        for policy, by_turn in turns.items():
            records = by_turn.get(index, [])
            if not records:
                continue
            row[policy] = {
                "n": len(records),
                "prompt_len": statistics.fmean(r["prompt_len"] for r in records),
                "reused": sum(r["num_cached_tokens"] for r in records) / sum(r["prompt_len"] for r in records),
                "ttft_p50_ms": 1000 * statistics.median(r["ttft_s"] for r in records),
                "turn_latency_ms": 1000 * statistics.fmean(r["turn_latency_s"] for r in records),
            }
        base = row.get(baseline)
        for policy in turns:
            if policy != baseline and policy in row and base:
                row[policy]["ttft_delta_pct"] = 100 * (row[policy]["ttft_p50_ms"] / base["ttft_p50_ms"] - 1)
        rows.append(row)
    return rows


def markdown(rows, policies, baseline="gpu_only"):
    header = ["Turn", "prompt tokens"] + [f"`{p}` TTFT p50 ms (Δ), reuse" for p in policies]
    lines = ["| " + " | ".join(header) + " |", "|" + " ---: |" * len(header)]
    for row in rows:
        cells = [str(row["turn"])]
        first = next((row[p] for p in policies if p in row), None)
        cells.append(f"{first['prompt_len']:.0f}" if first else "")
        for policy in policies:
            item = row.get(policy)
            if item is None:
                cells.append("")
                continue
            delta = f" ({item['ttft_delta_pct']:+.0f}%)" if "ttft_delta_pct" in item else ""
            cells.append(f"{item['ttft_p50_ms']:.0f}{delta}, {100 * item['reused']:.0f}%")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", default="gpu_only")
    parser.add_argument("--out", type=Path, help="Write the JSON rows here")
    args = parser.parse_args()
    turns = load_turns(args.results, args.config)
    rows = summarize_turns(turns, args.baseline)
    policies = sorted(turns, key=lambda p: (p != args.baseline, p))
    print(markdown(rows, policies, args.baseline))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
