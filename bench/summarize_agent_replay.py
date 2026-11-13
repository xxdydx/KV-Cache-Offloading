"""Summarize an agent-replay matrix: paired per-seed deltas vs a baseline (gpu_only) with 95% t-intervals.

Run directories are named <config>-<policy>-s<seed>-r<rep>. Configs listed in
--merge are pooled under the first name as extra repeats (e.g. a headline config
and its repeat). --attach SRC:DST adds SRC's runs (e.g. an ablation without its
own gpu_only control) to DST's groups with the same seed and repeat.

Run: python3 -m bench.summarize_agent_replay RESULTS_DIR [--merge ctx16k,ctx16k-rep]
     [--attach ablate16k:ctx16k] [--baseline POLICY] [--out FILE]
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# Two-sided 95% Student t critical values by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
NAME = re.compile(r"^(?P<config>.+)-(?P<policy>gpu_only(?:_[a-z0-9]+)*|cpu_[a-z0-9_]+|vllm_[a-z0-9_]+)-s(?P<seed>\d+)-r(?P<rep>\d+)$")


def load(directory: Path):
    """Each run's headline metrics plus prefill/decode time from its step trace."""
    runs = []
    for run in sorted(directory.iterdir()):
        match = NAME.match(run.name)
        if not match or not (run / "result.json").is_file():
            continue
        result = json.loads((run / "result.json").read_text())
        summary = result["summary"]
        prefill = sum(s["end_s"] - s["start_s"] for s in result["steps"] if s["phase"] == "prefill")
        decode = sum(s["end_s"] - s["start_s"] for s in result["steps"] if s["phase"] != "prefill")
        runs.append({**match.groupdict(), "wall_s": summary["wall_s"],
                     "traj_per_min": summary["completed_trajectories_per_min"],
                     "ttft_p50_ms": summary["turn_ttft_ms"]["p50"], "ttft_p95_ms": summary["turn_ttft_ms"]["p95"],
                     "reuse": summary["prompt_tokens_reused"] / summary["prompt_tokens_total"],
                     "work": (summary["prompt_tokens_total"],
                              sum(len(turn["token_times_s"]) for turn in result["turns"])),
                     "prefill_s": prefill, "decode_s": decode})
    return runs


def interval(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    return mean, T95[len(values) - 1] * statistics.stdev(values) / len(values) ** 0.5


def summarize(runs, merge=(), attach=None, baseline="gpu_only"):
    alias = {name: merge[0] for name in merge} if merge else {}
    attach = attach or {}
    groups = defaultdict(dict)  # config -> {(seed, rep): {policy: run}}
    for run in runs:
        if run["config"] in attach:
            config, key = attach[run["config"]], (run["seed"], run["rep"])
            groups[config].setdefault(key, {})[run["policy"]] = run
            continue
        config = alias.get(run["config"], run["config"])
        key = (run["seed"], run["rep"] if run["config"] == config else f"{run['config']}-{run['rep']}")
        groups[config].setdefault(key, {})[run["policy"]] = run
    table = []
    for config, pairs in groups.items():
        policies = sorted({p for runs_ in pairs.values() for p in runs_}, key=lambda p: (p != baseline, p))
        for policy in policies:
            own = [r[policy] for r in pairs.values() if policy in r]
            row = {"config": config, "policy": policy, "n": len(own)}
            for metric in ("wall_s", "traj_per_min", "ttft_p50_ms", "ttft_p95_ms", "reuse", "prefill_s", "decode_s"):
                row[metric] = statistics.fmean(r[metric] for r in own)
            paired = [r for r in pairs.values() if policy in r and baseline in r]
            if policy != baseline and paired:
                deltas = [100 * (r[policy]["traj_per_min"] / r[baseline]["traj_per_min"] - 1) for r in paired]
                ttft = [100 * (r[policy]["ttft_p50_ms"] / r[baseline]["ttft_p50_ms"] - 1) for r in paired]
                row["paired_n"] = len(deltas)
                row["traj_delta_pct"], row["traj_delta_ci95"] = interval(deltas)
                row["ttft_p50_delta_pct"], row["ttft_p50_delta_ci95"] = interval(ttft)
                row["wins"] = sum(d > 0 for d in deltas)
                # Pairs must do identical work (prompt and output tokens) to be comparable.
                row["work_mismatches"] = sum(r[policy]["work"] != r[baseline]["work"] for r in paired)
            table.append(row)
    return table


def markdown(table, baseline="gpu_only"):
    lines = [f"| Config | Policy | n | wall s | traj/min | Δ traj/min vs {baseline} (95% CI) | wins | "
             "TTFT p50 ms (Δ) | reuse | prefill s | decode s |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in table:
        delta = wins = ttft = ""
        if "traj_delta_pct" in row:
            ci = row["traj_delta_ci95"]
            delta = f"{row['traj_delta_pct']:+.1f}%" + (f" ± {ci:.1f}" if ci is not None else "")
            wins = f"{row['wins']}/{row['paired_n']}"
            if row["work_mismatches"]:
                wins += f" ({row['work_mismatches']} unequal work)"
            ttft = f" ({row['ttft_p50_delta_pct']:+.0f}%)"
        lines.append(f"| {row['config']} | `{row['policy']}` | {row['n']} | {row['wall_s']:.2f} | "
                     f"{row['traj_per_min']:.2f} | {delta} | {wins} | {row['ttft_p50_ms']:.0f}{ttft} | "
                     f"{100 * row['reuse']:.0f}% | {row['prefill_s']:.2f} | {row['decode_s']:.2f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--merge", default="", help="Comma-separated configs to pool under the first")
    parser.add_argument("--attach", action="append", default=[], metavar="SRC:DST",
                        help="Pair SRC's runs with DST's gpu_only runs of the same seed/repeat")
    parser.add_argument("--baseline", default="gpu_only", help="Policy the deltas are paired against")
    parser.add_argument("--out", type=Path, help="Write the JSON table here")
    args = parser.parse_args()
    attach = dict(item.split(":", 1) for item in args.attach)
    table = summarize(load(args.results), [m for m in args.merge.split(",") if m], attach, args.baseline)
    print(markdown(table, args.baseline))
    if args.out:
        args.out.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
