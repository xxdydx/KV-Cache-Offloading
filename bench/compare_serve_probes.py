"""Compare serialized and concurrent probe runs; arguments: SERIAL CONCURRENT OUT."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare(serial, concurrent, out):
    out.mkdir(parents=True, exist_ok=True)
    def read(folder):
        return list(csv.DictReader((folder / "summary.csv").open()))
    tables = [read(serial), read(concurrent)]
    lengths = sorted({int(r["prompt_length"]) for r in tables[0] if r["scenario"] == "solo"})
    def vals(table, length, scenario, role, metric):
        return [float(r[metric]) for r in table if int(r["prompt_length"]) == length and r["scenario"] == scenario and r["role"] == role]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    configs = [("max_num_seqs=1", "#2563eb"), ("max_num_seqs=8", "#d45a26")]
    for table, (label, color) in zip(tables, configs):
        for ax, scenario, role, metric in [(axes[0], "inject_long", "decoder", "max_itl_ms"), (axes[1], "queued_short", "probe", "ttft_ms")]:
            samples = [vals(table, n, scenario, role, metric) for n in lengths]
            med = [statistics.median(v) for v in samples]
            ax.errorbar(range(len(lengths)), med, yerr=[[m-min(v) for m,v in zip(med,samples)], [max(v)-m for m,v in zip(med,samples)]], marker="o", capsize=4, linewidth=2, label=label, color=color)
    axes[0].set(title="Longest pause in an ongoing decode", ylabel="Maximum inter-token gap (ms)")
    axes[1].set(title="First-token latency of a queued short request", ylabel="512-token probe TTFT (ms)")
    for ax in axes:
        ax.set_xticks(range(len(lengths)), [f"{n:,}" for n in lengths])
        ax.set_xlabel("Arriving / preceding long prompt (tokens)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
    fig.suptitle("Concurrency exposes prefill/decode interference", fontsize=17, fontweight="bold")
    fig.supxlabel("RTX 6000 Ada · Qwen3.5-9B GPTQ INT4 · FP8 KV · eager · 3 repetitions · median with min–max range\n"
                  "Same model and serving flags; concurrency and port changed. Separate runs; clocks were not locked.", fontsize=9)
    fig.savefig(out / "concurrency_comparison.png", dpi=180)
    fig.savefig(out / "concurrency_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), constrained_layout=True, sharex=True)
    for ax, folder, (label, color) in zip(axes, [serial, concurrent], configs):
        records = [json.loads(l) for l in (folder/"requests.jsonl").read_text().splitlines()]
        pair = [r for r in records if r["scenario"] == "inject_long" and r["prompt_length"] == 8192 and r["repeat"] == 0]
        decoder = next(r for r in pair if r["role"] == "decoder")
        injected = next(r for r in pair if r["role"] == "injected")
        origin = decoder["sent"]
        for y, record in enumerate([decoder, injected]):
            first = record["events"][0]["t"]
            ax.barh(y, first-record["sent"], left=record["sent"]-origin, height=.2, color=color, alpha=.18)
            ax.scatter([e["t"]-origin for e in record["events"]], [y]*len(record["events"]), marker="|", color=color, s=100)
        ax.axvline(injected["sent"]-origin, color="#555", linestyle="--", linewidth=1, label="8K request submitted")
        ax.set(title=label, yticks=[0,1], yticklabels=["Existing decoder", "New 8K request"], ylim=(-.5,1.5), xlabel="Seconds since decoder request was sent")
        ax.grid(axis="x", alpha=.2)
        ax.legend(loc="upper right", frameon=False)
    fig.suptitle("One 8K injection: each tick is a received token", fontsize=16, fontweight="bold")
    fig.supxlabel("Shading is waiting + prefill before first output; it does not isolate GPU prefill time. Representative repetition 0.", fontsize=9)
    fig.savefig(out / "interference_timeline.png", dpi=180)
    fig.savefig(out / "interference_timeline.pdf")
    plt.close(fig)
    lines = ["# Serialized versus concurrent serving", "",
             "The same Qwen3.5-9B GPTQ INT4 model ran on one RTX 6000 Ada with FP8 KV and eager execution. The independent experiment server changed max_num_seqs from 1 to 8 and port from 8001 to 8002; all other launch arguments were preserved.", "",
             "Each result is the median of three repetitions. Only two benchmark requests overlap; eight is the configured capacity, not measured batch size.", "",
             "| Long prompt | Decoder max gap, seqs=1 (ms) | Decoder max gap, seqs=8 (ms) | Short probe TTFT, seqs=1 (ms) | Short probe TTFT, seqs=8 (ms) |", "|---:|---:|---:|---:|---:|"]
    for n in lengths:
        numbers = [statistics.median(vals(t,n,"inject_long","decoder","max_itl_ms")) for t in tables]
        numbers += [statistics.median(vals(t,n,"queued_short","probe","ttft_ms")) for t in tables]
        lines.append(f"| {n:,} | " + " | ".join(f"{v:.1f}" for v in numbers) + " |")
    lines += ["", "The injection test adds the long request after the existing decoder emits 16 of its 128 tokens. The queued-short test submits a 512-token/64-output-token probe 20 ms after a long-prompt/128-output-token request.", "",
              "The concurrent server's startup log confirms enable_chunked_prefill=True. The pauses occur with chunking enabled; this experiment does not compare chunking on versus off.", "",
              "The maximum token gap exposes an individual freeze that p99 can miss when it occurs in fewer than 1% of gaps. This is a small controlled experiment, not a production p99 estimate.", "",
              "These runs measure single-GPU scheduling behavior. They do not establish disaggregation speedup or a network break-even point. The project spec uses BF16 Qwen3-8B; its KV-size formula should not be applied directly to this Qwen3.5 model.", "",
              "The original service was restored after the concurrent run; see restoration.json. Raw token IDs/timestamps, scheduler samples, metadata, and counter-validation files live in the two input run directories.", "",
              "## Regenerate", "", "```sh",
              f"python3 -m bench.compare_serve_probes {serial} {concurrent} {out}", "```", ""]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:13]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("concurrent", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    compare(args.serial, args.concurrent, args.out)
