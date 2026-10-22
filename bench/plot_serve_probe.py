"""Regenerate figures from serve_probe.py results: python -m bench.plot_serve_probe RESULTS_DIR."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def render(folder):
    rows = list(csv.DictReader((folder / "summary.csv").open()))
    for row in rows:
        for key in ("prompt_length", "repeat", "output_tokens"):
            row[key] = int(row[key])
        for key in ("ttft_ms", "mean_tpot_ms", "p99_itl_ms", "max_itl_ms", "total_s"):
            row[key] = float(row[key])
    records = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
    metadata = json.loads((folder / "metadata.json").read_text())
    match = re.search(r"--max-num-seqs (\d+)", " ".join(metadata["server_processes"]))
    concurrency = int(match.group(1)) if match else None
    serialized = concurrency == 1
    multi = []
    for record in records:
        events = record["events"]
        if not all(len(e["token_ids"]) == 1 for e in events):
            multi.append(record["request_id"])
        assert sum(len(e["token_ids"]) for e in events) == record["requested_output"] == record["usage"]["completion_tokens"]
        assert record["sent"] <= events[0]["t"] <= events[-1]["t"] <= record["done"]
        assert all(a["t"] <= b["t"] for a,b in zip(events, events[1:]))
        assert not (record["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    def counter_sum(filename, counter):
        return sum(float(line.rsplit(" ", 1)[1]) for line in (folder / filename).read_text().splitlines()
                   if re.match(re.escape(counter) + r"(?:\{| )", line))
    success_delta = counter_sum("metrics_after.txt", "vllm:request_success_total") - counter_sum("metrics_before.txt", "vllm:request_success_total")
    generated_delta = counter_sum("metrics_after.txt", "vllm:generation_tokens_total") - counter_sum("metrics_before.txt", "vllm:generation_tokens_total")
    samples = json.loads((folder / "scheduler_samples.json").read_text())
    audit = dict(requests=len(records), output_tokens=sum(r["requested_output"] for r in records),
                 server_success_counter_delta=success_delta, server_generation_counter_delta=generated_delta,
                 max_sampled_running=max(v for sample in samples for k,v in sample.items() if k.startswith("vllm:num_requests_running")),
                 max_sampled_waiting=max(v for sample in samples for k,v in sample.items() if k.startswith("vllm:num_requests_waiting")),
                 requests_with_multitoken_events=multi, all_requested_token_counts_matched=True,
                 all_reported_prefix_cache_hits_zero=True)
    audit["server_counters_match_probe"] = success_delta == len(records) and generated_delta == audit["output_tokens"]
    (folder / "validation.json").write_text(json.dumps(audit, indent=2))
    lengths = sorted({r["prompt_length"] for r in rows if r["scenario"] == "solo"})
    repeats = len([r for r in rows if r["scenario"] == "solo" and r["prompt_length"] == lengths[0]])
    colors = ["#2563eb", "#db5a28", "#0f8b71"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})

    def values(scenario, role, length, metric):
        return [r[metric] for r in rows if r["scenario"] == scenario and r["role"] == role and r["prompt_length"] == length]

    def series(ax, scenario, role, metric, label, color):
        vals = [values(scenario, role, length, metric) for length in lengths]
        meds = [statistics.median(v) for v in vals]
        ax.errorbar(range(len(lengths)), meds,
                    yerr=[[m-min(v) for m,v in zip(meds, vals)], [max(v)-m for m,v in zip(meds, vals)]],
                    marker="o", capsize=4, linewidth=2, color=color, label=label)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    fig.suptitle(f"Qwen3.5-9B on RTX 6000 Ada: max_num_seqs={concurrency}", fontsize=17, fontweight="bold")
    ax = axes[0, 0]
    series(ax, "solo", "target", "ttft_ms", "Request alone", colors[0])
    ax.set(title="First-token latency grows with prompt length", ylabel="Time to first token (ms)")
    ax = axes[0, 1]
    series(ax, "solo", "target", "mean_tpot_ms", "Mean TPOT", colors[0])
    series(ax, "solo", "target", "p99_itl_ms", "Per-request p99 token gap", colors[1])
    ax.set(title="Decode latency after the first token", ylabel="Latency (ms)")
    ax.legend(frameon=False)
    ax = axes[1, 0]
    series(ax, "queued_short", "probe", "ttft_ms", "512-token probe queued behind long request", colors[1])
    base = values("solo", "target", 512, "ttft_ms")
    ax.axhline(statistics.median(base), color=colors[0], linestyle="--", label="512-token request alone")
    ax.set(title="Latency of a short request submitted after a long one", ylabel="Probe time to first token (ms)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    series(ax, "inject_long", "decoder", "p99_itl_ms", "Ongoing decoder: per-request p99 gap", colors[2])
    series(ax, "inject_long", "decoder", "max_itl_ms", "Ongoing decoder: maximum gap", colors[1])
    ax.axhline(statistics.median(values("solo", "target", 512, "p99_itl_ms")), color=colors[0], linestyle="--", label="Solo 512-token request: p99 gap")
    ax.set(title="Existing decoder after a long request arrives", ylabel="Decoder token gap (ms)")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks(range(len(lengths)), [f"{n:,}" for n in lengths])
        ax.set_xlabel("Long prompt length (tokens)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)
    footnote = "This measures queueing, not concurrent prefill/decode interference." if serialized else "Concurrent prefill/decode; single GPU; no KV transfer."
    fig.supxlabel(f"GPTQ INT4 weights · FP8 KV · eager · max_num_seqs={concurrency} · {repeats} repetitions · median with min–max bars\n"
                  f"Cold-prefix requests; loopback client timings. {footnote}", fontsize=9)
    fig.savefig(folder / "latency_summary.png", dpi=180)
    fig.savefig(folder / "latency_summary.pdf")
    plt.close(fig)

    target_length = 8192 if 8192 in lengths else lengths[-1]
    selected = [("solo", "512-token request alone", [r for r in records if r["scenario"] == "solo" and r["prompt_length"] == 512 and r["repeat"] == 0]),
                ("queued_short", f"Short request behind {target_length:,}-token prompt", [r for r in records if r["scenario"] == "queued_short" and r["prompt_length"] == target_length and r["repeat"] == 0]),
                ("inject_long", f"{target_length:,}-token prompt injected during decode", [r for r in records if r["scenario"] == "inject_long" and r["prompt_length"] == target_length and r["repeat"] == 0])]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), constrained_layout=True)
    for ax, (scenario, title, reqs) in zip(axes, selected):
        origin = min(r["sent"] for r in reqs)
        for y, record in enumerate(sorted(reqs, key=lambda r:r["sent"])):
            color = colors[y]
            first = record["events"][0]["t"]
            ax.barh(y, first-record["sent"], left=record["sent"]-origin, height=0.22, alpha=0.2, color=color)
            ax.scatter([e["t"]-origin for e in record["events"]], [y]*len(record["events"]), marker="|", s=90, color=color)
            ax.scatter([record["sent"]-origin], [y], marker="v", color=color, s=40)
        ordered = sorted(reqs, key=lambda r:r["sent"])
        ax.set_yticks(range(len(ordered)), [f"{r['role']} ({r['usage']['prompt_tokens']:,} input tokens)" for r in ordered])
        ax.set(title=title, xlabel="Seconds since first request was sent", ylim=(-0.6, len(reqs)-0.4))
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Measured streaming timelines — one representative repetition", fontsize=16, fontweight="bold")
    fig.legend(handles=[Line2D([], [], marker="v", color="gray", linestyle="none", label="Request sent"),
                        Line2D([], [], marker="|", color="gray", linestyle="none", label="Output token received"),
                        Line2D([], [], color="gray", linewidth=8, alpha=0.2, label="Waiting + prefill before first token")],
               loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(folder / "request_timelines.png", dpi=180)
    fig.savefig(folder / "request_timelines.pdf")
    plt.close(fig)

    lines = [f"# Serving experiment: max_num_seqs={concurrency}", "",
             f"RTX 6000 Ada (48 GB); InfiniCloud/Qwen3.5-9B-GPTQ-INT4-G128; FP8 KV cache; eager execution; max_num_seqs={concurrency}.", "",
             f"Each point has {repeats} repetitions. Table values are medians; figures show the observed min–max range. No prefix reuse (unique cache salt per request).", "",
             "| Prompt tokens | Solo TTFT (ms) | Solo mean TPOT (ms) | Solo p99 token gap (ms) | Queued short TTFT (ms) | Active decoder p99 gap (ms) |", "|---:|---:|---:|---:|---:|---:|"]
    for length in lengths:
        nums = [statistics.median(values(scenario, role, length, metric)) for scenario, role, metric in
                [("solo", "target", "ttft_ms"), ("solo", "target", "mean_tpot_ms"), ("solo", "target", "p99_itl_ms"),
                 ("queued_short", "probe", "ttft_ms"), ("inject_long", "decoder", "p99_itl_ms")]]
        lines.append(f"| {length:,} | " + " | ".join(f"{v:.1f}" for v in nums) + " |")
    lines += ["", "## Interpretation and limits", "",
              "- The queued-short test submits a 512-token/64-output-token probe 20 ms after a long-prompt/128-output-token request. Its TTFT includes the blocker's prefill AND remaining decode; this is not an isolated prefill measurement.",
              "- The injection test starts a 512-token/128-output-token decoder, then submits the long prompt after the decoder emits 16 tokens. The injected request generates 32 tokens.",
              ("- max_num_seqs=1 serializes requests. This run cannot establish concurrent prefill/decode interference." if serialized else "- max_num_seqs=8 allows the long prefill to run while another request is decoding. Only two benchmark requests are submitted at once."),
              "- This run does not measure network transfer cost or disaggregation speedup. A separate chunking ablation is needed to establish chunking benefits.",
              "- p99 token gap is computed within each request, then summarized by its median across repeats. It is not p99 of request-mean TPOT, nor a statistically stable production tail estimate.",
              f"- Client timings include local HTTP/streaming overhead. Requests used greedy sampling, ignored EOS, and returned exactly the requested number of output tokens. {len(multi)} request(s) had multi-token streaming events; their per-token gap metrics are omitted (NaN). All plotted token-gap measurements require single-token events.",
              "- Existing prefix caching is enabled but a unique salt prevents reuse across our requests. Warmup used the largest prompt and a separate 128-token decode.",
              "- The existing service remains available to other clients, so exclusive load is not guaranteed. Scheduler samples and cumulative metrics are included for inspection.",
              f"- Validation: {audit['requests']} requests, {audit['output_tokens']:,} output tokens, all requested counts matched, and no reported prefix-cache hits. Server counters {'match the probe exactly' if audit['server_counters_match_probe'] else 'do not match the probe; see validation.json'}. Sampled maximum running/waiting: {audit['max_sampled_running']:g}/{audit['max_sampled_waiting']:g}.",
              "- Qwen3.5-9B with quantized weights and FP8 KV differs from the spec's BF16 Qwen3-8B; do not fit that model's KV-byte assumptions to these results.", "",
              "## Reproduce", "", "On the GPU host with the same running server:", "", "```sh",
              "python serve_probe.py --out /tmp/new-probe --lengths 512 2048 8192 16384 --repeats 3", "```", "",
              "Copy the result directory back, then regenerate figures locally:", "", "```sh", "python3 -m bench.plot_serve_probe results/<run-directory>", "```", "",
              "Dependencies: aiohttp for the probe; matplotlib for plotting. Exact server command, versions, model revision, and GPU metadata are recorded in metadata.json.", ""]
    (folder / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    render(parser.parse_args().folder)
