"""Validate and plot the Qwen3-8B prefill-budget ablation from raw results."""

import argparse
import csv
import hashlib
import json
import re
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.metrics import percentile


def read_records(folder):
    return [json.loads(line) for line in (folder/"requests.jsonl").read_text().splitlines()]


def token_times(r):
    return [e["t"] for e in r["events"]]


def gaps(r):
    t = token_times(r)
    if any(len(e["token_ids"]) != 1 for e in r["events"]):
        return None
    return [(b-a)*1000 for a,b in zip(t,t[1:])]


def ttft(r):
    return (r["events"][0]["t"]-r["sent"])*1000


def audit(folder, records):
    for r in records:
        t = token_times(r)
        assert r["sent"] <= t[0] <= t[-1] <= r["done"]
        assert all(a<=b for a,b in zip(t,t[1:]))
        assert sum(len(e["token_ids"]) for e in r["events"]) == r["requested_output"] == r["usage"]["completion_tokens"]
        assert len(r["prompt_token_ids"]) == r["usage"]["prompt_tokens"]
        assert not (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens",0)
    def counter(file,name):
        return sum(float(line.rsplit(" ",1)[1]) for line in (folder/file).read_text().splitlines()
                   if re.match(re.escape("vllm:"+name)+r"(?:\{| )",line))
    counts = dict(requests=len(records),output_tokens=sum(r["requested_output"] for r in records),
                  prompt_tokens=sum(len(r["prompt_token_ids"]) for r in records))
    for key,metric in [("requests","request_success_total"),("output_tokens","generation_tokens_total"),
                       ("prompt_tokens","prompt_tokens_total")]:
        delta = counter("metrics_after.txt",metric)-counter("metrics_before.txt",metric)
        assert delta == counts[key], (str(folder),key,delta,counts[key])
        counts[key+"_counter_delta"] = delta
    counts["preemptions"] = counter("metrics_after.txt","num_preemptions_total")-counter("metrics_before.txt","num_preemptions_total")
    counts["requests_with_bundled_events"] = [r["request_id"] for r in records if gaps(r) is None]
    return counts


def main(root, followup=None):
    experiment = json.loads((root/"experiment.json").read_text())
    completion = json.loads((root/"completion.json").read_text())
    assert completion["passed"] and not completion["compute_processes"].strip()
    args = experiment["args"]
    budgets = sorted(args["budgets"])
    locations = {b:(root,f"chunk{b}_seqs8") for b in budgets}
    if followup:
        extra_experiment = json.loads((followup/"experiment.json").read_text())
        extra_completion = json.loads((followup/"completion.json").read_text())
        assert extra_completion["passed"] and not extra_completion["compute_processes"].strip()
        ea = extra_experiment["args"]
        assert len(ea["budgets"]) == 1 and ea["concurrencies"] == [8]
        for field in ["model","lengths","repeats","paired_decoder_baselines","mixed_requests","mixed_concurrency","gpu"]:
            assert ea[field] == args[field], ("Follow-up configuration mismatch",field)
        budget = ea["budgets"][0]
        assert budget not in locations
        locations[budget] = (followup,"seqs8")
        budgets = sorted(locations)
    lengths = args["lengths"]
    repeats = args["repeats"]
    assert args["concurrencies"] == [8] and args["paired_decoder_baselines"]
    reference_inputs = None
    reference_trace = None
    reference_command = None
    rows, samples, audits = [], {}, {}
    for budget in budgets:
        run,label = locations[budget]
        folder = run/label
        mixed = run/(label+"_mixed")
        rs = read_records(folder)
        mr = read_records(mixed)
        md = json.loads((folder/"metadata.json").read_text())
        mm = json.loads((mixed/"metadata.json").read_text())
        assert md["checkpoint"] == mm["checkpoint"] == experiment["checkpoint"]
        assert md["model_config"] == mm["model_config"] == experiment["model_config"]
        command = json.loads((run/f"{label}_command.json").read_text())
        assert command[command.index("--max-num-batched-tokens")+1] == str(budget)
        command[command.index("--max-num-batched-tokens")+1] = "BUDGET"
        if reference_command is None:
            reference_command = command
        assert command == reference_command, "Other serving flags changed"
        index = {(r["scenario"],r["role"],r["prompt_length"],r["repeat"]):r for r in rs}
        assert len(index) == len(rs) == 2+len(lengths)*repeats*6
        inputs = {k:r["prompt_token_ids"] for k,r in index.items()}
        if reference_inputs is None:
            reference_inputs = inputs
        assert inputs == reference_inputs, "Controlled inputs differ across budgets"
        trace = json.loads((mixed/"trace.json").read_text())
        if reference_trace is None:
            reference_trace = trace
        assert trace == reference_trace, "Mixed inputs differ across budgets"
        assert len(trace) == args["mixed_requests"]
        assert sum(t["length"]==8192 for t in trace)*5 == len(trace)
        audit_label = f"budget{budget}"
        audits[audit_label] = audit(folder,rs)
        audits[audit_label+"_mixed"] = audit(mixed,mr)
        assert len(mr) == args["mixed_concurrency"]+len(trace)*repeats
        assert not audits[audit_label]["requests_with_bundled_events"], "Controlled token gaps unavailable"
        values = {}
        for n in lengths:
            maxg, control_maxg, extra, short, solo, injected_ttft = [], [], [], [], [], []
            for repeat in range(repeats):
                decoder = index[("inject_long","decoder",n,repeat)]
                injected = index[("inject_long","injected",n,repeat)]
                control = index[("decoder_control","decoder",n,repeat)]
                assert control["prompt_token_ids"] == decoder["prompt_token_ids"]
                assert sum(e["t"]<=injected["sent"] for e in decoder["events"]) == 16
                td,tc = token_times(decoder),token_times(control)
                maxg.append(max(gaps(decoder)))
                control_maxg.append(max(gaps(control)))
                extra.append(((td[-1]-td[15])-(tc[-1]-tc[15]))*1000)
                short.append(ttft(index[("queued_short","probe",n,repeat)]))
                solo.append(ttft(index[("solo","target",n,repeat)]))
                injected_ttft.append(ttft(injected))
            values.update({f"max_gap_{n}_ms":maxg,f"extra_completion_{n}_ms":extra,
                           f"control_max_gap_{n}_ms":control_maxg,
                           f"queued_short_ttft_{n}_ms":short,f"solo_ttft_{n}_ms":solo,
                           f"injected_ttft_{n}_ms":injected_ttft})
        trials = json.loads((mixed/"trials.json").read_text())
        assert len(trials) == repeats
        for trial in trials:
            measured = [r for r in mr if r["scenario"]=="mixed" and r["repeat"]==trial["repeat"]]
            assert len(measured) == len(trace)
            assert sorted(r["prompt_token_ids"] for r in measured) == sorted(t["prompt"] for t in trace)
            start,end = trial["window_start_s"],trial["window_end_s"]
            assert all(start<=r["sent"]<=r["done"]<=end for r in measured)
            assert abs((end-start)-trial["elapsed_s"])<1e-9
            assert abs(sum(r["requested_output"] for r in measured)/(end-start)-trial["output_tokens_per_s"])<1e-9
            valid_gaps = [gaps(r) for r in measured]
            if all(g is not None for g in valid_gaps):
                pooled = [v for g in valid_gaps for v in g]
                assert abs(percentile(pooled,99)-trial["pooled_itl_p99_ms"])<1e-9
            else:
                assert trial["pooled_itl_p99_ms"] is None
            assert abs(percentile([ttft(r) for r in measured],50)-trial["ttft_p50_ms"])<1e-9
            assert abs(percentile([ttft(r) for r in measured],99)-trial["ttft_p99_ms"])<1e-9
            events = sorted([(r["sent"],1) for r in measured]+[(r["done"],-1) for r in measured])
            active = peak = 0
            for _,delta in events:
                active += delta
                peak = max(peak,active)
            assert active==0 and peak<=args["mixed_concurrency"]
        values["mixed_output_tokens_per_s"] = [t["output_tokens_per_s"] for t in trials]
        values["mixed_p99_itl_ms"] = [t["pooled_itl_p99_ms"] for t in trials if t["pooled_itl_p99_ms"] is not None]
        values["mixed_ttft_p50_ms"] = [t["ttft_p50_ms"] for t in trials]
        values["mixed_ttft_p99_ms"] = [t["ttft_p99_ms"] for t in trials]
        row = dict(prefill_budget=budget)
        for metric,v in values.items():
            if v:
                row.update({metric:stats.median(v),metric+"_min":min(v),metric+"_max":max(v)})
        samples[budget] = values
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (root/"summary.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    validation = dict(passed=True,checkpoint=experiment["checkpoint"],same_inputs=True,
                      only_budget_changed=True,injections_after_exactly_16_tokens=True,
                      total_requests=sum(a["requests"] for a in audits.values()),
                      total_output_tokens=sum(a["output_tokens"] for a in audits.values()),
                      server_counter_audits=audits,gpu_released=True,
                      memory_events=completion["memory"]["memory.events"])
    if followup:
        validation["followup"] = dict(path=str(followup),memory_events=extra_completion["memory"]["memory.events"])
    resources = [json.loads(line) for line in (root/"resource_samples.jsonl").read_text().splitlines()]
    if followup:
        resources += [json.loads(line) for line in (followup/"resource_samples.jsonl").read_text().splitlines()]
    loaded = [g for sample in resources for g in sample["gpu"]
              if int(g["index"])==args["gpu"] and float(g["utilization.gpu [%]"])>=50]
    validation["loaded_gpu_temperature_range_c"] = [min(float(g["temperature.gpu"]) for g in loaded),
                                                    max(float(g["temperature.gpu"]) for g in loaded)]
    validation["loaded_gpu_sm_clock_range_mhz"] = [min(float(g["clocks.current.sm [MHz]"]) for g in loaded),
                                                   max(float(g["clocks.current.sm [MHz]"]) for g in loaded)]
    (root/"validation.json").write_text(json.dumps(validation,indent=2))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(2,2,figsize=(12,8.8),layout="constrained")
    def line(ax,metric,label,color):
        vs = [samples[b][metric] for b in budgets]
        if any(not v for v in vs):
            return
        med = [stats.median(v) for v in vs]
        ax.errorbar(range(len(budgets)),med,
                    yerr=[[m-min(v) for m,v in zip(med,vs)],[max(v)-m for m,v in zip(med,vs)]],
                    marker="o",capsize=4,label=label,color=color)
    for n,color in [(2048,"#059669"),(8192,"#2563eb"),(16384,"#d45a26")]:
        if n in lengths:
            line(axes[0,0],f"max_gap_{n}_ms",f"{n:,}-token arrival",color)
    line(axes[0,0],"control_max_gap_8192_ms","Uninterrupted control","#777777")
    axes[0,0].set(title="Largest pause in an active decoder",ylabel="Maximum inter-token gap (ms)")
    line(axes[0,1],"solo_ttft_8192_ms","8K request alone","#2563eb")
    line(axes[0,1],"queued_short_ttft_8192_ms","Short request behind 8K","#d45a26")
    axes[0,1].set(title="First-token latency",ylabel="TTFT (ms)")
    line(axes[1,0],"mixed_output_tokens_per_s","Mixed workload","#059669")
    axes[1,0].set(title="Mixed burst throughput",ylabel="Output tokens / second")
    line(axes[1,1],"mixed_p99_itl_ms","Median of per-trial pooled p99","#8b5cf6")
    axes[1,1].axhline(50,color="#777777",linestyle="--",linewidth=1,label="50 ms token-gap target")
    axes[1,1].set(title="Mixed burst token-gap tail",ylabel="Pooled p99 inter-token gap (ms)")
    for ax in axes.flat:
        ax.set_xticks(range(len(budgets)),[f"{b:,}" for b in budgets])
        ax.set_xlabel("Prefill token budget")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False,fontsize=9)
    fig.suptitle("Qwen3-8B BF16: what smaller prefill chunks change",fontsize=17,fontweight="bold")
    fig.supxlabel("One RTX 6000 Ada · vLLM 0.11 · eager · median and min–max across 3 repetitions\n"
                  "Mixed: 20 requests, 80% 512-token / 20% 8K prompts, 128 output tokens, up to 8 in flight. Finite diagnostic; clocks unlocked.",fontsize=9)
    fig.savefig(root/"chunk_comparison.png",dpi=180)
    fig.savefig(root/"chunk_comparison.pdf")
    plt.close(fig)
    lines = ["# Qwen3-8B prefill chunk-budget ablation", "",
             f"{len(budgets)} configurations of the same pinned Qwen3-8B checkpoint, BF16 weights/KV, vLLM 0.11.0, eager, TP=1, max_num_seqs=8, prefix caching disabled, one RTX 6000 Ada. Only max_num_batched_tokens changes. The initial execution order was "+", ".join(map(str,args["budgets"]))+(f"; followed by {extra_experiment['args']['budgets'][0]} in a separate supervisor run" if followup else "")+". GPU clocks were recorded, not locked.", "",
             "Three repetitions per measurement; tables show medians. Min–max ranges are retained in summary.csv and the plot. All paired input token IDs are checked across configurations.", "",
             "## Controlled 8K arrival", "",
             "An active 512-token-prompt decoder produces 128 tokens. An 8K/32-output request arrives after its 16th token. Each case has an immediately preceding uninterrupted decoder control with the exact same input. Completion penalty compares token-16-to-token-128 duration against that fresh control.", "",
             "| Prefill budget | Decoder largest gap (ms) | Extra completion time (ms) | 8K solo TTFT (ms) | Short request behind 8K: TTFT (ms) |",
             "|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['prefill_budget']:,} | "+" | ".join(f"{row[m]:.1f}" for m in
                     ["max_gap_8192_ms","extra_completion_8192_ms","solo_ttft_8192_ms","queued_short_ttft_8192_ms"])+" |")
    lines += ["", "The queued-short test submits a 512-token/64-output request 20 ms after an 8K/128-output request. This is a separate scenario from the injection test. Its latency includes queueing and prefill.", "",
              "## Mixed finite bursts", "",
              "Each of three trials completes the same seeded 20-request list: 16 prompts of 512 tokens, four of 8,192 tokens, 128 output tokens each, and up to eight requests in flight. A client semaphore refills a slot when a request completes, so actual arrival times change with completion times. Time spent waiting for a client slot is excluded from request TTFT. Throughput includes the complete finite burst and its drain; it is not a sustained saturation estimate or a matched-throughput comparison.", "",
              "| Prefill budget | Output tokens/s | Pooled p99 token gap (ms) | Request TTFT p50 (ms) | Request TTFT p99 (ms) |",
              "|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['prefill_budget']:,} | "+" | ".join(f"{row[m]:.1f}" if m in row else "unavailable" for m in
                     ["mixed_output_tokens_per_s","mixed_p99_itl_ms","mixed_ttft_p50_ms","mixed_ttft_p99_ms"])+" |")
    lines += ["", "Mixed token-gap p99 pools individual gaps within each trial, then reports the median of the three trial p99s. Trials with any multi-token streaming event have no individual-token tail estimate. These small trials do not establish production p99 latency.", "",
              "## Scope and validation", "",
              f"Validated {validation['total_requests']} requests and {validation['total_output_tokens']:,} output tokens against server request/input/output counters. Requested lengths, monotonic timestamps, absence of reported prefix-cache hits, identical cross-budget inputs, exact injection points, and the mixed client concurrency bound are checked. See validation.json, raw requests, saved traces, metrics, server commands/logs and resource_samples.jsonl.", "",
              "Greedy output identities are not required to match across chunk/batch shapes: BF16 arithmetic can change generated token choices. Output counts and computation shapes are fixed. Client-observed timings include HTTP and CPU overhead; no GPU kernel profile was collected. Temperature/clocks may vary between configurations, so the broad budget trend and reported spread matter more than small differences.", "",
              "This measures a scheduling ablation on one GPU. No new disaggregated serving comparison is run. A winning result against two tuned replicas at matched throughput remains a separate requirement. Both experiment servers and all GPU compute processes exited at completion.", "",
              "## Reproduce", "", "```sh",
              "python -m bench.run_8b_interference --model /path/to/Qwen3-8B --out /tmp/chunk-sweep --budgets 1024 256 2048 512 --concurrencies 8 --paired-decoder-baselines --mixed-requests 20",
              f"python3 -m bench.plot_chunk_sweep {root}"+(f" --followup {followup}" if followup else ""), "```", ""]
    smallest,largest = rows[0],rows[-1]
    reduction = 100*(1-smallest["max_gap_8192_ms"]/largest["max_gap_8192_ms"])
    lines[2:2] = [f"Reducing the token budget from {largest['prefill_budget']} to {smallest['prefill_budget']} cut the 8K-arrival maximum decoder gap from {largest['max_gap_8192_ms']:.1f} to {smallest['max_gap_8192_ms']:.1f} ms ({reduction:.1f}%). Mixed-burst throughput changed from {largest['mixed_output_tokens_per_s']:.1f} to {smallest['mixed_output_tokens_per_s']:.1f} output tokens/s. Solo 8K TTFT changed from {largest['solo_ttft_8192_ms']:.1f} to {smallest['solo_ttft_8192_ms']:.1f} ms. Smaller chunks address individual pauses; compare the completion-penalty column to see the remaining total delay.", ""]
    if "mixed_p99_itl_ms" in smallest:
        penalty = 100*(1-smallest["mixed_output_tokens_per_s"]/largest["mixed_output_tokens_per_s"])
        index = lines.index("## Scope and validation")
        lines[index:index] = ["## Interpretation", "",
            f"The smallest tested budget achieved a mixed-burst pooled p99 token gap of {smallest['mixed_p99_itl_ms']:.1f} ms, with {penalty:.1f}% lower observed finite-burst throughput than budget {largest['prefill_budget']}. Its request TTFT p99 was {smallest['mixed_ttft_p99_ms']:.1f} ms versus {largest['mixed_ttft_p99_ms']:.1f} ms. Smoother streaming therefore comes with increased initial waiting. A pooled token-gap target alone does not imply that the full TTFT/latency SLO is met.", "",
            "Scheduling alone removes most of the individual streaming pauses in this diagnostic. This weakens the case for building disaggregation solely to address those pauses. Use the smallest tested budget as a low-tail baseline and retain a larger budget as another latency/throughput tradeoff. A disaggregated implementation should demonstrate better latency or goodput against tuned two-replica baselines at the same GPU budget and matched throughput before further transport optimizations are justified.", ""]
    (root/"report.md").write_text("\n".join(lines))
    sources = ["bench/run_8b_interference.py","bench/serve_probe.py","bench/mixed_probe.py",
               "bench/runtime_info.py","bench/metrics.py","bench/plot_chunk_sweep.py"]
    (root/"source_sha256.json").write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--followup",type=Path)
    args = p.parse_args()
    main(args.root,args.followup)
