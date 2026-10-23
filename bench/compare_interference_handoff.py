"""Compare matched-model interference diagnostics with separate KV phase costs.

Arguments: INTERFERENCE_RUN HANDOFF_RUN. This does not estimate a serving speedup.
"""

import argparse
import csv
import json
import re
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(run, handoff):
    completion = json.loads((run / "completion.json").read_text())
    assert completion["passed"] and not completion["compute_processes"].strip()
    tables = [records(run / name / "requests.jsonl") for name in ["seqs1", "seqs8"]]
    metadata = [json.loads((run / name / "metadata.json").read_text()) for name in ["seqs1", "seqs8"]]
    transfer_metadata = json.loads((handoff / "two_gpu_tcp/metadata.json").read_text())
    checkpoint = transfer_metadata["checkpoint"]
    assert all(m["checkpoint"] == checkpoint for m in metadata), "Checkpoint mismatch"
    assert all(m["model_config"] == transfer_metadata["config"] for m in metadata)
    commands = [json.loads((run / f"{name}_command.json").read_text()) for name in ["seqs1", "seqs8"]]
    normalized = []
    for command in commands:
        command[command.index("--max-num-seqs")+1] = "CONCURRENCY"
        normalized.append(command)
    assert normalized[0] == normalized[1], "Other serving flags changed"
    def key(r):
        return (r["scenario"], r["role"], r["prompt_length"], r["repeat"])
    indexed = [{key(r):r for r in table} for table in tables]
    assert indexed[0].keys() == indexed[1].keys()
    bundled = []
    output_differences = []
    for k, a in indexed[0].items():
        b = indexed[1][k]
        assert a["prompt_token_ids"] == b["prompt_token_ids"], f"Input mismatch: {k}"
        for r in [a,b]:
            ids = [i for e in r["events"] for i in e["token_ids"]]
            event_times = [e["t"] for e in r["events"]]
            assert r["sent"] <= event_times[0] <= event_times[-1] <= r["done"]
            assert all(x <= y for x,y in zip(event_times,event_times[1:]))
            assert len(ids) == r["requested_output"] == r["usage"]["completion_tokens"]
            assert len(r["prompt_token_ids"]) == r["usage"]["prompt_tokens"]
            assert not (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens",0)
            if any(len(e["token_ids"]) != 1 for e in r["events"]):
                bundled.append(k)
        if [i for e in a["events"] for i in e["token_ids"]] != [i for e in b["events"] for i in e["token_ids"]]:
            output_differences.append(k)
    assert not bundled, "Cannot calculate individual token gaps for bundled streams"
    lengths = metadata[0]["args"]["lengths"]
    repeats = metadata[0]["args"]["repeats"]
    for index in indexed:
        for n in lengths:
            for repeat in range(repeats):
                decoder = index[("inject_long","decoder",n,repeat)]
                injected = index[("inject_long","injected",n,repeat)]
                assert sum(e["t"] <= injected["sent"] for e in decoder["events"]) == 16
    expected_requests = 2 + len(lengths)*repeats*5
    assert all(len(t) == expected_requests for t in tables)
    audits = []
    for label, table in zip(["seqs1","seqs8"], tables):
        def counter(filename, name):
            return sum(float(line.rsplit(" ",1)[1]) for line in (run/label/filename).read_text().splitlines()
                       if re.match(re.escape(name)+r"(?:\{| )", line))
        audit = dict(configuration=label, requests=len(table),
                     output_tokens=sum(r["requested_output"] for r in table),
                     prompt_tokens=sum(len(r["prompt_token_ids"]) for r in table))
        for name, metric in [("requests","request_success_total"),("output_tokens","generation_tokens_total"),
                             ("prompt_tokens","prompt_tokens_total")]:
            delta = counter("metrics_after.txt","vllm:"+metric)-counter("metrics_before.txt","vllm:"+metric)
            audit[name+"_server_counter_delta"] = delta
            assert delta == audit[name], f"Server counter mismatch: {label}, {name}: {delta}"
        audits.append(audit)
    transfers = records(handoff / "two_gpu_tcp/requests.jsonl")
    assert all(r["exact_token_match"] and r["kv_exact_match"] for r in transfers)
    def select(index, n, repeat, scenario, role):
        return indexed[index][(scenario,role,n,repeat)]
    def times(r):
        return [e["t"] for e in r["events"]]
    def maxgap(r):
        t = times(r)
        return max((b-a)*1000 for a,b in zip(t,t[1:]))
    def remaining(r):
        t = times(r)
        return (t[-1]-t[15])*1000
    def ttft(r):
        return (times(r)[0]-r["sent"])*1000
    rows = []
    samples = {}
    phases = ["export_ms", "socket_send_ms", "import_ms"]
    for n in lengths:
        values = {
            "serial_max_gap_ms":[maxgap(select(0,n,r,"inject_long","decoder")) for r in range(repeats)],
            "concurrent_max_gap_ms":[maxgap(select(1,n,r,"inject_long","decoder")) for r in range(repeats)],
            "paired_remaining_decode_delta_ms":[remaining(select(1,n,r,"inject_long","decoder"))-remaining(select(0,n,r,"inject_long","decoder")) for r in range(repeats)],
            "serial_short_ttft_ms":[ttft(select(0,n,r,"queued_short","probe")) for r in range(repeats)],
            "concurrent_short_ttft_ms":[ttft(select(1,n,r,"queued_short","probe")) for r in range(repeats)],
        }
        matched = [r for r in transfers if r["prompt_tokens"] == n and r["handoff_after"] == 1]
        if matched:
            assert len(matched) == repeats
            values.update({p:[r[p] for r in matched] for p in phases})
        samples[n] = values
        row = dict(prompt_tokens=n, repetitions=repeats)
        for metric, v in values.items():
            row.update({metric:stats.median(v), metric+"_min":min(v), metric+"_max":max(v)})
        rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (run / "handoff_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    validation = dict(passed=True, same_checkpoint=checkpoint, same_inputs=True,
                      only_concurrency_flag_changed=True, requests_per_configuration=expected_requests,
                      all_output_counts_match=True, no_prefix_cache_hits=True, single_token_events_only=True,
                      injections_after_exactly_16_received_tokens=True,
                      outputs_differ_between_concurrency_modes=output_differences,
                      gpu_released=not completion["compute_processes"].strip(),
                      server_counters=audits,
                      cgroup_memory_events=completion["memory"]["memory.events"],
                      transfer_source=str(handoff), interference_source=str(run))
    (run / "validation.json").write_text(json.dumps(validation, indent=2))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,3,figsize=(14,5.6),layout="constrained")
    def line(ax, metric, label, color, subset=lengths):
        x = list(range(len(subset)))
        v = [samples[n][metric] for n in subset]
        m = [stats.median(a) for a in v]
        ax.errorbar(x,m,yerr=[[b-min(a) for a,b in zip(v,m)], [max(a)-b for a,b in zip(v,m)]],
                    label=label,color=color,marker="o",capsize=3)
        ax.set_xticks(x,[f"{n:,}" for n in subset])
    line(axes[0],"serial_max_gap_ms","Serialized","#2563eb")
    line(axes[0],"concurrent_max_gap_ms","Concurrent, chunked prefill","#d45a26")
    axes[0].set_title("Largest pause in the active decoder")
    line(axes[1],"paired_remaining_decode_delta_ms","Concurrent minus serialized","#8b5cf6")
    axes[1].set_title("Extra time to finish after token 16")
    available = [n for n in lengths if "export_ms" in samples[n]]
    for metric,label,color in zip(phases,["Pack + GPU → CPU","TCP send, loopback","CPU → GPU + import"],
                                  ["#2563eb","#d45a26","#059669"]):
        line(axes[2],metric,label,color,available)
    axes[2].set_title("Separate nano-vllm transfer phase costs")
    for ax in axes:
        ax.set_xlabel("Long prompt (tokens)")
        ax.set_ylabel("Milliseconds")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Qwen3-8B BF16: interference and the cost of moving KV",fontweight="bold",fontsize=16)
    fig.supxlabel("RTX 6000 Ada · median and min–max of 3 runs · different metrics in each panel\n"
                  "vLLM interference versus nano-vllm operation timings; these do not measure a disaggregated serving speedup.",fontsize=10)
    fig.savefig(run / "handoff_comparison.png",dpi=180)
    fig.savefig(run / "handoff_comparison.pdf")
    plt.close(fig)
    lines = ["# Qwen3-8B interference rerun and handoff comparison", "",
             "This rerun uses the exact pinned Qwen3-8B checkpoint from the handoff experiment, BF16 weights and BF16 KV, eager execution, and one RTX 6000 Ada. It repeats the earlier serialized-versus-concurrent request patterns with vLLM, at a 2,048-token batch budget and chunked prefill enabled in both configurations.", "",
             f"Checkpoint: `{checkpoint}`. vLLM {metadata[0]['versions']['vllm']}, torch {metadata[0]['versions']['torch']}. The earlier 9B experiment used a different vLLM version as well as a different model/quantization; cross-model numerical differences are not a controlled model-only comparison.", "",
             "The only launch argument changed between these two 8B runs is max_num_seqs (1 versus 8). At most two requests overlap; eight is capacity. Prefix caching is disabled. Inputs are identical across configurations. Medians below summarize three repetitions; min–max spreads are retained in handoff_comparison.csv and the plots.", "",
             "| Long prompt | Serialized max gap (ms) | Concurrent max gap (ms) | Paired extra decode completion time (ms) | Serialized short TTFT (ms) | Concurrent short TTFT (ms) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['prompt_tokens']:,} | " + " | ".join(f"{r[k]:.1f}" for k in
                     ["serial_max_gap_ms","concurrent_max_gap_ms","paired_remaining_decode_delta_ms",
                      "serial_short_ttft_ms","concurrent_short_ttft_ms"]) + " |")
    lines += ["", "The injection test starts a 512-token-prompt decoder with 128 output tokens and submits the long request after output token 16; the injected request produces 32 output tokens. Extra completion time is the paired difference in token-16-to-token-128 duration between concurrency modes. It includes all subsequent interference and API/runtime effects, not only the worst single pause. The queued-short test submits a 512-token/64-output probe 20 ms after a long-prompt/128-output request.", "",
              "## Compare with moving the 8B KV cache", "",
              "These are the previously measured nano-vllm two-GPU raw-byte loopback-TCP phase medians, for handoff after the first output token. They use the same checkpoint and BF16 KV. They are not new transfer measurements, isolated PCIe microbenchmarks, or an end-to-end exposed-transfer duration.", "",
              "| Prompt | Pack + GPU → CPU (ms) | TCP send (ms) | CPU → GPU + import (ms) |",
              "|---:|---:|---:|---:|"]
    for r in rows:
        if "export_ms" in r:
            lines.append(f"| {r['prompt_tokens']:,} | " + " | ".join(f"{r[p]:.1f}" for p in phases) + " |")
    lines += ["", "Send and receive overlap, phases are measured on different workers, and correctness timings include additional validation elsewhere. Do not add these medians and call the result measured end-to-end latency. In a disaggregated pipeline, the long request's transfer delay and the existing decoder's pause also affect different requests.", "",
              "## What the evidence supports", "",
              "Compare the measured decoder pauses and cumulative completion penalty with the transfer-phase table. Interference establishes a reason to isolate prefill from active decoding. Transfer remains a substantial cost, so these measurements alone cannot establish a winning break-even point or net serving benefit. A faster transport may help, but that needs measurement.", "",
              "The decisive next comparison is two colocated replicas versus two GPUs doing concurrent prefill/decode disaggregation, with the same trace and matched throughput. This test uses one serving GPU at a time and no disaggregated serving load. It does not satisfy that two-GPU baseline or establish production p99 latency.", "",
              "## Validation and limits", "",
              f"Both configurations completed {expected_requests} requests (including two warmups), with exact requested output counts, single-token stream events, no reported prefix-cache hits, identical paired input token IDs and the same serving arguments except concurrency. Output token sequences differed across concurrency modes for {len(output_differences)} request pairs; batched BF16 greedy inference is not asserted to be batch-invariant. See validation.json for those pairs.", "",
              "Token timestamps are client observations on loopback and include API/CPU scheduling overhead. Only three repetitions and one controlled overlap are measured per point. Clocks, temperature, utilization, power and container memory were sampled; GPU clocks were not locked. The 8 GiB CPU limit and two shared CPUs remain possible sources of runtime noise. The earlier handoff uses nano-vllm, whereas interference uses vLLM, so this is a model/cache-matched cost comparison, not an engine-matched performance test.", "",
              f"Experiment processes have exited. Final compute-process listing: `{completion['compute_processes'].strip() or 'empty'}`. Final cgroup memory events: `{completion['memory']['memory.events']}`. Raw requests, server commands/logs, scheduler metrics and resource samples are retained.", "",
              "## Reproduce", "", "```sh",
              "# Use the isolated vLLM environment on the GPU host:",
              "python -m bench.run_8b_interference --model /path/to/Qwen3-8B --out /tmp/8b-interference",
              f"python3 -m bench.compare_serve_probes {run/'seqs1'} {run/'seqs8'} {run/'serving_comparison'}",
              f"python3 -m bench.compare_interference_handoff {run} {handoff}", "```", ""]
    at8k = next((r for r in rows if r["prompt_tokens"] == 8192), None)
    if at8k:
        lines[2:2] = [
            f"At 8K, allowing overlap increased the active decoder's median largest token gap from {at8k['serial_max_gap_ms']:.1f} to {at8k['concurrent_max_gap_ms']:.1f} ms. Its paired remaining completion time increased by {at8k['paired_remaining_decode_delta_ms']:.1f} ms. In return, the queued short request's median first-token latency fell from {at8k['serial_short_ttft_ms']:.1f} to {at8k['concurrent_short_ttft_ms']:.1f} ms. This is the measured scheduling tradeoff; it supports investigating prefill/decode isolation, while a disaggregated serving win remains unmeasured.", ""]
    (run / "report.md").write_text("\n".join(lines))
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("handoff",type=Path)
    args = parser.parse_args()
    main(args.run,args.handoff)
