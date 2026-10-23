"""Render the local and two-GPU handoff evidence: python -m bench.plot_kv_handoff RUN_DIR.

RUN_DIR contains single_gpu/ and two_gpu_tcp/ result directories.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def render(folder):
    local=read_jsonl(folder/"single_gpu/requests.jsonl")
    tcp=read_jsonl(folder/"two_gpu_tcp/requests.jsonl")
    tcp_metadata=json.loads((folder/"two_gpu_tcp/metadata.json").read_text())
    raw=tcp_metadata["args"].get("wire_format","pickle")=="raw"
    migrations=[r for r in local if r["mode"]=="handoff"]
    lengths=sorted({r["prompt_tokens"] for r in migrations})
    positions=sorted({r["handoff_after"] for r in migrations})
    for records in (migrations,tcp):
        assert all(r["exact_token_match"] for r in records)
        assert all((r.get("transfer") or r)["kv_exact_match"] for r in records)
        assert all((r.get("transfer") or r)["resume_decode_only"] for r in records)
    for r in migrations:
        t=r["transfer"]
        assert set(t["source_blocks"]).isdisjoint(t["destination_blocks"])
    config=json.loads((folder/"single_gpu/metadata.json").read_text())["config"]
    per_token=2*config["num_hidden_layers"]*config["num_key_value_heads"]*config["head_dim"]*2
    for r in migrations:
        assert r["transfer"]["payload_bytes"]==r["transfer"]["cached_tokens"]*per_token
    for r in tcp:
        assert r["payload_bytes"]==r["cached_tokens"]*per_token
        assert r["wire_payload_bytes"]>=r["payload_bytes"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True,gridspec_kw={"width_ratios":[1,1.25]})
    ax=axes[0]
    ax.imshow([[1 for _ in positions] for _ in lengths],cmap="Greens",vmin=0,vmax=1,aspect="auto",alpha=.25)
    for y,length in enumerate(lengths):
        for x,after in enumerate(positions):
            a=[r for r in migrations if r["prompt_tokens"]==length and r["handoff_after"]==after]
            b=[r for r in tcp if r["prompt_tokens"]==length and r["handoff_after"]==after]
            ax.text(x,y,f"Local {len(a)}/{len(a)}\nTCP {len(b)}/{len(b)}",ha="center",va="center",fontsize=11)
    ax.set_xticks(range(len(positions)),[f"After output token {p}" for p in positions])
    ax.set_yticks(range(len(lengths)),[f"{n:,}" for n in lengths])
    ax.set(title="Exact greedy continuation: all cases passed",ylabel="Prompt length (tokens)")
    selected=[r for r in tcp if r["prompt_tokens"]==8192 and r["handoff_after"]==1]
    metrics=[("Pack +\nGPU → CPU","export_ms"),("Encode metadata" if raw else "Serialize","serialization_ms"),("TCP send\n(loopback)","socket_send_ms"),
             ("Decode metadata" if raw else "Deserialize","deserialization_ms"),("CPU → GPU\n+ import","import_ms")]
    samples=[[r[key] for r in selected] for _,key in metrics]
    med=[statistics.median(v) for v in samples]
    ax=axes[1]
    ax.bar(range(len(metrics)),med,color=["#2563eb","#8494b4","#d45a26","#8494b4","#0f8b71"])
    ax.errorbar(range(len(metrics)),med,yerr=[[m-min(v) for m,v in zip(med,samples)],[max(v)-m for m,v in zip(med,samples)]],fmt="none",color="#333",capsize=4)
    for i,m in enumerate(med):
        ax.annotate(f"{m:.1f}",(i,m),xytext=(0,6),textcoords="offset points",ha="center",fontsize=9)
    ax.set_xticks(range(len(metrics)),[name for name,_ in metrics],fontsize=9)
    ax.set(title="8K prompt: CPU-staged handoff phase durations",ylabel="Milliseconds (median, min–max)")
    ax.grid(axis="y",alpha=.2)
    ax.set_ylim(top=max(max(v) for v in samples)*1.25)
    fig.suptitle("Qwen3-8B BF16: verified KV handoff between two GPUs",fontsize=17,fontweight="bold")
    fig.supxlabel(f"{len(migrations)} local remaps + {len(tcp)} two-worker TCP handoffs · 64 output tokens per request · source cache overwritten\n"
                  "Two RTX 6000 Ada cards on ONE host. Phase durations are separate measurements; loopback is not an inter-machine link.",fontsize=9)
    fig.savefig(folder/"handoff_results.png",dpi=180)
    fig.savefig(folder/"handoff_results.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    labels=[f"{n:,}" for n in lengths]
    for ax,records,kind in [(axes[0],migrations,"Local remap"),(axes[1],tcp,"GPU 0 → TCP → GPU 1")]:
        for title,key,color in [("GPU → CPU","export_ms","#2563eb"),("CPU → GPU","import_ms","#0f8b71")]:
            samples=[]
            for n in lengths:
                selected=[r for r in records if r["prompt_tokens"]==n and r["handoff_after"]==1]
                samples.append([(r.get("transfer") or r)[key] for r in selected])
            med=[statistics.median(v) for v in samples]
            ax.errorbar(range(len(lengths)),med,yerr=[[m-min(v) for m,v in zip(med,samples)],[max(v)-m for m,v in zip(med,samples)]],label=title,color=color,marker="o",capsize=4)
        ax.set_xticks(range(len(lengths)),labels)
        ax.set(title=kind,xlabel="Prompt length (tokens)",ylabel="Copy duration (ms)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("Synchronous CPU-staged copy baseline",fontsize=16,fontweight="bold")
    fig.supxlabel("Three repetitions; median with min–max range. These copies use pageable host memory and Python block scatter; no overlap or compression.",fontsize=9)
    fig.savefig(folder/"copy_scaling.png",dpi=180)
    fig.savefig(folder/"copy_scaling.pdf")
    plt.close(fig)
    rows=[]
    for r in migrations:
        rows.append(dict(transport="single_gpu",prompt_tokens=r["prompt_tokens"],handoff_after=r["handoff_after"],repeat=r["repeat"],
                         payload_bytes=r["transfer"]["payload_bytes"],export_ms=r["transfer"]["export_ms"],import_ms=r["transfer"]["import_ms"],
                         exact_token_match=r["exact_token_match"]))
    for r in tcp:
        rows.append(dict(transport="two_gpu_tcp",prompt_tokens=r["prompt_tokens"],handoff_after=r["handoff_after"],repeat=r["repeat"],
                         payload_bytes=r["payload_bytes"],export_ms=r["export_ms"],import_ms=r["import_ms"],exact_token_match=r["exact_token_match"]))
    with (folder/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    selected=[r for r in tcp if r["prompt_tokens"]==8192 and r["handoff_after"]==1]
    lines=["# Qwen3-8B KV handoff correctness and CPU-staging measurements","",
           f"{len(migrations)}/{len(migrations)} single-GPU remaps and {len(tcp)}/{len(tcp)} two-GPU loopback-TCP handoffs produced exactly the same 64 output token IDs as uninterrupted greedy nano-vllm generation.","",
           "Two RTX 6000 Ada GPUs in one sandbox, BF16 weights and KV, TP=1 per worker, eager execution. Three repetitions at prompt lengths 255, 256, 257, 512, 2,048 and 8,192; handoff after output token 1 and 17. The 256-token block boundaries are deliberately exercised.","",
           "## What each GPU does","",
           "Both GPUs load their own copy of the same model weights once. GPU 0 processes the input prompt (prefill), storing the attention keys and values for the processed tokens. It generates the first output token, or continues to output token 17 in the later-handoff tests. GPU 1 receives that cache and the token history/position, then generates the rest of the 64-token completion. The model weights are not sent in each handoff.","",
           "```text","GPU 0 VRAM → sender CPU tensor → TCP loopback → receiver CPU tensor → GPU 1 VRAM","   prefill             bytes transferred between processes                  decode","```","",
           "CPU RAM is separate from GPU VRAM. Copies between them use the host/GPU connection (PCIe). TCP transports an ordered stream of bytes; it does not perform inference. In this sandbox both workers share one physical host, so TCP uses its local network stack and does not cross a physical network link.","",
           "The benchmark runs one request at a time to establish correctness. A serving system would overlap GPU 0 prefilling a new request with GPU 1 decoding earlier requests; this experiment does not yet measure that workload.","",
           "## What was verified","",
           "- Greedy outputs repeat exactly across uninterrupted baseline runs.",
           "- Local imports use disjoint physical blocks. The source blocks are overwritten with NaNs before continuation.",
           "- Two-worker tests use separate processes on physical GPU 0 and GPU 1, the same checkpoint, and a real TCP connection bound to 127.0.0.1.",
           "- Imported KV tensors match the exported payload bit for bit. Receivers resume with decode steps only, with no prefill recomputation.",
           f"- Payload size matches the architecture: {per_token:,} bytes per cached token. An 8K prompt exports {selected[0]['payload_bytes']:,} bytes ({selected[0]['payload_bytes']/1024**2:.0f} MiB), excluding framing/metadata. Handoff after token 17 also includes the cached first 16 output tokens.","",
           "## 8K prompt: CPU-staged reference path","",
           "Separate phase durations, median [min, max] across three repetitions:","","| Phase | Duration (ms) |","|---|---:|"]
    for (label,key),sample in zip(metrics,[[r[key] for r in selected] for _,key in metrics]):
        lines.append(f"| {label.replace(chr(10),' ')} | {statistics.median(sample):.2f} [{min(sample):.2f}, {max(sample):.2f}] |")
    lines += ["","## Scope and next step","",
              f"This establishes the correctness milestone and a CPU-staged {'raw-byte' if raw else 'pickled'} TCP reference path. Both GPUs are on one host: TCP uses loopback, so these numbers do not measure physical-network bandwidth or demonstrate cross-machine performance.","",
              ("The raw-byte path pickles only a small metadata header, then sends a memoryview of the CPU tensor directly. The receiver reads into a preallocated CPU tensor. This avoids full-cache pickle byte copies. Verification reads back one GPU block at a time, and the sender releases its CPU cache after sending." if raw else "The pickle reference serializes the full CPU tensor and can require several full-cache byte copies. It needs more CPU RAM than the raw path."),"",
              "Request exchange timings include the receiver's extra validation copy and its remaining decode. Per-token timestamps from the local test likewise include verification and source poisoning. Neither is a clean serving-performance benchmark. Phase times are measured on their respective worker clocks; send and receive can overlap and must not be added indiscriminately.","",
              "Export includes gathering the paged GPU blocks and creating CPU storage. Import includes allocating fresh blocks, copying to the GPU, scattering into its cache, and rebuilding block metadata. These are full operation timings, not isolated PCIe-bandwidth measurements.","",
              "There is no concurrent load, matched-throughput vLLM comparison, pipeline, pinned-memory transport, compression, or selective router in this run. Greedy equality against nano-vllm verifies handoff consistency; comparison to a separate reference engine remains future work.","",
              "Next: run the same two-worker protocol on separate machines and establish a bandwidth ceiling, then optimize the measured staging costs. The metadata format uses pickle between trusted local benchmark processes; this is not a public-facing service.","",
              "## Reproduce","","With the pinned checkpoint and nano-vllm dependencies on a two-GPU CUDA host:","","```sh",
              "CUDA_VISIBLE_DEVICES=0 python -m bench.kv_handoff --model /path/to/Qwen3-8B --out /tmp/local-run",
              "python -m bench.kv_tcp_handoff --model /path/to/Qwen3-8B --out /tmp/tcp-run",
              "# Place those outputs under RUN_DIR/single_gpu and RUN_DIR/two_gpu_tcp:",
              "python -m bench.plot_kv_handoff RUN_DIR","```","",
              "Metadata records versions and the checkpoint configuration. checkpoint.json pins the model revision. Raw input and output token IDs, copy timings, and validations are retained in the result directories.",""]
    lines += ["## Experiment arguments","",
              "- `--lengths`: input prompt lengths in tokens. 255/256/257 exercise either side of a cache-block boundary; 8,192 exercises a large transfer.",
              "- `--handoff-after 1 17`: move immediately after the first output token, or after 17 output tokens. The receiver generates the remaining output, not 64 additional tokens.",
              "- `--output-tokens 64`: total completion length, including tokens already generated by GPU 0.",
              "- `--repeats 3` and `--seed`: repeat each case with reproducible synthetic inputs.",
              "- `--prefill-chunk 2048`: cap prompt work per prefill step in the single-GPU harness. This is distinct from the 256-token KV allocation block size.",
              "- `--memory-fraction .6`: GPU memory budget. It does not change the sandbox's CPU RAM limit.",
              "- `--init-port`: private process-group setup port in the local harness. The two-worker benchmark uses independent setup ports and a separate local TCP socket for KV transfer.",""]
    (folder/"report.md").write_text("\n".join(lines))
    print("\n".join(lines[:6]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    render(parser.parse_args().folder)
