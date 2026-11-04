"""Plan KV pressure on any machine; optionally run bounded real-GPU batch trials.

The plan estimates memory only. It never predicts latency or manufactures
performance results. Execution requires a real local checkpoint and CUDA GPU.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

from bench.baseline import positive_int
from bench.inventory import save_json


def plan_workloads(config, batches, contexts, output_tokens, block_size=256):
    if config.get("model_type") != "qwen3":
        raise ValueError("The current baseline supports Qwen3 only")
    kv_bytes_per_token = (2 * config["num_hidden_layers"] * config["num_key_value_heads"]
                          * config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]) * 2)
    cells = []
    for context in sorted(set(contexts)):
        for batch in sorted(set(batches)):
            if min(batch, context, output_tokens) < 1:
                raise ValueError("Batch, context, and output lengths must be positive")
            cached_length = context + output_tokens - 1
            cells.append({
                "batch_size": batch, "prompt_tokens": context, "output_tokens": output_tokens,
                "kv_bytes_per_token_per_request": kv_bytes_per_token,
                "prompt_kv_bytes": batch * context * kv_bytes_per_token,
                "peak_valid_kv_bytes": batch * cached_length * kv_bytes_per_token,
                "required_kv_blocks": batch * math.ceil(cached_length / block_size),
                "peak_block_kv_bytes": batch * math.ceil(cached_length / block_size) * block_size * kv_bytes_per_token,
                "within_model_context": context + output_tokens <= config["max_position_embeddings"],
            })
    return {"mode": "memory_plan", "kv_dtype": "bfloat16", "block_size": block_size,
            "model_config": config, "cells": cells,
            "limitations": ["Analytic KV storage estimates, not measured allocations or speed predictions.",
                             "Exclude weights, activations, workspace, graph pools, staging, and allocator overhead.",
                             "This is simultaneous fixed-batch pressure, not an open-loop online arrival benchmark.",
                             "Host offload capacity must use actual free cgroup RAM; full KV on CPU may not fit."]}


def run_cell(command, log_path, timeout_s):
    """Run one isolated GPU process, cleaning up its process group on interruption."""
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return {"exit_code": process.wait(timeout=timeout_s), "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "timed_out": True}
        finally:
            # Also stop any compiler children left after main-process exit.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Pinned checkpoint config.json; sufficient for memory planning")
    parser.add_argument("--out", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--batch-sizes", type=positive_int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--contexts", type=positive_int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--output-tokens", type=positive_int, default=128)
    parser.add_argument("--run", action="store_true", help="Execute real GPU trials; otherwise only plan memory")
    parser.add_argument("--model", type=Path, help="Full local checkpoint, required for --run")
    parser.add_argument("--timeout-seconds", type=positive_int, default=600, help="Per-cell limit, including load/compile/warmup")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--prefill-budget", type=positive_int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    config = json.loads(args.config.read_text())
    if args.run:
        if args.model is None or not (args.model / "config.json").is_file():
            parser.error("--run requires a local --model checkpoint containing config.json")
        if json.loads((args.model / "config.json").read_text()) != config:
            parser.error("Planning config differs from the execution checkpoint config")
    try:
        plan = plan_workloads(config, args.batch_sizes, args.contexts, args.output_tokens)
    except ValueError as error:
        parser.error(str(error))
    save_json(args.out / "plan.json", plan)
    for cell in plan["cells"]:
        print(f"B={cell['batch_size']:3d} context={cell['prompt_tokens']:5d} "
              f"prompt KV={cell['prompt_kv_bytes']/1024**3:6.2f} GiB "
              f"peak block KV={cell['peak_block_kv_bytes']/1024**3:6.2f} GiB "
              f"context supported={cell['within_model_context']}")
    if not args.run:
        print("Memory plan only: no inference run and no latency/throughput prediction.")
        return
    rows = []
    aborted = False
    for cell in plan["cells"]:
        name = f"b{cell['batch_size']}-ctx{cell['prompt_tokens']}"
        row = {**cell, "status": "not_run_after_error" if aborted else "unsupported_context"}
        if not aborted and cell["within_model_context"]:
            command = [sys.executable, "-m", "bench.baseline", "--model", str(args.model),
                       "--out", str(args.out / name), "--batch-size", str(cell["batch_size"]),
                       "--prompt-tokens", str(cell["prompt_tokens"]), "--output-tokens", str(args.output_tokens),
                       "--prefill-budget", str(args.prefill_budget), "--repeats", str(args.repeats),
                       "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
            if args.cuda_graphs:
                command.append("--cuda-graphs")
            log_path = args.out / f"{name}.log"
            row.update(run_cell(command, log_path, args.timeout_seconds), command=command)
            summary_path = args.out / name / "summary.json"
            capacity_path = args.out / name / "capacity.json"
            if capacity_path.exists():
                row["capacity"] = json.loads(capacity_path.read_text())
            log = log_path.read_text()
            if row["timed_out"]:
                row["status"] = "timeout"
            elif row["exit_code"] == 0 and summary_path.exists():
                row.update(status="completed", measured_summary=json.loads(summary_path.read_text()))
            elif "Insufficient KV blocks" in log:
                row["status"] = "insufficient_gpu_kv_capacity"
            elif "CUDA out of memory" in log or "torch.OutOfMemoryError" in log:
                row["status"] = "gpu_out_of_memory"
            else:
                row["status"] = "error"
            aborted = row["status"] in ("timeout", "error")
            print(f"{name}: {row['status']}", flush=True)
        rows.append(row)
        save_json(args.out / f"{name}-outcome.json", row)
    save_json(args.out / "outcomes.json", rows)
    if any(row["status"] not in ("completed", "unsupported_context") for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
