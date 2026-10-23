"""Run isolated, sequential vLLM interference probes and release the GPU.

Use the Python interpreter from the vLLM environment. No existing service is
stopped. The server listens only on loopback; both configurations use one GPU.
"""

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bench.runtime_info import memory_info


def gpu_snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,"
        "clocks.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv,nounits"
    ], text=True)


def stop_server(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    # Reap any remaining workers in our own process group only.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"
    ], text=True).strip()
    if active:
        raise RuntimeError(f"Expected idle GPUs before isolated benchmark: {active}")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1",args.port)) == 0:
            raise RuntimeError("Experiment port is already in use")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                       OMP_NUM_THREADS="2", TORCHINDUCTOR_COMPILE_THREADS="1",
                       VLLM_WORKER_MULTIPROC_METHOD="spawn",
                       VLLM_ATTENTION_BACKEND="FLASH_ATTN", HF_HUB_OFFLINE="1",
                       HF_HOME="/scratch/hf", VLLM_NO_USAGE_STATS="1")
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(), args=vars(args),
                    memory=memory_info(), gpu_before=gpu_snapshot(),
                    checkpoint=str(Path(args.model).resolve()),
                    model_config=json.loads(Path(args.model, "config.json").read_text()),
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   [Path(__file__), Path(__file__).with_name("serve_probe.py")]})
    (out / "experiment.json").write_text(json.dumps(metadata, indent=2))
    finished = threading.Event()
    origin = time.perf_counter()
    current_label = "setup"

    def monitor():
        with (out / "resource_samples.jsonl").open("w") as handle:
            while not finished.is_set():
                sample = dict(t=time.perf_counter()-origin, configuration=current_label, memory=memory_info(),
                              gpu=list(csv.DictReader(gpu_snapshot().splitlines(), skipinitialspace=True)))
                handle.write(json.dumps(sample)+"\n")
                handle.flush()
                finished.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    passed = False
    try:
        configurations = [(budget, concurrency) for budget in args.budgets for concurrency in args.concurrencies]
        for budget, concurrency in configurations:
            label = f"chunk{budget}_seqs{concurrency}" if len(args.budgets)>1 else f"seqs{concurrency}"
            current_label = label
            command = [str(Path(sys.executable).with_name("vllm")), "serve", args.model,
                       "--served-model-name", "qwen3-8b", "--host", "127.0.0.1",
                       "--port", str(args.port), "--dtype", "bfloat16",
                       "--kv-cache-dtype", "auto", "--tensor-parallel-size", "1",
                       "--max-model-len", str(max(args.lengths)+256),
                       "--max-num-seqs", str(concurrency),
                       "--max-num-batched-tokens", str(budget), "--enable-chunked-prefill",
                       "--no-enable-prefix-caching", "--gpu-memory-utilization", "0.6",
                       "--enforce-eager", "--generation-config", "vllm", "--seed", "20261023",
                       "--disable-log-requests"]
            (out / f"{label}_command.json").write_text(json.dumps(command, indent=2))
            print("START", label, flush=True)
            with (out / f"{label}_server.log").open("w") as log:
                server = subprocess.Popen(command, env=environment, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic()+600
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"{label} server exited {server.returncode}; see log")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{label} server startup timed out")
                    time.sleep(2)
                print("HEALTHY", label, flush=True)
                probe_command = [sys.executable, "-m", "bench.serve_probe",
                                "--url", f"http://127.0.0.1:{args.port}", "--model", "qwen3-8b",
                                "--checkpoint", args.model,
                                "--description", f"Qwen3-8B BF16 weights and KV; eager; prefill budget {budget}",
                                "--out", str(out / label), "--lengths", *map(str,args.lengths),
                                "--repeats", str(args.repeats)]
                if args.paired_decoder_baselines:
                    probe_command.append("--paired-decoder-baselines")
                subprocess.run(probe_command, env=environment, check=True)
                if args.mixed_requests:
                    subprocess.run([sys.executable, "-m", "bench.mixed_probe",
                                    "--url", f"http://127.0.0.1:{args.port}", "--model", "qwen3-8b",
                                    "--checkpoint", args.model, "--out", str(out / (label+"_mixed")),
                                    "--requests", str(args.mixed_requests), "--concurrency", str(args.mixed_concurrency),
                                    "--repeats", str(args.repeats)], env=environment, check=True)
            finally:
                stop_server(server)
            print("STOPPED", label, flush=True)
        passed = True
    finally:
        finished.set()
        thread.join(timeout=10)
        final = dict(passed=passed, memory=memory_info(), gpu_after=gpu_snapshot(),
                     compute_processes=subprocess.check_output([
                         "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                         "--format=csv,noheader"], text=True))
        (out / "completion.json").write_text(json.dumps(final, indent=2))
        print("COMPLETE", json.dumps(final), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048, 8192, 16384])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budgets", nargs="+", type=int, default=[2048])
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1,8])
    parser.add_argument("--paired-decoder-baselines", action="store_true")
    parser.add_argument("--mixed-requests", type=int, default=0)
    parser.add_argument("--mixed-concurrency", type=int, default=8)
    args = parser.parse_args()
    if any(n<1 for n in args.budgets+args.concurrencies+args.lengths) or args.repeats<1:
        parser.error("budgets, concurrency, lengths and repeats must be positive")
    main(args)
