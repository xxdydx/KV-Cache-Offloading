"""Minimal two-worker, same-host disaggregation probe (TP=1, BF16, eager).

Read source_loop and decode_loop first. The I/O thread receives CPU snapshots;
only each process's main thread calls its engine. Failures abort the trial.
"""

import argparse
import atexit
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import queue
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing.connection import wait
from pathlib import Path

from bench.disagg_protocol import (
    Limits, receive_message, receive_snapshot, send_message, send_snapshot, validate_offer,
)
from bench.runtime_info import memory_info


def check_host_memory(payload_bytes, host_budget):
    """Budget both live CPU payloads, plus headroom under an available cgroup limit."""
    needed = 2 * payload_bytes
    if needed > host_budget:
        raise MemoryError("Sender plus receiver snapshots exceed host payload budget")
    info = memory_info()
    limit, current = info.get("memory.max"), info.get("memory.current")
    if isinstance(limit, int) and isinstance(current, int):
        # Model loading can leave reclaimable file cache charged to the cgroup.
        # This is a preflight estimate, not a promise that allocation cannot fail.
        live_estimate = max(0, current - info.get("memory.inactive_file", 0))
        if live_estimate + needed + 256 * 1024**2 > limit:
            raise MemoryError("Insufficient cgroup CPU memory for handoff plus headroom")


def full_request_blocks(prompt_tokens, output_tokens, block_size):
    # The final output token is returned without being processed into KV.
    return math.ceil((prompt_tokens + output_tokens - 1) / block_size)


def measured_step(engine, emit, phase, request_ids):
    """Host intervals plus local CUDA duration; NVTX permits later GPU overlap validation."""
    start = time.perf_counter()
    gpu_ms = None
    cuda = engine.model_runner.kv_cache.device.type == "cuda"
    if cuda:
        import torch

        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(phase + ":" + ",".join(request_ids))
        begin.record()
    try:
        result = engine.step_with_events()
        if cuda:
            end.record()
            end.synchronize()
            gpu_ms = begin.elapsed_time(end)
    finally:
        if cuda:
            torch.cuda.nvtx.range_pop()
    emit("STEP", phase=phase, request_ids=request_ids, start=start,
         end=time.perf_counter(), gpu_ms=gpu_ms)
    return result


def source_loop(engine, requests, connection, limits, emit, host_budget):
    """One source request and one transfer at a time; destination decodes independently."""
    from nanovllm.sampling_params import SamplingParams

    for request in requests:
        request_id, prompt = request["request_id"], request["prompt"]
        emit("OFFER", request_id=request_id)
        send_message(connection, "OFFER", request_id=request_id,
                     num_prompt_tokens=len(prompt), max_tokens=limits.output_tokens)
        receive_message(connection, "PERMIT", request_id)
        # No snapshot is held while waiting for destination capacity.
        check_host_memory(limits.payload_bytes(len(prompt)), host_budget)
        manager = engine.scheduler.block_manager
        if math.ceil(len(prompt) / limits.block_size) > len(manager.free_block_ids):
            raise MemoryError("Source lacks blocks for prompt")
        local_id = engine.add_request(prompt, SamplingParams(
            temperature=0, max_tokens=limits.output_tokens, ignore_eos=True))
        emit("ADMITTED", request_id=request_id, local_id=local_id)
        while True:
            result = measured_step(engine, emit, "prefill", [request_id])
            if not result.token_events:
                continue
            if len(result.token_events) != 1:
                raise RuntimeError("Probe expects one source request")
            token = result.token_events[0]
            if token.seq_id != local_id or token.output_index != 0 or token.finished:
                raise RuntimeError("Expected an unfinished first output token")
            emit("TOKEN", request_id=request_id, output_index=0,
                 token_id=token.token_id, finished=False)
            break
        start = time.perf_counter()
        check_host_memory(limits.payload_bytes(len(prompt)), host_budget)
        snapshot = engine.detach_request(local_id)
        engine.release_detached_request(local_id)
        emit("SOURCE_RELEASED", request_id=request_id, start=start,
             end=time.perf_counter(), used_blocks=len(manager.used_block_ids))
        start = time.perf_counter()
        send_snapshot(connection, request_id, snapshot)
        del snapshot
        emit("SENT", request_id=request_id, start=start, end=time.perf_counter())
        receive_message(connection, "READY", request_id)
        emit("READY_RECEIVED", request_id=request_id)
    send_message(connection, "STOP")


def receive_loop(connection, limits, inbox, replies, timeout):
    """Socket owner: never touches an engine or GPU tensor."""
    seen = set()
    try:
        while True:
            offer = receive_message(connection)
            if offer["kind"] == "STOP":
                inbox.put(("stop", None), timeout=timeout)
                return
            validate_offer(offer, limits)
            request_id = offer["request_id"]
            if request_id in seen:
                raise ValueError("Duplicate handoff; replay is unsupported")
            seen.add(request_id)
            inbox.put(("offer", offer), timeout=timeout)
            if replies.get(timeout=timeout) != "permit":
                raise RuntimeError("Missing capacity permission")
            send_message(connection, "PERMIT", request_id=request_id)
            snapshot = receive_snapshot(connection, offer, limits)
            inbox.put(("snapshot", (request_id, snapshot)), timeout=timeout)
            del snapshot
            if replies.get(timeout=timeout) != "ready":
                raise RuntimeError("Missing import acceptance")
            send_message(connection, "READY", request_id=request_id)
    except Exception as error:
        inbox.put(("error", error), timeout=timeout)


def decode_loop(engine, connection, limits, emit, host_budget, timeout):
    inbox, replies = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
    reader = threading.Thread(target=receive_loop,
                              args=(connection, limits, inbox, replies, timeout), daemon=True)
    reader.start()
    active = {}  # Destination-local ID -> (global request ID, full-lifetime block budget).
    pending = None
    reserved = None
    stopped = False
    manager = engine.scheduler.block_manager
    try:
        while not stopped or active:
            try:
                # Idle workers sleep on the inbox; active workers keep stepping.
                kind, value = inbox.get(timeout=0 if active else timeout)
            except queue.Empty:
                if not active:
                    raise TimeoutError("Timed out waiting for source")
            else:
                if kind == "error":
                    raise value
                if kind == "stop":
                    stopped = True
                elif kind == "offer":
                    pending = value
                elif kind == "snapshot":
                    request_id, snapshot = value
                    if reserved is None or request_id != reserved[0]:
                        raise RuntimeError("Snapshot has no reservation")
                    start = time.perf_counter()
                    local_id = engine.resume_request(snapshot)
                    active[local_id] = reserved
                    reserved = None
                    del snapshot, value
                    emit("READY", request_id=request_id, local_id=local_id,
                         start=start, end=time.perf_counter())
                    replies.put("ready", timeout=timeout)
            if pending is not None:
                prompt, outputs = validate_offer(pending, limits)
                blocks = full_request_blocks(prompt, outputs, limits.block_size)
                if blocks > len(manager.blocks):
                    raise MemoryError("Destination pool cannot fit request's output budget")
                # Reserve eventual growth too: active decoders cannot consume this grant.
                if blocks + sum(item[1] for item in active.values()) <= len(manager.blocks):
                    check_host_memory(limits.payload_bytes(prompt), host_budget)
                    reserved = (pending["request_id"], blocks)
                    emit("PERMIT", request_id=reserved[0], reserved_blocks=blocks)
                    pending = None
                    replies.put("permit", timeout=timeout)
            if active:
                result = measured_step(engine, emit, "decode", [item[0] for item in active.values()])
                if result.num_tokens >= 0:
                    raise RuntimeError("Destination unexpectedly recomputed prefill")
                for token in result.token_events:
                    request_id = active[token.seq_id][0]
                    emit("TOKEN", request_id=request_id, output_index=token.output_index,
                         token_id=token.token_id, finished=token.finished)
                    if token.finished:
                        del active[token.seq_id]
                        emit("FINISHED", request_id=request_id)
        if manager.used_block_ids:
            raise RuntimeError("Destination leaked KV block ownership")
    finally:
        # Wake blocked recv calls, including on a failed import. No automatic replay.
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            replies.put_nowait("abort")
        except queue.Full:
            pass
        reader.join(timeout=min(timeout, 1))


def worker(role, args, channel, listener=None):
    """Spawn entry point: set GPU visibility before importing any CUDA package."""
    engine = None
    connection = None
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0 if role == "prefill" else 1])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MAX_JOBS"] = "2"

    def emit(kind, **fields):
        channel.send(dict(kind=kind, role=role, t=time.perf_counter(), **fields))

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nanovllm_upstream"))
        import torch

        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("This probe requires two CUDA GPUs; no CUDA device is available")
        from nanovllm.engine.llm_engine import LLMEngine

        port = args.init_ports[0 if role == "prefill" else 1]
        engine = LLMEngine(args.model, tensor_parallel_size=1, enforce_eager=True,
                           max_num_seqs=2, max_num_batched_tokens=args.prefill_chunk,
                           max_model_len=max(args.lengths) + args.output_tokens,
                           gpu_memory_utilization=args.memory_fraction,
                           distributed_init_method=f"tcp://127.0.0.1:{port}")
        atexit.unregister(engine.exit)
        cache = engine.model_runner.kv_cache
        if cache.dtype != torch.bfloat16:
            raise ValueError("Probe requires BF16 KV")
        limits = Limits(cache.shape[1], cache.shape[-2], cache.shape[-1],
                        engine.model_runner.block_size,
                        engine.model_runner.config.hf_config.vocab_size,
                        max(args.lengths), args.output_tokens, args.host_payload_bytes // 2)
        uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        if uuid is None:  # Older supported torch releases lack this property.
            uuid = subprocess.check_output(["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
                                           "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        if not uuid:
            raise RuntimeError("Cannot identify physical GPU")
        emit("BOOT", gpu_uuid=str(uuid),
             memory=memory_info(), blocks=cache.shape[2], limits=vars(limits))
        command = channel.recv()
        if command["kind"] != "START":
            raise ValueError("Expected START")
        if role == "prefill":
            connection = socket.create_connection(args.address, timeout=args.timeout)
            connection.settimeout(args.timeout)
            source_loop(engine, args.requests, connection, limits, emit, args.host_payload_bytes)
        else:
            listener.settimeout(args.timeout)
            connection, _ = listener.accept()
            connection.settimeout(args.timeout)
            decode_loop(engine, connection, limits, emit, args.host_payload_bytes, args.timeout)
        emit("CLEAN", used_blocks=len(engine.scheduler.block_manager.used_block_ids),
             memory=memory_info(), vram_allocated_bytes=torch.cuda.memory_allocated(),
             vram_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException:
        emit("FAILED", error=traceback.format_exc())
        raise
    finally:
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        if engine is not None:
            engine.exit()
    emit("DONE")


def summarize(events, requests, error):
    """Measure on one same-host perf_counter clock; never call host overlap GPU proof."""
    start = next((e["t"] for e in events if e["kind"] == "START"), None)
    rows = []
    for request in requests:
        request_id = request["request_id"]
        tokens = sorted((e for e in events if e["kind"] == "TOKEN"
                         and e["request_id"] == request_id), key=lambda e: e["output_index"])
        complete = bool(tokens) and (len(tokens) == request["max_tokens"]
                    and [e["output_index"] for e in tokens] == list(range(request["max_tokens"]))
                    and all(not e["finished"] for e in tokens[:-1])
                    and tokens[-1]["finished"])
        if complete and (start is None or any(b["t"] < a["t"] for a, b in zip(tokens, tokens[1:]))):
            complete = False
        gaps = [b["t"] - a["t"] for a, b in zip(tokens, tokens[1:])]
        rows.append(dict(request_id=request_id, status="completed" if complete else "failed",
                         token_ids=[e["token_id"] for e in tokens],
                         ttft_s=tokens[0]["t"] - start if tokens and start is not None else None,
                         completion_s=tokens[-1]["t"] - start if complete else None,
                         mean_itl_s=statistics.mean(gaps) if gaps else None,
                         max_itl_s=max(gaps) if gaps else None))
    prefill = [e for e in events if e["kind"] == "STEP" and e["phase"] == "prefill"]
    decode = [e for e in events if e["kind"] == "STEP" and e["phase"] == "decode"]
    overlap = sum(max(0, min(p["end"], d["end"]) - max(p["start"], d["start"]))
                  for p in prefill for d in decode)
    completed = [r for r in rows if r["status"] == "completed"]
    elapsed = max((r["completion_s"] for r in completed), default=0)
    clean = {e["role"] for e in events if e["kind"] == "CLEAN" and e["used_blocks"] == 0}
    passed = error is None and len(completed) == len(requests) and clean == {"prefill", "decode"}
    return dict(passed=passed, error=error, trial_aborted=error is not None,
                requests=rows, failed_requests=len(rows) - len(completed),
                mean_ttft_s=statistics.mean(r["ttft_s"] for r in completed) if completed else None,
                median_ttft_s=statistics.median(r["ttft_s"] for r in completed) if completed else None,
                mean_completion_s=statistics.mean(r["completion_s"] for r in completed) if completed else None,
                median_completion_s=statistics.median(r["completion_s"] for r in completed) if completed else None,
                output_tokens_per_s=len(completed) * requests[0]["max_tokens"] / elapsed if elapsed else None,
                host_step_overlap_s=overlap, host_overlap_observed=overlap > 0,
                gpu_concurrency_validated=False,
                limitations=["Two-request finite same-host diagnostic; throughput includes drain",
                             "All requests arrive together; this is not an open-loop serving trace",
                             "Host step intervals are overlap candidates; validate GPU kernels with NVTX/Nsight",
                             "No uninterrupted GPU baseline comparison in this probe",
                             "Per-step CUDA timing adds synchronization; not a competitive performance run"])


def run(args):
    config = json.loads((Path(args.model) / "config.json").read_text())
    if max(args.lengths) + args.output_tokens > config.get("max_position_embeddings", 0):
        raise ValueError("Requested context exceeds checkpoint limit")
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    payload = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim * max(args.lengths) * 2
    check_host_memory(payload, args.host_payload_bytes)
    rng = random.Random(args.seed)
    args.requests = [dict(request_id=f"request-{i}", prompt=[rng.randrange(config["vocab_size"])
                        for _ in range(length)], max_tokens=args.output_tokens)
                     for i, length in enumerate(args.lengths)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs.json").write_text(json.dumps(args.requests))
    (out / "manifest.json").write_text(json.dumps(dict(args=vars(args), config=config,
        versions={name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        memory=memory_info()), indent=2))
    events, processes, channels = [], [], []
    error = None
    ctx = mp.get_context("spawn")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    args.address = listener.getsockname()
    with (out / "events.jsonl").open("w") as log:
        def record(event):
            events.append(event)
            log.write(json.dumps(event) + "\n")
            log.flush()

        try:
            for role in ("decode", "prefill"):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=worker, args=(role, args, child,
                                      listener if role == "decode" else None), name=role)
                process.start()
                child.close()
                processes.append(process)
                channels.append(parent)
            listener.close()
            booted, done = {}, set()
            deadline = time.monotonic() + args.startup_timeout
            while len(done) < 2:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Worker startup/trial deadline exceeded")
                ready = wait([c for i, c in enumerate(channels) if i not in done], timeout=min(1, remaining))
                for channel in ready:
                    event = channel.recv()
                    record(event)
                    if event["kind"] == "FAILED":
                        raise RuntimeError(event["error"])
                    if event["kind"] == "BOOT":
                        booted[event["role"]] = event
                        if len(booted) == 2:
                            if booted["prefill"]["limits"] != booted["decode"]["limits"]:
                                raise ValueError("Workers disagree on KV layout")
                            if booted["prefill"]["gpu_uuid"] == booted["decode"]["gpu_uuid"]:
                                raise ValueError("Workers must use distinct physical GPUs")
                            arrival = time.perf_counter()
                            record(dict(kind="START", t=arrival))
                            for request in args.requests:
                                record(dict(kind="SUBMIT", request_id=request["request_id"],
                                            t=arrival))
                            for peer in channels:
                                peer.send(dict(kind="START"))
                            deadline = time.monotonic() + args.timeout
                    if event["kind"] == "DONE":
                        done.add(channels.index(channel))
                for i, process in enumerate(processes):
                    if process.exitcode is not None and i not in done and not channels[i].poll():
                        raise RuntimeError(f"{process.name} exited without DONE: {process.exitcode}")
            for process in processes:
                process.join(timeout=2)
                if process.is_alive() or process.exitcode != 0:
                    raise RuntimeError(f"{process.name} did not exit cleanly")
        except BaseException as failure:
            error = f"{type(failure).__name__}: {failure}"
            record(dict(kind="ABORT", t=time.perf_counter(), error=error))
        finally:
            listener.close()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            for channel in channels:
                channel.close()
    summary = summarize(events, args.requests, error)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    # Chrome/Perfetto trace: host intervals on one same-host monotonic clock.
    trace = [dict(name=e["phase"], cat="engine", ph="X", pid=e["role"], tid=0,
                  ts=e["start"] * 1e6, dur=(e["end"] - e["start"]) * 1e6,
                  args=dict(request_ids=e["request_ids"], gpu_ms=e["gpu_ms"]))
             for e in events if e["kind"] == "STEP"]
    (out / "timeline.json").write_text(json.dumps(dict(traceEvents=trace)))
    print(json.dumps(summary, indent=2))
    return summary["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--lengths", nargs=2, type=int, default=[512, 8192])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--host-payload-bytes", type=int, default=3 * 1024**3)
    parser.add_argument("--init-ports", nargs=2, type=int, default=[24531, 24532])
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if (min(args.lengths) < 1 or max(args.lengths) > 8192 or not 2 <= args.output_tokens <= 256
            or args.gpus[0] == args.gpus[1] or min(args.gpus) < 0
            or args.init_ports[0] == args.init_ports[1]
            or not all(1024 <= p <= 65535 for p in args.init_ports)
            or args.prefill_chunk < 1 or args.host_payload_bytes < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0
            or not 0 < args.memory_fraction < 1):
        parser.error("Invalid probe limits; use distinct GPUs/ports, prompts 1..8192, outputs 2..256")
    args.model = str(Path(args.model).resolve())
    raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
