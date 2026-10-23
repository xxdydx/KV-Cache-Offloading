"""Two GPU workers on one host: CPU-staged KV handoff over loopback TCP.

python -m bench.kv_tcp_handoff --model /path/to/Qwen3-8B --out results/new-tcp-run
This validates process/device separation. Loopback timings are NOT a physical
network bandwidth measurement. No inference weights are sent between workers.
"""

import argparse
import importlib.metadata
import math
import json
import os
import pickle
import random
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.runtime_info import memory_info


def send_object(connection, value):
    start = time.perf_counter()
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    serialization_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    connection.sendall(struct.pack("!Q",len(payload)))
    connection.sendall(payload)
    return dict(serialization_ms=serialization_ms, socket_send_ms=(time.perf_counter()-start)*1000,
                wire_payload_bytes=len(payload), framing_bytes=8)


def recv_exact(connection, size):
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        n = connection.recv_into(view[offset:])
        if not n:
            raise EOFError("Peer closed during message")
        offset += n
    return buffer


def receive_object(connection):
    size = struct.unpack("!Q",recv_exact(connection,8))[0]
    if size > 8*1024**3:
        raise ValueError("Frame exceeds benchmark's 8 GiB limit")
    start = time.perf_counter()
    payload = recv_exact(connection,size)
    receive_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    value = pickle.loads(payload)  # Trusted benchmark processes, bound to loopback.
    return value, dict(socket_receive_ms=receive_ms,deserialization_ms=(time.perf_counter()-start)*1000)


def send_snapshot(connection, snapshot, wire_format):
    if wire_format == "pickle":
        return send_object(connection,dict(command="resume",snapshot=snapshot))
    import torch
    metadata = {k:getattr(snapshot,k) for k in (
        "token_ids","num_prompt_tokens","num_cached_tokens","block_size","temperature","max_tokens","ignore_eos")}
    header = dict(command="resume",snapshot_metadata=metadata,kv_shape=list(snapshot.kv.shape),
                  kv_dtype=str(snapshot.kv.dtype),payload_bytes=snapshot.payload_bytes)
    stats = send_object(connection,header)
    # A byte view shares the CPU tensor's storage: no giant pickle/bytes copy.
    view = memoryview(snapshot.kv.view(torch.uint8).numpy()).cast("B")
    start = time.perf_counter()
    connection.sendall(view)
    stats["socket_send_ms"] += (time.perf_counter()-start)*1000
    stats["wire_payload_bytes"] += snapshot.payload_bytes
    return stats


def read_snapshot_payload(connection, message, receive):
    if "snapshot" in message:
        return message["snapshot"], receive
    import torch
    dtype = {"torch.bfloat16":torch.bfloat16,"torch.float16":torch.float16}[message["kv_dtype"]]
    shape = message["kv_shape"]
    if math.prod(shape)*2 != message["payload_bytes"] or message["payload_bytes"] > 8*1024**3:
        raise ValueError("Invalid raw KV payload length")
    start = time.perf_counter()
    payload = torch.empty(shape,dtype=dtype,device="cpu")
    view = memoryview(payload.view(torch.uint8).numpy()).cast("B")
    receive["allocation_ms"] = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    offset = 0
    while offset < len(view):
        n = connection.recv_into(view[offset:])
        if not n:
            raise EOFError("Peer closed during raw KV payload")
        offset += n
    receive["socket_receive_ms"] += (time.perf_counter()-start)*1000
    return dict(message["snapshot_metadata"],kv=payload), receive


def make_engine(args, port):
    from bench.kv_handoff import LLMEngine, torch
    torch.set_num_threads(2)
    return LLMEngine(args.model, enforce_eager=True,max_num_seqs=1,
                     max_num_batched_tokens=2048,max_model_len=max(args.lengths)+args.output_tokens+256,
                     gpu_memory_utilization=.6,distributed_init_method=f"tcp://127.0.0.1:{port}")


def receiver(args):
    from bench.kv_handoff import run_request,timed,import_kv,verify_imported_kv
    from nanovllm.engine.kv_handoff import KVSnapshot
    out=Path(args.out)
    engine=make_engine(args,24432)
    warm=engine.tokenizer.encode("Explain why the sky appears blue.")
    run_request(engine,warm,args.output_tokens)
    with socket.socket() as server:
        server.bind(("127.0.0.1",0))
        server.listen(1)
        server.settimeout(300)
        (out/"receiver_ready.json").write_text(json.dumps({"port":server.getsockname()[1]}))
        with server.accept()[0] as connection:
            connection.settimeout(300)
            while True:
                message,receive=receive_object(connection)
                if message["command"] == "stop":
                    send_object(connection,{"stopped":True})
                    return
                snapshot,receive=read_snapshot_payload(connection,message,receive)
                if isinstance(snapshot,dict):
                    snapshot=KVSnapshot(**snapshot)
                manager=engine.scheduler.block_manager
                assert engine.is_finished() and not manager.used_block_ids
                manager.hash_to_block_id.clear()
                seq,import_ms=timed(lambda:import_kv(engine.model_runner,manager,snapshot))
                destination_blocks=seq.block_table.copy()
                verified,verify_ms=timed(lambda:verify_imported_kv(engine.model_runner,seq,snapshot))
                assert verified,"Cross-device KV mismatch"
                engine.scheduler.running.append(seq)
                start=time.perf_counter()
                times=[]
                while not engine.is_finished():
                    _, processed_tokens=engine.step()
                    assert processed_tokens < 0,"Receiver unexpectedly recomputed prefill"
                    times.append(time.perf_counter()-start)
                response=dict(token_ids=seq.completion_token_ids,import_ms=import_ms,
                              kv_exact_match=True,resume_decode_only=True,validation_readback_ms=verify_ms,
                              destination_blocks=destination_blocks,resume_decode_ms=(time.perf_counter()-start)*1000,
                              resumed_token_times_s=times,memory=memory_info(),**receive)
                assert len(response["token_ids"]) == args.output_tokens and not manager.used_block_ids
                send_object(connection,response)
                print(f"RESUMED {snapshot.num_cached_tokens} cached tokens",flush=True)
                del snapshot,message,response


def sender(args):
    from bench.kv_handoff import run_request,timed,export_kv,SamplingParams,torch
    out=Path(args.out)
    engine=make_engine(args,24431)
    rng=random.Random(args.seed)
    text="\n".join(f"Record {i}: the measured value is {rng.randrange(10000000)}. Explain the result clearly and check the arithmetic." for i in range(max(args.lengths)))
    pool=engine.tokenizer.encode(text,add_special_tokens=False,truncation=True,max_length=max(args.lengths))
    prompts={n:pool[:n] for n in args.lengths}
    (out/"prompts.json").write_text(json.dumps(prompts))
    run_request(engine,prompts[max(args.lengths)],args.output_tokens)
    baselines={}
    for length in args.lengths:
        for repeat in range(args.repeats):
            record=run_request(engine,prompts[length],args.output_tokens)
            if length in baselines:
                assert record["token_ids"] == baselines[length]
            baselines[length]=record["token_ids"]
            with (out/"baselines.jsonl").open("a") as f:
                f.write(json.dumps(dict(prompt_tokens=length,repeat=repeat,**record))+"\n")
    port=json.loads((out/"receiver_ready.json").read_text())["port"]
    jobs=[(n,after,repeat) for n in args.lengths for after in args.handoff_after for repeat in range(args.repeats)]
    rng.shuffle(jobs)
    with socket.create_connection(("127.0.0.1",port),timeout=300) as connection:
        connection.settimeout(300)
        for length,after,repeat in jobs:
            manager=engine.scheduler.block_manager
            assert engine.is_finished() and not manager.used_block_ids
            manager.hash_to_block_id.clear()
            engine.add_request(prompts[length],SamplingParams(temperature=0,max_tokens=args.output_tokens,ignore_eos=True))
            source=engine.scheduler.waiting[-1]
            while source.num_completion_tokens < after:
                engine.step()
            engine.scheduler.running.remove(source)
            snapshot,export_ms=timed(lambda:export_kv(engine.model_runner,source))
            cached_tokens=snapshot.num_cached_tokens
            payload_bytes=snapshot.payload_bytes
            indices=torch.tensor(source.block_table,dtype=torch.long,device=engine.model_runner.kv_cache.device)
            engine.model_runner.kv_cache.index_fill_(2,indices,float("nan"))
            torch.cuda.synchronize()
            start=time.perf_counter()
            wire=send_snapshot(connection,snapshot,args.wire_format)
            del snapshot
            response,_=receive_object(connection)
            roundtrip_ms=(time.perf_counter()-start)*1000
            matched=response["token_ids"] == baselines[length]
            manager.deallocate(source)
            record=dict(prompt_tokens=length,handoff_after=after,repeat=repeat,
                        cached_tokens=cached_tokens,payload_bytes=payload_bytes,export_ms=export_ms,
                        exchange_with_validation_and_decode_ms=roundtrip_ms,
                        source_poisoned=True,exact_token_match=matched,**wire,**response)
            with (out/"requests.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            if not matched:
                raise AssertionError(f"Cross-GPU continuation differs: L={length}, after={after}")
            print(f"TCP PASS L={length} after={after} repeat={repeat} cached={cached_tokens} "
                  f"bytes={payload_bytes} export={export_ms:.2f}ms import={response['import_ms']:.2f}ms "
                  f"send={wire['socket_send_ms']:.2f}ms",flush=True)
        send_object(connection,{"command":"stop"})
        receive_object(connection)
    (out/"validation.json").write_text(json.dumps(dict(passed=True,handoff_requests=len(jobs),
        exact_token_matches=len(jobs),baseline_requests=len(args.lengths)*args.repeats,
        output_tokens_per_request=args.output_tokens,source_gpu=0,destination_gpu=1,
        transport="loopback TCP, same host; not an inter-machine network measurement",
        same_checkpoint=True,source_blocks_poisoned=True,imported_kv_bitwise_equal=True,
        wire_format=args.wire_format,memory=memory_info()),indent=2))


def parent(args):
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    metadata=dict(started_utc=datetime.now(timezone.utc).isoformat(),args=vars(args),
                  checkpoint=str(Path(args.model).resolve()),memory=memory_info(),
                  config=json.loads(Path(args.model,"config.json").read_text()),
                  versions={p:importlib.metadata.version(p) for p in ["torch","triton","transformers","flash-attn"]},
                  gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,uuid,memory.total,driver_version","--format=csv"],text=True),
                  limitations=["Two separate GPU workers on one host, loopback TCP",
                               f"Wire format: {args.wire_format}; synchronous CPU staging, no pipeline",
                               "Roundtrip includes receiver's extra KV verification and remaining decode",
                               "No concurrent requests and no stock-vLLM comparison"])
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    config=metadata["config"]
    bytes_per_token=2*config["num_hidden_layers"]*config["num_key_value_heads"]*config["head_dim"]*2
    largest_payload=(max(args.lengths)+max(args.handoff_after)-1)*bytes_per_token
    # Conservative copy-count estimate. The raw path holds two CPU tensors;
    # pickle can additionally hold several serialized/deserialized byte copies.
    estimated_cpu_bytes=3*1024**3+largest_payload*(2 if args.wire_format=="raw" else 7)
    cap=metadata["memory"].get("memory.max")
    metadata["estimated_cpu_bytes"]=estimated_cpu_bytes
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    if isinstance(cap,int) and estimated_cpu_bytes > cap*.8:
        raise RuntimeError(f"Estimated CPU memory {estimated_cpu_bytes/1024**3:.1f} GiB exceeds safe budget "
                           f"for {cap/1024**3:.1f} GiB container; use raw wire format, shorter prompts or more RAM")
    common=[sys.executable,"-u","-m","bench.kv_tcp_handoff","--model",args.model,"--out",args.out,
            "--lengths",*map(str,args.lengths),"--handoff-after",*map(str,args.handoff_after),
            "--repeats",str(args.repeats),"--output-tokens",str(args.output_tokens),"--seed",str(args.seed),
            "--wire-format",args.wire_format]
    child=None
    try:
        with (out/"receiver.log").open("w") as log:
            child=subprocess.Popen(common+["--role","receiver"],env={**os.environ,"CUDA_VISIBLE_DEVICES":"1","OMP_NUM_THREADS":"2"},stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+240
        while not (out/"receiver_ready.json").exists():
            if child.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError("Receiver did not start; inspect receiver.log")
            time.sleep(1)
        print("Receiver GPU 1 ready; starting sender GPU 0",flush=True)
        with (out/"sender.log").open("w") as log:
            subprocess.run(common+["--role","sender"],env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"2"},stdout=log,stderr=subprocess.STDOUT,check=True)
        child.wait(timeout=60)
        if child.returncode:
            raise RuntimeError("Receiver failed; inspect receiver.log")
        print((out/"validation.json").read_text(),flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--lengths",nargs="+",type=int,default=[255,256,257,512,2048,8192])
    parser.add_argument("--handoff-after",nargs="+",type=int,default=[1,17])
    parser.add_argument("--output-tokens",type=int,default=64)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--seed",type=int,default=20261023)
    parser.add_argument("--role",choices=["parent","sender","receiver"],default="parent")
    parser.add_argument("--wire-format",choices=["raw","pickle"],default="raw")
    args=parser.parse_args()
    if any(n<1 for n in args.lengths) or any(not 0<n<args.output_tokens for n in args.handoff_after):
        parser.error("Invalid prompt length or handoff point")
    {"parent":parent,"sender":sender,"receiver":receiver}[args.role](args)
