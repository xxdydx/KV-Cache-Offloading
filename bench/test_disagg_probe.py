"""CPU validation of the minimal handoff protocol and real engine worker loops."""

import copy
import json
import socket
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from bench import disagg_probe as probe
from bench import disagg_protocol as wire
from bench.test_engine_events import api
from bench.test_engine_handoff import cached_engine


@pytest.fixture
def limits():
    return wire.Limits(layers=2, heads=1, head_dim=4, block_size=256,
                       vocab_size=100, max_prompt_tokens=257, output_tokens=4,
                       max_payload_bytes=2 * 2 * 257 * 1 * 4 * 2)


def snapshot(count):
    bits = torch.arange(2 * 2 * count * 4, dtype=torch.int16).reshape(2, 2, count, 1, 4)
    bits.flatten()[0] = -32768
    bits.flatten()[1] = 32767
    return SimpleNamespace(kv=bits.view(torch.bfloat16), token_ids=[1] * count + [10],
                           num_prompt_tokens=count, num_cached_tokens=count,
                           block_size=256, temperature=0, max_tokens=4, ignore_eos=True)


def offer(count):
    return dict(version=1, kind="OFFER", request_id="A", num_prompt_tokens=count, max_tokens=4)


@pytest.mark.parametrize("count", [255, 256, 257])
def test_raw_protocol_preserves_bits_and_continuation_state(limits, count):
    original = snapshot(count)
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(3)
        right.settimeout(3)
        sent = pool.submit(wire.send_snapshot, left, "A", original)
        restored = wire.receive_snapshot(right, offer(count), limits)
        sent.result(timeout=3)
    assert torch.equal(restored.kv.view(torch.uint8), original.kv.view(torch.uint8))
    assert restored.token_ids == original.token_ids
    assert restored.num_cached_tokens == count


@pytest.mark.parametrize("field,value", [
    ("kv_dtype", "torch.float16"), ("kv_shape", [2, 2, 999, 1, 4]),
    ("payload_bytes", 1), ("request_id", "B"), ("block_size", 512),
    ("token_ids", [-1] * 258), ("temperature", float("nan")),
    ("ignore_eos", False), ("num_cached_tokens", 256),
])
def test_bad_snapshot_rejected_before_tensor_allocation(limits, monkeypatch, field, value):
    header = dict(version=1, kind="TRANSFER", **wire.snapshot_header("A", snapshot(257)))
    header[field] = value
    left, right = socket.socketpair()
    with left, right:
        # Raw JSON allows testing an invalid NaN supplied by a peer.
        encoded = json.dumps(header).encode()
        left.sendall(struct.pack("!I", len(encoded)) + encoded)
        monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("Allocated invalid payload"))
        with pytest.raises(ValueError):
            wire.receive_snapshot(right, offer(257), limits)


def test_oversized_frame_and_wrong_version_are_rejected():
    invalid = json.dumps(dict(kind="STOP", version=999)).encode()
    for payload in (struct.pack("!I", wire.MAX_HEADER_BYTES + 1),
                    struct.pack("!I", len(invalid)) + invalid):
        left, right = socket.socketpair()
        with left, right:
            left.sendall(payload)
            left.shutdown(socket.SHUT_WR)
            with pytest.raises((ValueError, EOFError)):
                wire.receive_message(right)


def test_truncated_tensor_raises_eof(limits):
    left, right = socket.socketpair()
    with left, right:
        wire.send_message(left, "TRANSFER", **wire.snapshot_header("A", snapshot(257)))
        left.sendall(b"short")
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            wire.receive_snapshot(right, offer(257), limits)


def test_host_memory_budget_includes_both_snapshots(monkeypatch):
    monkeypatch.setattr(probe, "memory_info", lambda: {})
    with pytest.raises(MemoryError, match="Sender plus receiver"):
        probe.check_host_memory(100, 199)
    probe.check_host_memory(100, 200)
    monkeypatch.setattr(probe, "memory_info", lambda: {"memory.max": 1024, "memory.current": 900})
    with pytest.raises(MemoryError, match="cgroup"):
        probe.check_host_memory(100, 200)
    monkeypatch.setattr(probe, "memory_info", lambda: {
        "memory.max": 1024**3, "memory.current": 900 * 1024**2,
        "memory.inactive_file": 600 * 1024**2})
    probe.check_host_memory(100 * 1024**2, 200 * 1024**2)


def engines(api):
    source = cached_engine(api, [[10], [20]], budget=512)
    destination = cached_engine(api, [], budget=512)
    calls = []

    def advance(method, seqs, is_prefill):
        calls.append((threading.get_ident(), is_prefill))
        return [seq.last_token + 1 for seq in seqs]

    destination.model_runner.call = advance
    return source, destination, calls


def test_two_requests_stream_release_before_send_and_clean_up(api, limits, monkeypatch):
    source, destination, calls = engines(api)
    requests = [dict(request_id="A", prompt=[1] * 255, max_tokens=4),
                dict(request_id="B", prompt=[2] * 257, max_tokens=4)]
    events = []
    real_send = probe.send_snapshot
    sent_snapshots = []

    def send_after_release(connection, request_id, snapshot):
        assert not source.scheduler.block_manager.used_block_ids
        assert not source._detached_requests
        assert source.is_finished()
        sent_snapshots.append(snapshot.kv.clone())
        real_send(connection, request_id, snapshot)

    monkeypatch.setattr(probe, "send_snapshot", send_after_release)
    monkeypatch.setattr(probe, "memory_info", lambda: {})

    def emitter(role):
        import time
        return lambda kind, **kw: events.append(dict(kind=kind, role=role, t=time.perf_counter(), **kw))

    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=2) as pool:
        left.settimeout(3)
        right.settimeout(3)
        decoded = pool.submit(probe.decode_loop, destination, right, limits,
                              emitter("decode"), 1024**2, 3)
        produced = pool.submit(probe.source_loop, source, requests, left, limits,
                               emitter("prefill"), 1024**2)
        produced.result(timeout=5)
        decoded.result(timeout=5)
    assert len(sent_snapshots) == 2
    for request_id, expected in (("A", [10, 11, 12, 13]), ("B", [20, 21, 22, 23])):
        tokens = sorted((e for e in events if e["kind"] == "TOKEN" and e["request_id"] == request_id),
                        key=lambda e: e["output_index"])
        assert [e["token_id"] for e in tokens] == expected
        assert [e["output_index"] for e in tokens] == list(range(4))
        assert [e["finished"] for e in tokens] == [False, False, False, True]
    assert calls and all(not prefill for _, prefill in calls)
    assert len({thread for thread, _ in calls}) == 1  # Socket thread never executes inference.
    assert not source.scheduler.block_manager.used_block_ids
    assert not destination.scheduler.block_manager.used_block_ids


def test_decode_advances_while_socket_waits_for_next_offer(api, limits, monkeypatch):
    source, destination, _ = engines(api)
    monkeypatch.setattr(probe, "memory_info", lambda: {})
    finished = threading.Event()

    def emit(kind, **fields):
        if kind == "FINISHED":
            finished.set()

    source.add_request([1] * 255, api.SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
    source.step_with_events()
    original = source.detach_request(source.scheduler.running[0].seq_id)
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(3)
        right.settimeout(3)
        decoded = pool.submit(probe.decode_loop, destination, right, limits, emit, 1024**2, 3)
        wire.send_message(left, "OFFER", request_id="A", num_prompt_tokens=255, max_tokens=4)
        wire.receive_message(left, "PERMIT", "A")
        wire.send_snapshot(left, "A", original)
        wire.receive_message(left, "READY", "A")
        assert finished.wait(timeout=3), "Decoder blocked waiting for another transfer"
        wire.send_message(left, "STOP")
        decoded.result(timeout=3)


def test_destination_capacity_failure_aborts_before_payload(api, limits, monkeypatch):
    destination = cached_engine(api, [], num_blocks=1)
    monkeypatch.setattr(probe, "memory_info", lambda: {})
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(3)
        right.settimeout(3)
        decoded = pool.submit(probe.decode_loop, destination, right, limits, lambda *a, **kw: None,
                              1024**2, 3)
        wire.send_message(left, "OFFER", request_id="A", num_prompt_tokens=257, max_tokens=4)
        with pytest.raises(MemoryError, match="output budget"):
            decoded.result(timeout=3)
        assert left.recv(1) == b""  # No permission and no CPU snapshot allocation.
    assert not destination.scheduler.block_manager.used_block_ids


def test_full_lifetime_reservation_waits_for_existing_decode(api, limits, monkeypatch):
    source, _, _ = engines(api)
    destination = cached_engine(api, [], num_blocks=2)
    destination.model_runner.call = lambda method, seqs, prefill: [s.last_token + 1 for s in seqs]
    monkeypatch.setattr(probe, "memory_info", lambda: {})
    events = []
    requests = [dict(request_id=r, prompt=[1] * 257, max_tokens=4) for r in ("A", "B")]
    emit = lambda kind, **kw: events.append(dict(kind=kind, **kw))
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=2) as pool:
        left.settimeout(3)
        right.settimeout(3)
        decoded = pool.submit(probe.decode_loop, destination, right, limits, emit, 1024**2, 3)
        produced = pool.submit(probe.source_loop, source, requests, left, limits, emit, 1024**2)
        produced.result(timeout=5)
        decoded.result(timeout=5)
    finish_a = next(i for i, e in enumerate(events) if e["kind"] == "FINISHED" and e["request_id"] == "A")
    permit_b = next(i for i, e in enumerate(events) if e["kind"] == "PERMIT" and e["request_id"] == "B")
    assert finish_a < permit_b
    assert not destination.scheduler.block_manager.used_block_ids


def test_source_overlaps_next_offer_with_earlier_receipt(api, limits, monkeypatch):
    """Concurrency gate, host side: OFFER B must not wait for READY A."""
    source, destination, _ = engines(api)
    monkeypatch.setattr(probe, "memory_info", lambda: {})
    requests = [dict(request_id="A", prompt=[1] * 255, max_tokens=4),
                dict(request_id="B", prompt=[2] * 257, max_tokens=4)]
    events = []
    emit = lambda kind, **kw: events.append(dict(kind=kind, **kw))
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=2) as pool:
        left.settimeout(3)
        right.settimeout(3)
        decoded = pool.submit(probe.decode_loop, destination, right, limits, emit, 1024**2, 3)
        produced = pool.submit(probe.source_loop, source, requests, left, limits, emit, 1024**2)
        produced.result(timeout=5)
        decoded.result(timeout=5)
    order = [e["kind"] + ":" + e.get("request_id", "") for e in events]
    assert order.index("SENT:A") < order.index("OFFER:B") < order.index("READY_RECEIVED:A")
    assert order.index("OFFER:B") < order.index("FINISHED:A")
    # Receipts are drained before source_loop returns; STOP follows on the wire
    # (never emitted), and the destination exiting cleanly proves the ordering.
    assert order[-1] == "READY_RECEIVED:B"
    assert not source.scheduler.block_manager.used_block_ids
    assert not destination.scheduler.block_manager.used_block_ids


def test_gate_overlap_detects_prefill_during_other_request_decode():
    events = [
        dict(kind="ADMITTED", request_id="B", local_id=1, t=10.0),
        dict(kind="READY", request_id="A", local_id=5, t=10.0),
        dict(kind="FINISHED", request_id="A", t=12.0),
        dict(kind="STEP", phase="prefill", request_ids=[1], start=10.5, end=11.5),
        dict(kind="STEP", phase="decode", request_ids=[5], start=10.0, end=12.0),
    ]
    overlaps = probe.find_gate_overlap(events)
    assert overlaps == [dict(prefill_request="B", decode_request="A", overlap_s=1.0)]
    # Same-request prefill/decode overlap must not count: the gate is cross-request.
    events[2]["t"] = 9.5  # decode of A ends before B's prefill starts
    assert probe.find_gate_overlap(events) == []


def test_import_failure_closes_connection_without_ready(api, limits, monkeypatch):
    destination = cached_engine(api, [])
    monkeypatch.setattr(probe, "memory_info", lambda: {})

    def fail(snapshot):
        raise RuntimeError("import failed")

    destination.resume_request = fail
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(3)
        right.settimeout(3)
        decoded = pool.submit(probe.decode_loop, destination, right, limits,
                              lambda *a, **kw: None, 1024**2, 3)
        wire.send_message(left, "OFFER", request_id="A", num_prompt_tokens=255, max_tokens=4)
        wire.receive_message(left, "PERMIT", "A")
        wire.send_snapshot(left, "A", snapshot(255))
        with pytest.raises(EOFError):
            wire.receive_message(left, "READY", "A")
        with pytest.raises(RuntimeError, match="import failed"):
            decoded.result(timeout=3)
    assert not destination.scheduler.block_manager.used_block_ids


def test_replayed_offer_is_rejected(api, limits, monkeypatch):
    destination = cached_engine(api, [])
    destination.model_runner.call = lambda method, seqs, prefill: [s.last_token + 1 for s in seqs]
    monkeypatch.setattr(probe, "memory_info", lambda: {})
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=1) as pool:
        left.settimeout(3)
        right.settimeout(3)
        decoded = pool.submit(probe.decode_loop, destination, right, limits,
                              lambda *a, **kw: None, 1024**2, 3)
        wire.send_message(left, "OFFER", request_id="A", num_prompt_tokens=255, max_tokens=4)
        wire.receive_message(left, "PERMIT", "A")
        wire.send_snapshot(left, "A", snapshot(255))
        wire.receive_message(left, "READY", "A")
        wire.send_message(left, "OFFER", request_id="A", num_prompt_tokens=255, max_tokens=4)
        with pytest.raises(EOFError):
            wire.receive_message(left)
        with pytest.raises(ValueError, match="Duplicate"):
            decoded.result(timeout=3)


def test_preflight_rejects_payload_budget_before_workers_start(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(dict(max_position_embeddings=1024,
        num_hidden_layers=2, num_key_value_heads=1, head_dim=4)))
    args = SimpleNamespace(model=str(model), lengths=[255, 257], output_tokens=4, host_payload_bytes=100)
    monkeypatch.setattr(probe.mp, "get_context", lambda *a: pytest.fail("Started workers"))
    with pytest.raises(MemoryError, match="payload budget"):
        probe.run(args)


def test_summary_never_calls_host_intervals_gpu_concurrency():
    requests = [dict(request_id="A", max_tokens=2)]
    events = [dict(kind="START", t=1),
              dict(kind="TOKEN", request_id="A", output_index=0, token_id=10, finished=False, t=2),
              dict(kind="TOKEN", request_id="A", output_index=1, token_id=11, finished=True, t=4),
              dict(kind="STEP", phase="prefill", start=1, end=3),
              dict(kind="STEP", phase="decode", start=2, end=4),
              *[dict(kind="CLEAN", role=r, used_blocks=0) for r in ("prefill", "decode")]]
    summary = probe.summarize(events, requests, None)
    assert summary["passed"] and summary["host_step_overlap_s"] == 1
    assert summary["requests"][0]["max_itl_s"] == 2  # Includes the handoff gap.
    assert summary["gpu_concurrency_validated"] is False
    assert not probe.summarize(events, requests, "worker crashed")["passed"]
    broken = copy.deepcopy(events)
    broken[2]["output_index"] = 0
    assert probe.summarize(broken, requests, None)["failed_requests"] == 1
