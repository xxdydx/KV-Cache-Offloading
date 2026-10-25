import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from bench.test_engine_events import api, make_engine, params


def cached_engine(api, batches, budget=512, num_blocks=16):
    engine = make_engine(api, batches, budget=budget, num_blocks=num_blocks)
    runner = engine.model_runner
    runner.world_size = 1
    runner.block_size = 256
    shape = (2, 2, num_blocks, 256, 1, 4)
    runner.kv_cache = torch.arange(2 * 2 * num_blocks * 256 * 4).remainder(251).to(
        torch.bfloat16
    ).reshape(shape)
    return engine


def prefilled_engine(api, length=5, max_tokens=3, batches=None):
    if batches is None:
        batches = [[10]]
    engine = cached_engine(api, batches)
    request_id = engine.add_request([1] * length, params(api, max_tokens=max_tokens))
    engine.step_with_events()
    return engine, request_id


def test_detach_preserves_source_blocks_and_other_requests_can_progress(api):
    source, request_id = prefilled_engine(api, batches=[[10], [20]])
    original = source.scheduler.running[0]
    blocks = original.block_table.copy()
    snapshot = source.detach_request(request_id)
    assert snapshot.token_ids == [1] * 5 + [10]
    assert snapshot.num_cached_tokens == 5
    assert source._detached_requests[request_id] is original
    assert original.block_table == blocks
    assert source.scheduler.block_manager.used_block_ids == set(blocks)
    assert source.is_finished()

    other = source.add_request([2], params(api, max_tokens=1))
    result = source.step_with_events()
    assert result.token_events == [api.TokenEvent(other, 0, 20, True)]
    assert source.scheduler.block_manager.used_block_ids == set(blocks)

    with pytest.raises(ValueError, match="already detached"):
        source.detach_request(request_id)
    source.release_detached_request(request_id)
    assert not source._detached_requests
    assert not source.scheduler.block_manager.used_block_ids
    with pytest.raises(KeyError):
        source.release_detached_request(request_id)
    with pytest.raises(KeyError):
        source.reattach_request(request_id)


def test_reattach_continues_original_id_and_output_position(api):
    source, request_id = prefilled_engine(api, batches=[[10], [11], [12]])
    source.detach_request(request_id)
    assert source.reattach_request(request_id) == request_id
    assert not source._detached_requests
    with pytest.raises(KeyError):
        source.reattach_request(request_id)
    assert source.step_with_events().token_events == [api.TokenEvent(request_id, 1, 11, False)]
    final = source.step_with_events()
    assert final.token_events == [api.TokenEvent(request_id, 2, 12, True)]
    assert final.outputs == [(request_id, [10, 11, 12])]
    assert not source.scheduler.block_manager.used_block_ids


def test_failed_export_keeps_request_running_and_blocks_owned(api, monkeypatch):
    source, request_id = prefilled_engine(api)
    original = source.scheduler.running[0]
    blocks = set(original.block_table)

    def fail(*args):
        raise RuntimeError("export failed")

    monkeypatch.setattr(api, "export_kv", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        source.detach_request(request_id)
    assert list(source.scheduler.running) == [original]
    assert not source._detached_requests
    assert source.scheduler.block_manager.used_block_ids == blocks


def test_detach_rejects_queued_partial_prefill_finished_and_unknown_requests(api):
    source = cached_engine(api, [[90], [91], [10]], budget=2)
    request_id = source.add_request([1] * 5, params(api, max_tokens=1))
    with pytest.raises(ValueError, match="handoff boundary"):
        source.detach_request(request_id)
    source.step_with_events()
    blocks = set(source.scheduler.waiting[0].block_table)
    with pytest.raises(ValueError, match="handoff boundary"):
        source.detach_request(request_id)
    assert source.scheduler.block_manager.used_block_ids == blocks
    source.step_with_events()
    source.step_with_events()
    with pytest.raises(ValueError, match="handoff boundary"):
        source.detach_request(request_id)
    with pytest.raises(ValueError, match="handoff boundary"):
        source.detach_request(-1)
    assert not source._detached_requests
    assert not source.scheduler.block_manager.used_block_ids


@pytest.mark.parametrize("length", [255, 256, 257])
def test_resume_imports_exact_kv_and_continues_decode_without_replaying_first_token(api, length):
    source, source_id = prefilled_engine(api, length=length)
    snapshot = source.detach_request(source_id)
    destination = cached_engine(api, [[11], [12]])
    destination.model_runner.kv_cache.fill_(-1)
    destination_id = destination.resume_request(snapshot)
    seq = destination.scheduler.running[0]
    manager = destination.scheduler.block_manager
    assert destination_id == seq.seq_id
    assert destination_id != source_id
    assert not destination.scheduler.waiting
    assert not seq.is_prefill
    assert seq.num_cached_tokens == length
    assert seq.completion_token_ids == [10]
    assert len(seq.block_table) == (length + 255) // 256
    verifier = sys.modules["nanovllm.engine.kv_handoff"].verify_imported_kv
    assert verifier(destination.model_runner, seq, snapshot)
    if length >= 256:
        assert manager.blocks[seq.block_table[0]].hash != -1

    source.release_detached_request(source_id)
    assert not source.scheduler.block_manager.used_block_ids
    first = destination.step_with_events()
    assert first.num_tokens == -1
    assert first.token_events == [api.TokenEvent(destination_id, 1, 11, False)]
    final = destination.step_with_events()
    assert final.token_events == [api.TokenEvent(destination_id, 2, 12, True)]
    assert final.outputs == [(destination_id, [10, 11, 12])]
    assert all(not prefill for prefill, _ in destination.model_runner.calls)
    assert not manager.used_block_ids


@pytest.mark.parametrize("mismatch", ["dtype", "block_size", "shape", "finished"])
def test_resume_rejects_invalid_snapshot_without_admitting_or_allocating(api, mismatch):
    source, request_id = prefilled_engine(api)
    snapshot = source.detach_request(request_id)
    if mismatch == "dtype":
        snapshot = replace(snapshot, kv=snapshot.kv.to(torch.float16))
    elif mismatch == "block_size":
        snapshot = replace(snapshot, block_size=512)
    elif mismatch == "shape":
        snapshot = replace(snapshot, kv=snapshot.kv[:, :, :-1])
    else:
        snapshot = replace(snapshot, max_tokens=1)
    destination = cached_engine(api, [])
    with pytest.raises(ValueError):
        destination.resume_request(snapshot)
    assert destination.is_finished()
    assert not destination.scheduler.block_manager.used_block_ids
    assert request_id in source._detached_requests


def test_resume_out_of_blocks_preserves_existing_destination_and_source_can_reattach(api):
    source, source_id = prefilled_engine(api, length=257, batches=[[10], [11], [12]])
    snapshot = source.detach_request(source_id)
    destination = cached_engine(api, [[20]], num_blocks=1)
    existing_id = destination.add_request([2], params(api, max_tokens=2))
    destination.step_with_events()
    existing = destination.scheduler.running[0]
    blocks = set(existing.block_table)
    with pytest.raises(MemoryError):
        destination.resume_request(snapshot)
    assert list(destination.scheduler.running) == [existing]
    assert existing.seq_id == existing_id
    assert destination.scheduler.block_manager.used_block_ids == blocks
    assert source.reattach_request(source_id) == source_id
    assert source.step_with_events().token_events == [api.TokenEvent(source_id, 1, 11, False)]


def test_failed_import_releases_allocated_destination_blocks(api, monkeypatch):
    source, request_id = prefilled_engine(api, length=256)
    snapshot = source.detach_request(request_id)
    destination = cached_engine(api, [])
    manager = destination.scheduler.block_manager

    def fail(seq):
        raise RuntimeError("import failed")

    monkeypatch.setattr(manager, "hash_blocks", fail)
    with pytest.raises(RuntimeError, match="import failed"):
        destination.resume_request(snapshot)
    assert destination.is_finished()
    assert not manager.used_block_ids
    assert len(manager.free_block_ids) == 16
    assert request_id in source._detached_requests


@pytest.mark.parametrize("sync_fails", [False, True])
def test_cuda_import_waits_for_readiness_before_admission_and_rolls_back_on_failure(api, monkeypatch, sync_fails):
    source, request_id = prefilled_engine(api)
    snapshot = source.detach_request(request_id)
    destination = cached_engine(api, [])
    manager = destination.scheduler.block_manager
    device = torch.device("cuda:0")
    destination.model_runner.kv_cache = SimpleNamespace(device=device)
    imported = []
    synchronized = []

    def fake_import(runner, manager, snapshot):
        seq = api.Sequence(snapshot.token_ids[:snapshot.num_prompt_tokens], params(api, max_tokens=3))
        seq.append_token(10)
        manager.allocate_for_resume(seq, snapshot.num_cached_tokens)
        imported.append(seq)
        return seq

    def synchronize(actual_device):
        assert actual_device == device
        assert not destination.scheduler.running
        synchronized.append(actual_device)
        if sync_fails:
            raise RuntimeError("copy failed")

    monkeypatch.setattr(api, "import_kv", fake_import)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    if sync_fails:
        with pytest.raises(RuntimeError, match="copy failed"):
            destination.resume_request(snapshot)
        assert not destination.scheduler.running
        assert not manager.used_block_ids
    else:
        assert destination.resume_request(snapshot) == imported[0].seq_id
        assert list(destination.scheduler.running) == imported
    assert synchronized == [device]


def test_exit_releases_detached_source_blocks(api):
    source, request_id = prefilled_engine(api)
    source.detach_request(request_id)
    manager = source.scheduler.block_manager
    source.ps = []
    calls = []
    source.model_runner.call = lambda method: calls.append(method)
    source.exit()
    assert not source._detached_requests
    assert not manager.used_block_ids
    assert calls == ["exit"]
