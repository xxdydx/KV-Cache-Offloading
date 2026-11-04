"""Automatic preservation/admission with real scheduler and block ownership."""

from collections import deque

import pytest
import torch

from bench.test_engine_events import api, params  # noqa: F401
from bench.test_engine_handoff import cached_engine
from bench.test_cpu_prefix_cache import completed_prefix, assert_pool_consistent


def automatic_cache(engine, blocks=2):
    pool = engine.model_runner.kv_cache
    size = pool[:, :, 0].numel() * pool.element_size()
    return engine.create_cpu_prefix_cache(max_bytes=blocks * size, automatic=True)


def write_scheduled_kv(engine):
    """Scripted model also overwrites newly processed KV, never cached prefixes."""
    original = engine.model_runner.call
    pool = engine.model_runner.kv_cache
    size = engine.scheduler.block_manager.block_size

    def run(method, seqs, is_prefill):
        tokens = original(method, seqs, is_prefill)
        for seq in seqs:
            start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            for logical in range(start // size, (end + size - 1) // size):
                lo, hi = max(start - logical * size, 0), min(end - logical * size, size)
                pool[:, :, seq.block_table[logical], lo:hi].fill_(seq[0])
        return tokens

    engine.model_runner.call = run


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_automatic_save_restore_and_next_turn_without_manual_calls(api, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    engine, manager, key = completed_prefix(api, [[90], [91], [12]], device)
    pool = engine.model_runner.kv_cache
    source = manager.hash_to_block_id[key]
    expected = pool[:, :, source].contiguous().cpu().view(torch.uint8).clone()
    cache = automatic_cache(engine, blocks=1)
    write_scheduled_kv(engine)
    for prompt in ([3] * 257, [4]):
        engine.add_request(prompt, params(api, max_tokens=1))
        engine.step_with_events()
    assert key not in manager.hash_to_block_id and key in cache
    engine.add_request([1] * 257 + [10, 11] + [2] * 10, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == 13
    target = manager.hash_to_block_id[key]
    assert target != source
    assert torch.equal(pool[:, :, target].contiguous().cpu().view(torch.uint8), expected)
    assert cache.stats == {"saved_blocks": 1, "restored_blocks": 1, "budget_skips": 1}
    assert not manager.used_block_ids
    assert_pool_consistent(manager)


@pytest.mark.parametrize("cpu_logical", [0, 1])
def test_all_gpu_hits_are_protected_before_any_cpu_restore(api, cpu_logical):
    engine = cached_engine(api, [[10], [90], [11]], budget=768, num_blocks=3)
    manager = engine.scheduler.block_manager
    original = [1] * 256 + [2] * 256 + [3]
    engine.add_request(original, params(api, max_tokens=1))
    engine.step_with_events()
    first = manager.compute_hash([1] * 256)
    second = manager.compute_hash([2] * 256, first)
    keys = [first, second]
    victim = manager.hash_to_block_id[keys[cpu_logical]]
    gpu_hit = manager.hash_to_block_id[keys[1 - cpu_logical]]
    cache = automatic_cache(engine, blocks=1)
    write_scheduled_kv(engine)
    # Adversarial but valid free-list ordering: after evicting the CPU hit,
    # the other required GPU prefix is the first recyclable slot.
    tail = next(i for i in manager.free_block_ids if i not in (victim, gpu_hit))
    manager.free_block_ids = deque([victim, gpu_hit, tail])
    engine.add_request([9], params(api, max_tokens=1))
    engine.step_with_events()
    assert keys[cpu_logical] in cache
    assert manager.free_block_ids[0] == gpu_hit
    expected = engine.model_runner.kv_cache[:, :, gpu_hit].contiguous().view(torch.uint8).clone()
    engine.add_request(original + [8] * 10, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == 11
    assert manager.hash_to_block_id[keys[1 - cpu_logical]] == gpu_hit
    assert torch.equal(engine.model_runner.kv_cache[:, :, gpu_hit].contiguous().view(torch.uint8), expected)
    assert cache.stats["restored_blocks"] == 1
    assert_pool_consistent(manager)


def test_cpu_hit_waits_for_capacity_without_mutating_pool_during_lookup(api):
    engine, manager, key = completed_prefix(api, [[90], [91], [12]])
    cache = automatic_cache(engine, blocks=1)
    write_scheduled_kv(engine)
    engine.add_request([3] * 257, params(api, max_tokens=2))
    engine.step_with_events()
    engine.add_request([1] * 257 + [10, 11] + [2] * 10, params(api, max_tokens=1))
    waiting = engine.scheduler.waiting[0]
    before = (manager.used_block_ids.copy(), manager.hash_to_block_id.copy(), cache.stats)
    assert key in cache and not manager.free_block_ids
    assert manager.can_allocate(waiting) == -1
    assert (manager.used_block_ids, manager.hash_to_block_id, cache.stats) == before
    assert not waiting.block_table
    assert engine.step_with_events().num_tokens == -1  # Existing decode completes.
    assert engine.step_with_events().num_tokens == 13
    assert cache.stats["restored_blocks"] == 1
    assert_pool_consistent(manager)


def test_full_cpu_budget_falls_back_to_recomputation(api):
    engine, manager, key = completed_prefix(api, [[90], [91], [12]])
    cache = engine.create_cpu_prefix_cache(max_bytes=1, automatic=True)
    for prompt in ([3] * 257, [4]):
        engine.add_request(prompt, params(api, max_tokens=1))
        engine.step_with_events()
    assert key not in cache and key not in manager.hash_to_block_id
    engine.add_request([1] * 257 + [10, 11] + [2] * 10, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == 269
    assert cache.stats["budget_skips"] >= 1
    assert cache.stats["saved_blocks"] == cache.stats["restored_blocks"] == 0
    assert_pool_consistent(manager)


def test_gpu_hits_are_shared_with_active_requests_without_cpu_copy(api):
    engine = cached_engine(api, [[10], [20]], budget=512, num_blocks=3)
    manager = engine.scheduler.block_manager
    cache = automatic_cache(engine)
    first_id = engine.add_request([1] * 257, params(api, max_tokens=3))
    engine.step_with_events()
    first = engine.scheduler.running[0]
    key = manager.compute_hash([1] * 256)
    block_id = manager.hash_to_block_id[key]
    engine.add_request([1] * 257 + [10] + [2] * 10, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == 12
    assert first.seq_id == first_id and manager.blocks[block_id].ref_count == 1
    assert cache.stats == {"saved_blocks": 0, "restored_blocks": 0, "budget_skips": 0}
    assert_pool_consistent(manager)


def test_close_detaches_automatic_hooks(api):
    engine, manager, key = completed_prefix(api, [[90], [91], [12]])
    cache = automatic_cache(engine)
    assert cache.automatic
    cache.close()
    assert not cache.automatic and manager._cpu_prefix_cache is None
    for prompt in ([3] * 257, [4]):
        engine.add_request(prompt, params(api, max_tokens=1))
        engine.step_with_events()
    engine.add_request([1] * 257 + [10, 11] + [2] * 10, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == 269
    assert cache.stats["saved_blocks"] == 0


def test_failed_preservation_does_not_pop_or_invalidate_victim(api, monkeypatch):
    engine, manager, key = completed_prefix(api)
    cache = automatic_cache(engine)
    victim = manager.hash_to_block_id[key]
    manager.free_block_ids.remove(victim)
    manager.free_block_ids.appendleft(victim)
    before = list(manager.free_block_ids)

    def fail(*args):
        raise RuntimeError("preservation copy failed")

    monkeypatch.setattr(cache, "save_block", fail)
    with pytest.raises(RuntimeError, match="preservation"):
        manager._allocate_block()
    assert list(manager.free_block_ids) == before
    assert manager.hash_to_block_id[key] == victim
    assert not manager.used_block_ids
    assert_pool_consistent(manager)


@pytest.mark.parametrize("policy", ["legacy", "fixed", "decode_aware"])
def test_prefill_batch_restore_failure_rolls_back_all_admissions(api, monkeypatch, policy):
    engine = cached_engine(api, [[10], [20], [90], [13, 14]], budget=1024, num_blocks=4,
                           prefill_policy=policy, prefill_chunk_size=1024)
    manager = engine.scheduler.block_manager
    engine.add_request([1] * 257, params(api, max_tokens=1))
    engine.step_with_events()
    cache = automatic_cache(engine, blocks=2)
    first = manager.compute_hash([1] * 256)
    cache.save_block(first)
    engine.add_request([3] * 257, params(api, max_tokens=1))
    engine.step_with_events()
    second = manager.compute_hash([3] * 256)
    cache.save_block(second)
    engine.add_request([9] * 769, params(api, max_tokens=1))
    engine.step_with_events()
    assert first not in manager.hash_to_block_id and second not in manager.hash_to_block_id
    ids = [engine.add_request([value] * 257, params(api, max_tokens=1)) for value in (1, 3)]
    calls = []
    original = cache._copy_entry

    def fail_second(entry, block_id):
        calls.append(block_id)
        if len(calls) == 2:
            engine.model_runner.kv_cache[:, :, block_id].fill_(0)
            raise RuntimeError("restore failed")
        original(entry, block_id)

    with monkeypatch.context() as patch:
        patch.setattr(cache, "_copy_entry", fail_second)
        with pytest.raises(RuntimeError, match="restore failed"):
            engine.step_with_events()
    assert [seq.seq_id for seq in engine.scheduler.waiting] == ids
    assert not engine.scheduler.running and not manager.used_block_ids
    assert all(not seq.block_table and seq.num_cached_tokens == seq.num_scheduled_tokens == 0
               for seq in engine.scheduler.waiting)
    assert len(engine.model_runner.calls) == 3
    assert_pool_consistent(manager)
    result = engine.step_with_events()
    assert result.num_tokens == 2  # One uncached tail token for each 257-token prompt.
    assert result.outputs == [(ids[0], [13]), (ids[1], [14])]
    assert_pool_consistent(manager)


@pytest.mark.parametrize("policy", ["legacy", "fixed", "decode_aware"])
def test_decode_growth_failure_restores_queue_and_tail_reservations(api, monkeypatch, policy):
    engine = cached_engine(api, [[80], [81], [10, 20], [11, 21]], budget=512, num_blocks=4,
                           prefill_policy=policy, prefill_chunk_size=512)
    manager = engine.scheduler.block_manager
    for value in (7, 8):
        engine.add_request([value] * 256, params(api, max_tokens=1))
        engine.step_with_events()
    cache = automatic_cache(engine, blocks=2)
    ids = [engine.add_request([value] * 256, params(api, max_tokens=2)) for value in (1, 2)]
    engine.step_with_events()
    before = manager.used_block_ids.copy()
    original = cache.save_block
    count = 0

    def fail_second(key):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("preservation failed")
        return original(key)

    with monkeypatch.context() as patch:
        patch.setattr(cache, "save_block", fail_second)
        with pytest.raises(RuntimeError, match="preservation failed"):
            engine.step_with_events()
    assert [seq.seq_id for seq in engine.scheduler.running] == ids
    assert manager.used_block_ids == before
    assert all(len(seq.block_table) == 1 and seq.num_scheduled_tokens == 0
               for seq in engine.scheduler.running)
    assert len(engine.model_runner.calls) == 3
    assert_pool_consistent(manager)
    assert engine.step_with_events().outputs == [(ids[0], [10, 11]), (ids[1], [20, 21])]
    assert not manager.used_block_ids
    assert_pool_consistent(manager)


def test_resume_allocation_failure_releases_partially_reserved_blocks(api, monkeypatch):
    engine, manager, _ = completed_prefix(api)
    cache = automatic_cache(engine)
    seq = api.Sequence([1] * 257, params(api, max_tokens=3))
    seq.append_token(10)

    def fail(*args):
        raise RuntimeError("preservation failed")

    monkeypatch.setattr(cache, "save_block", fail)
    with pytest.raises(RuntimeError, match="preservation failed"):
        manager.allocate_for_resume(seq, 257)
    assert not seq.block_table and seq.num_cached_tokens == 0
    assert not manager.used_block_ids
    assert_pool_consistent(manager)


def test_failed_later_admission_keeps_already_prefilled_chunks(api, monkeypatch):
    engine = cached_engine(api, [[20], [90], [91], [92], [10], [13, 14]],
                           budget=512, num_blocks=5)
    manager = engine.scheduler.block_manager
    engine.add_request([3] * 257, params(api, max_tokens=1))
    engine.step_with_events()
    cache = automatic_cache(engine, blocks=2)
    key = manager.compute_hash([3] * 256)
    cache.save_block(key)
    engine.add_request([9] * 1025, params(api, max_tokens=1))
    for _ in range(3):
        engine.step_with_events()
    assert key not in manager.hash_to_block_id
    first_id = engine.add_request([1] * 513, params(api, max_tokens=1))
    second_id = engine.add_request([3] * 257, params(api, max_tokens=1))
    engine.step_with_events()  # The first 512 tokens of the first request exist.
    first = engine.scheduler.waiting[0]
    blocks = first.block_table.copy()
    assert first.num_cached_tokens == 512

    def fail(*args):
        raise RuntimeError("restore failed")

    with monkeypatch.context() as patch:
        patch.setattr(cache, "_copy_entry", fail)
        with pytest.raises(RuntimeError, match="restore failed"):
            engine.step_with_events()
    assert [seq.seq_id for seq in engine.scheduler.waiting] == [first_id, second_id]
    assert not engine.scheduler.running
    assert first.block_table == blocks and first.num_cached_tokens == 512
    assert first.num_scheduled_tokens == 0
    assert manager.used_block_ids == set(blocks)
    assert_pool_consistent(manager)
    result = engine.step_with_events()
    assert result.num_tokens == 2
    assert result.outputs == [(first_id, [13]), (second_id, [14])]
    assert not manager.used_block_ids


def test_cpu_hits_do_not_make_an_oversized_active_request_fit(api):
    engine, manager, key = completed_prefix(api, [[90], [91]])
    cache = automatic_cache(engine, blocks=1)
    for prompt in ([3] * 257, [4]):
        engine.add_request(prompt, params(api, max_tokens=1))
        engine.step_with_events()
    assert key in cache and key not in manager.hash_to_block_id
    seq = api.Sequence([1] * 513, params(api, max_tokens=1))
    before = (list(manager.free_block_ids), cache.stats)
    assert manager.can_allocate(seq) == -1  # Three GPU pages needed, pool has two.
    assert (list(manager.free_block_ids), cache.stats) == before
    assert not seq.block_table
