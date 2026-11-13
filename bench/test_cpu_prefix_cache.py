"""Full-block CPU preservation using real cache/scheduler code and CPU tensors."""

import sys

import pytest
import torch

from bench.test_engine_events import api, params  # noqa: F401
from bench.test_engine_handoff import cached_engine


def completed_prefix(api, extra_batches=(), device="cpu"):
    engine = cached_engine(api, [[10], [11], *extra_batches], num_blocks=2)
    engine.model_runner.kv_cache = engine.model_runner.kv_cache.to(device)
    # Include arbitrary bit patterns (including NaNs) to check byte preservation.
    pool = engine.model_runner.kv_cache
    raw = torch.arange(pool.numel() * pool.element_size(), dtype=torch.int64).remainder(256).to(torch.uint8)
    pool.view(torch.uint8).copy_(raw.reshape(pool.view(torch.uint8).shape).to(device))
    engine.add_request([1] * 257, params(api, max_tokens=2))
    engine.step_with_events()
    engine.step_with_events()
    manager = engine.scheduler.block_manager
    key = manager.compute_hash([1] * 256)
    assert engine.is_finished() and key in manager.hash_to_block_id
    assert not manager.used_block_ids
    return engine, manager, key


def new_cache(engine, blocks=1):
    pool = engine.model_runner.kv_cache
    block_bytes = pool[:, :, 0].numel() * pool.element_size()
    return engine.create_cpu_prefix_cache(max_bytes=blocks * block_bytes)


def overwrite_prefix(api, engine):
    engine.add_request([3] * 257, params(api, max_tokens=1))
    engine.step_with_events()
    # Move the old physical block to the back of the free queue as well.
    engine.add_request([4], params(api, max_tokens=1))
    engine.step_with_events()


def assert_pool_consistent(manager):
    free = list(manager.free_block_ids)
    assert len(free) == len(set(free))
    assert set(free).isdisjoint(manager.used_block_ids)
    assert set(free) | manager.used_block_ids == set(range(len(manager.blocks)))
    assert all(manager.blocks[i].ref_count == 0 for i in free)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_round_trip_survives_recycling_and_reduces_next_turn_prefill(api, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    engine, manager, key = completed_prefix(api, [[90], [91], [12]], device)
    pool = engine.model_runner.kv_cache
    source_id = manager.hash_to_block_id[key]
    expected = pool[:, :, source_id].contiguous().cpu().view(torch.uint8).clone()
    cache = new_cache(engine)
    assert cache.save_block(key)
    assert key in cache and len(cache) == 1
    assert cache.bytes_used == cache.block_bytes == cache.max_bytes
    overwrite_prefix(api, engine)
    assert key not in manager.hash_to_block_id
    pool.fill_(-7)
    destination_id = cache.restore_block(key)
    assert destination_id != source_id
    assert torch.equal(pool[:, :, destination_id].contiguous().cpu().view(torch.uint8), expected)
    assert manager.blocks[destination_id].ref_count == 0
    assert_pool_consistent(manager)
    prompt = [1] * 257 + [10, 11] + [2] * 10
    request_id = engine.add_request(prompt, params(api, max_tokens=1))
    result = engine.step_with_events()
    assert result.num_tokens == 13  # 269 prompt positions minus the restored 256.
    assert result.outputs == [(request_id, [12])]
    assert not manager.used_block_ids
    assert_pool_consistent(manager)
    cold = cached_engine(api, [[12]], num_blocks=2)
    cold.add_request(prompt, params(api, max_tokens=1))
    assert cold.step_with_events().num_tokens == 269


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_hash_is_published_only_after_copy_and_failure_returns_block(api, error):
    engine, manager, key = completed_prefix(api)
    # A different valid hash avoids the existing GPU-hit shortcut.
    restored_key = manager.compute_hash([5] * 256)
    called = []

    def copy_then_fail(block_id):
        assert restored_key not in manager.hash_to_block_id
        assert block_id in manager.used_block_ids
        assert block_id not in manager.free_block_ids
        called.append(block_id)
        raise error("copy readiness failed")

    with pytest.raises(error, match="readiness"):
        manager.restore_cached_block(restored_key, [5] * 256, copy_then_fail)
    assert len(called) == 1
    assert restored_key not in manager.hash_to_block_id
    assert manager.blocks[called[0]].hash == -1
    assert not manager.used_block_ids
    assert_pool_consistent(manager)


def test_failed_restore_keeps_cpu_copy_and_retry_is_byte_exact(api, monkeypatch):
    engine, manager, key = completed_prefix(api, [[90], [91]])
    cache = new_cache(engine)
    expected = engine.model_runner.kv_cache[:, :, manager.hash_to_block_id[key]].contiguous().view(torch.uint8).clone()
    cache.save_block(key)
    overwrite_prefix(api, engine)

    def fail_copy(destination, source, **kwargs):
        destination.fill_(0)
        raise RuntimeError("injected copy failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail_copy)
        with pytest.raises(RuntimeError, match="injected"):
            cache.restore_block(key)
    assert key in cache and cache.bytes_used == cache.block_bytes
    assert key not in manager.hash_to_block_id
    assert_pool_consistent(manager)
    restored = cache.restore_block(key)
    assert torch.equal(engine.model_runner.kv_cache[:, :, restored].contiguous().view(torch.uint8), expected)


def test_failed_save_does_not_publish_or_consume_budget(api, monkeypatch):
    engine, manager, key = completed_prefix(api)
    cache = new_cache(engine)
    mapping = manager.hash_to_block_id.copy()

    def fail_copy(*args, **kwargs):
        raise RuntimeError("host allocation/copy failed")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail_copy)
        with pytest.raises(RuntimeError, match="host allocation/copy"):
            cache.save_block(key)
    assert len(cache) == cache.bytes_used == 0
    assert manager.hash_to_block_id == mapping
    assert_pool_consistent(manager)
    assert cache.save_block(key)


def test_budget_deduplication_and_explicit_discard(api):
    engine, manager, key = completed_prefix(api, [[90], [91]])
    cache = new_cache(engine)
    assert cache.save_block(key)
    assert not cache.save_block(key)
    overwrite_prefix(api, engine)
    assert not cache.save_block(key)  # CPU copy survives the GPU miss.
    # Publish another full block using the real scheduler.
    engine.model_runner.batches = iter([[92]])
    engine.add_request([6] * 257, params(api, max_tokens=1))
    engine.step_with_events()
    other = manager.compute_hash([6] * 256)
    with pytest.raises(MemoryError, match="budget"):
        cache.save_block(other)
    assert cache.bytes_used == cache.block_bytes and other not in cache
    assert cache.discard(key)
    assert not cache.discard(key)
    assert cache.bytes_used == 0
    assert cache.save_block(other)
    cache.clear()
    assert cache.bytes_used == len(cache) == 0
    assert other in manager.hash_to_block_id  # CPU release leaves GPU hits intact.


def test_budget_smaller_than_one_block_rejects_before_allocating(api, monkeypatch):
    engine, _, key = completed_prefix(api)
    cache = engine.create_cpu_prefix_cache(max_bytes=1)

    def fail_allocation(*args, **kwargs):
        raise AssertionError("budget check must precede host allocation")

    monkeypatch.setattr(torch, "empty", fail_allocation)
    with pytest.raises(MemoryError, match="budget"):
        cache.save_block(key)
    assert cache.bytes_used == 0


def test_restore_without_space_preserves_active_request_and_cpu_entry(api):
    engine, manager, key = completed_prefix(api, [[90], [91]])
    cache = new_cache(engine)
    cache.save_block(key)
    engine.add_request([3] * 257, params(api, max_tokens=2))
    engine.step_with_events()
    before = manager.used_block_ids.copy()
    assert not manager.free_block_ids and key not in manager.hash_to_block_id
    with pytest.raises(MemoryError, match="free KV blocks"):
        cache.restore_block(key)
    assert manager.used_block_ids == before
    assert key in cache
    assert_pool_consistent(manager)
    engine.step_with_events()
    assert cache.restore_block(key) in manager.hash_to_block_id.values()
    assert_pool_consistent(manager)


def test_existing_live_gpu_hit_is_not_overwritten_or_referenced_again(api, monkeypatch):
    engine = cached_engine(api, [[10]], num_blocks=2)
    engine.add_request([1] * 257, params(api, max_tokens=2))
    engine.step_with_events()
    manager = engine.scheduler.block_manager
    key = manager.compute_hash([1] * 256)
    cache = new_cache(engine)
    cache.save_block(key)
    block_id = manager.hash_to_block_id[key]
    references = manager.blocks[block_id].ref_count

    def fail_copy(*args, **kwargs):
        raise AssertionError("GPU hit must not be copied over")

    monkeypatch.setattr(torch.Tensor, "copy_", fail_copy)
    assert cache.restore_block(key) == block_id
    assert manager.blocks[block_id].ref_count == references
    assert not manager.free_block_ids
    assert_pool_consistent(manager)


def test_restore_chained_prefix_blocks_preserves_ancestry(api):
    engine = cached_engine(api, [[10], [90], [11]], budget=768, num_blocks=3)
    manager = engine.scheduler.block_manager
    original = [1] * 256 + [2] * 256 + [3]
    engine.add_request(original, params(api, max_tokens=1))
    engine.step_with_events()
    first = manager.compute_hash([1] * 256)
    second = manager.compute_hash([2] * 256, first)
    cache = new_cache(engine, blocks=2)
    cache.save_block(first)
    cache.save_block(second)
    engine.add_request([8] * 513, params(api, max_tokens=1))
    engine.step_with_events()
    assert first not in manager.hash_to_block_id and second not in manager.hash_to_block_id
    cache.restore_block(first)
    cache.restore_block(second)
    engine.add_request(original + [9] * 10, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == 11
    assert_pool_consistent(manager)


def test_rejects_unknown_and_unpublished_partial_blocks(api):
    engine, manager, key = completed_prefix(api)
    cache = new_cache(engine)
    with pytest.raises(KeyError):
        cache.save_block(-1)
    with pytest.raises(KeyError):
        cache.restore_block(key)
    block = manager.blocks[manager.hash_to_block_id[key]]
    block.token_ids = block.token_ids[:-1]
    with pytest.raises(ValueError, match="full prefix block"):
        cache.save_block(key)
    assert len(cache) == 0


@pytest.mark.parametrize("bad_budget", [0, -1, 1.5, True])
def test_invalid_budget_is_rejected_before_creating_cache(api, bad_budget):
    engine, _, _ = completed_prefix(api)
    with pytest.raises(ValueError, match="positive integer"):
        engine.create_cpu_prefix_cache(bad_budget)
    assert getattr(engine, "_cpu_prefix_cache", None) is None


def test_tp_and_layout_mismatch_are_rejected(api):
    engine, _, _ = completed_prefix(api)
    engine.model_runner.world_size = 2
    with pytest.raises(NotImplementedError, match="TP=1"):
        new_cache(engine)
    engine.model_runner.world_size = 1
    engine.model_runner.block_size = 128
    with pytest.raises(ValueError, match="layout"):
        new_cache(engine)


def test_conflicting_gpu_token_metadata_is_not_silently_reused(api):
    engine, manager, key = completed_prefix(api)
    cache = new_cache(engine)
    cache.save_block(key)
    manager.blocks[manager.hash_to_block_id[key]].token_ids = [7] * 256
    with pytest.raises(ValueError, match="conflicting token"):
        cache.save_block(key)
    with pytest.raises(ValueError, match="conflicting token"):
        cache.restore_block(key)
    assert cache.bytes_used == cache.block_bytes


def test_close_releases_pool_references_and_engine_exit_closes_cache(api):
    engine, _, key = completed_prefix(api)
    cache = new_cache(engine)
    cache.save_block(key)
    with pytest.raises(RuntimeError, match="already has"):
        new_cache(engine)
    cache.close()
    cache.close()
    assert cache.closed and cache.bytes_used == 0 and cache._manager is None
    with pytest.raises(RuntimeError, match="closed"):
        cache.save_block(key)
    with pytest.raises(RuntimeError, match="closed"):
        cache.restore_block(key)
    replacement = new_cache(engine)
    replacement.save_block(key)
    engine.ps = []
    calls = []
    engine.model_runner.call = lambda method: calls.append(method)
    engine.exit()
    assert replacement.closed and replacement.bytes_used == 0
    assert calls == ["exit"]


def test_discarded_slots_are_reused_without_new_host_allocation(api, monkeypatch):
    engine, manager, key = completed_prefix(api, [[90], [91]])
    cache = new_cache(engine)
    assert cache.reserved_bytes == 0
    assert cache.save_block(key)
    assert cache.reserved_bytes == cache.block_bytes
    overwrite_prefix(api, engine)
    other = manager.compute_hash([3] * 256)
    assert cache.discard(key)

    def fail_allocation(*args, **kwargs):
        raise AssertionError("a free slot must be reused")

    monkeypatch.setattr(torch, "empty", fail_allocation)
    assert cache.save_block(other)
    assert cache.reserved_bytes == cache.bytes_used == cache.block_bytes


def test_preallocated_multi_slab_slots_round_trip_byte_exact(api, monkeypatch):
    monkeypatch.setattr(sys.modules["nanovllm.engine.cpu_prefix_cache"], "_SLAB_CHUNK_BYTES", 1)
    engine, manager, key = completed_prefix(api, [[90], [91]])
    pool = engine.model_runner.kv_cache
    cache = engine.create_cpu_prefix_cache(max_bytes=3 * pool[:, :, 0].numel() * pool.element_size(),
                                           preallocate=True)
    assert len(cache._slabs) == 3 and cache.reserved_bytes == 3 * cache.block_bytes
    expected = {key: pool[:, :, manager.hash_to_block_id[key]].contiguous().view(torch.uint8).clone()}
    cache.save_block(key)
    overwrite_prefix(api, engine)
    other = manager.compute_hash([3] * 256)
    expected[other] = pool[:, :, manager.hash_to_block_id[other]].contiguous().view(torch.uint8).clone()
    cache.save_block(other)
    assert cache._entries[key].slot != cache._entries[other].slot
    for prefix_hash, data in expected.items():
        manager.hash_to_block_id.pop(prefix_hash, None)
        restored = cache.restore_block(prefix_hash)
        assert torch.equal(pool[:, :, restored].contiguous().view(torch.uint8), data)


def test_save_on_release_requires_automatic_mode(api):
    engine, _, _ = completed_prefix(api)
    with pytest.raises(ValueError, match="automatic"):
        engine.create_cpu_prefix_cache(max_bytes=1 << 20, save_on_release=True)
    assert getattr(engine, "_cpu_prefix_cache", None) is None
