"""Split streams, FP8 payloads, split (hybrid) admission, MRU eviction and prefetch hints."""

import sys

import pytest
import torch

from bench.test_engine_events import api, params  # noqa: F401
from bench.test_engine_handoff import cached_engine
from bench.test_cpu_prefix_cache import assert_pool_consistent, overwrite_prefix
from bench.test_automatic_cpu_cache import slow_copies, write_scheduled_kv


def Sequence(*args):
    return sys.modules["nanovllm.engine.sequence"].Sequence(*args)


def device_or_skip(device):
    slow = device == "cuda-slow-copies"
    device = "cuda" if slow else device
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    return device, slow


def finite_prefix(api, device, batches=(), blocks=1, num_blocks=2):
    """A finished request whose `blocks` full blocks are idle GPU hits (finite bf16 KV)."""
    steps = -(-(256 * blocks + 1) // 512) + 1  # Prefill chunks (budget 512), then one decode.
    engine = cached_engine(api, [[90]] * steps + [*batches], num_blocks=num_blocks)
    engine.model_runner.kv_cache = engine.model_runner.kv_cache.to(device)
    engine.add_request([1] * (256 * blocks + 1), params(api, max_tokens=2))
    while not engine.is_finished():
        engine.step_with_events()
    manager = engine.scheduler.block_manager
    keys, h = [], -1
    for _ in range(blocks):
        h = manager.compute_hash([1] * 256, h)
        keys.append(h)
    assert all(key in manager.hash_to_block_id for key in keys)
    return engine, manager, keys


def block_bytes(pool, block_id):
    return pool[:, :, block_id].contiguous().cpu().view(torch.uint8).clone()


def test_plan_split_balances_recompute_against_restore(api):
    plan = sys.modules["nanovllm.engine.block_manager"].plan_split
    assert plan(8, 256, 1.0, 1 / 256, 0) == 4  # Equal per-block costs: half each.
    assert plan(8, 256, 1.0, 0, 0) == 8  # Free recompute: no restores.
    assert plan(8, 256, 0, 1, 0) == 0  # Free restores: no recompute.
    assert plan(0, 256, 1.0, 1.0, 1.0) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_payload_is_smaller_and_restores_within_quantization_error(api, device):
    device, _ = device_or_skip(device)
    engine, manager, (key,) = finite_prefix(api, device, [[90], [91]])
    pool = engine.model_runner.kv_cache
    source = pool[:, :, manager.hash_to_block_id[key]].float().cpu().clone()
    bf16_bytes = pool[:, :, 0].numel() * pool.element_size()
    cache = engine.create_cpu_prefix_cache(max_bytes=4 * bf16_bytes, fp8=True)
    numel = pool[:, :, 0].numel()
    assert cache.block_bytes == numel + numel // pool.shape[-1] * pool.element_size()
    assert cache.save_block(key)
    overwrite_prefix(api, engine)
    assert key not in manager.hash_to_block_id
    target = cache.restore_block(key)
    cache.synchronize()
    restored = pool[:, :, target].float().cpu()
    # e4m3 keeps 3 mantissa bits: error is at most 2^-4 of each token-head's max.
    bound = source.abs().amax(-1, keepdim=True) / 16
    assert torch.all((restored - source).abs() <= bound)
    assert cache.stats["restored_blocks"] == 1


def split_stream_cache(api, device):
    engine, manager, (key,) = finite_prefix(api, device, [[90], [91], [92], [93]], num_blocks=3)
    pool = engine.model_runner.kv_cache
    size = pool[:, :, 0].numel() * pool.element_size()
    cache = engine.create_cpu_prefix_cache(max_bytes=size, split_streams=True, preallocate=True)
    assert cache._restore_stream is not cache._copy_stream
    # Pinned host memory is recycled across tests; poison it so a restore that
    # reads a slot too early cannot see matching stale bytes.
    for slab in cache._slabs:
        slab.view(torch.uint8).fill_(0xAB)
    return engine, manager, key, pool, cache


def forget_gpu_copy(manager, key):
    """Unpublish a GPU block without touching its bytes (and without ordering)."""
    block_id = manager.hash_to_block_id.pop(key)
    manager.blocks[block_id].hash, manager.blocks[block_id].token_ids = -1, []
    return block_id


@pytest.mark.parametrize("device", ["cuda", "cuda-slow-copies"])
def test_split_streams_restore_waits_for_its_save(api, device):
    device, slow = device_or_skip(device)
    engine, manager, key, pool, cache = split_stream_cache(api, device)
    if slow:
        slow_copies(cache, ["save_block"])  # Fast restores would overtake a missing wait.
    expected = block_bytes(pool, manager.hash_to_block_id[key])
    assert cache.save_block(key)
    forget_gpu_copy(manager, key)
    target = cache.restore_block(key)
    cache.synchronize()
    assert torch.equal(block_bytes(pool, target), expected)


@pytest.mark.parametrize("device", ["cuda", "cuda-slow-copies"])
def test_split_streams_slot_reuse_waits_for_pending_restore(api, device):
    device, slow = device_or_skip(device)
    engine, manager, key, pool, cache = split_stream_cache(api, device)
    expected = block_bytes(pool, manager.hash_to_block_id[key])
    assert cache.save_block(key)
    cache.synchronize()
    forget_gpu_copy(manager, key)
    if slow:
        slow_copies(cache, ["_copy_entry"])  # A fast save would overwrite the slot first.
    # A background (prefetch-style) restore: unlike restore_block, the compute
    # stream does not wait for it, so only the slot-read event orders the save.
    entry = cache._entries[key]
    target = manager.restore_cached_block(
        key, entry.token_ids, lambda block_id: cache._copy_entry(entry, block_id, wait=False))
    assert cache.discard(key)
    other = next(i for i in range(len(manager.blocks)) if i != target)
    pool[:, :, other].fill_(7)
    manager.blocks[other].update(12345, [2] * 256)
    manager.hash_to_block_id[12345] = other
    assert cache.save_block(12345)  # Reuses the slot the restore is still reading.
    cache.synchronize()
    assert torch.equal(block_bytes(pool, target), expected)
    assert torch.equal(cache._slot_view(0)[0].view(torch.uint8), block_bytes(pool, other))


def drop_gpu_copies(cache, manager, pool, keys):
    """Leave only CPU payloads for keys (finish saves first: zero_() bypasses ordering)."""
    for key in keys:
        cache.save_block(key)
    cache.synchronize()
    for key in keys:
        block_id = manager.hash_to_block_id.pop(key)
        manager.blocks[block_id].hash, manager.blocks[block_id].token_ids = -1, []
        pool[:, :, block_id].zero_()


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda-slow-copies"])
def test_split_admission_recomputes_head_while_restoring_tail(api, device):
    device, slow = device_or_skip(device)
    engine, manager, keys = finite_prefix(api, device, [[90], [91], [92]], blocks=3, num_blocks=5)
    pool = engine.model_runner.kv_cache
    size = pool[:, :, 0].numel() * pool.element_size()
    expected = [block_bytes(pool, manager.hash_to_block_id[key]) for key in keys]
    cache = engine.create_cpu_prefix_cache(max_bytes=3 * size, automatic=True)
    # Per-block recompute and restore cost the same: recompute 1 block, restore 2.
    cache.hybrid = (1.0, 1 / 256, 0)
    drop_gpu_copies(cache, manager, pool, keys)
    if slow:
        slow_copies(cache)
    write_scheduled_kv(engine)
    request = engine.add_request([1] * 768 + [5] * 10, params(api, max_tokens=1))
    head = engine.step_with_events()
    seq = next(seq for seq in engine.scheduler.waiting if seq.seq_id == request)
    assert head.num_tokens == 256 and seq.restore_span is None and seq.num_cached_tokens == 768
    assert cache.stats["restored_blocks"] == 2
    # Head blocks were computed, the rest restored; reads of the restored span
    # on the compute stream are ordered after the copies.
    assert torch.equal(block_bytes(pool, seq.block_table[1]), expected[1])
    assert torch.equal(block_bytes(pool, seq.block_table[2]), expected[2])
    assert manager.hash_to_block_id[keys[0]] == seq.block_table[0]
    tail = engine.step_with_events()
    assert tail.num_tokens == 10 and tail.outputs
    assert engine.is_finished() and not manager.used_block_ids
    assert_pool_consistent(manager)


def test_split_admission_that_recomputes_everything_uses_no_restore(api):
    engine, manager, keys = finite_prefix(api, "cpu", [[90], [91]], blocks=2, num_blocks=4)
    pool = engine.model_runner.kv_cache
    size = pool[:, :, 0].numel() * pool.element_size()
    cache = engine.create_cpu_prefix_cache(max_bytes=2 * size, automatic=True)
    cache.hybrid = (1.0, 0, 0)  # Recompute is free.
    drop_gpu_copies(cache, manager, pool, keys)
    request = engine.add_request([1] * 512 + [5], params(api, max_tokens=1))
    engine.step_with_events()
    seq = next(seq for seq in engine.scheduler.waiting if seq.seq_id == request)
    assert seq.num_cached_tokens == 512 and seq.restore_span is None  # Computed, not skipped.
    assert cache.stats["restored_blocks"] == 0


def test_preempting_a_split_admission_resets_its_span(api):
    engine, manager, keys = finite_prefix(api, "cpu", blocks=2, num_blocks=4)
    pool = engine.model_runner.kv_cache
    size = pool[:, :, 0].numel() * pool.element_size()
    cache = engine.create_cpu_prefix_cache(max_bytes=2 * size, automatic=True)
    cache.hybrid = (1.0, 1 / 256, 0)
    drop_gpu_copies(cache, manager, pool, keys)
    seq = Sequence([1] * 512 + [5], params(api, max_tokens=1))
    manager.allocate(seq, manager.can_allocate(seq))
    assert seq.restore_span == (256, 512) and seq.num_cached_tokens == 0
    manager.deallocate(seq)
    assert seq.restore_span is None and not manager.used_block_ids
    assert_pool_consistent(manager)


def test_hybrid_requires_automatic_cache(api):
    engine, _, _ = finite_prefix(api, "cpu")
    cache = engine.create_cpu_prefix_cache(max_bytes=1 << 20)
    with pytest.raises(ValueError, match="automatic"):
        cache.hybrid = (1.0, 0, 0)


def test_mru_eviction_recycles_the_latest_release_tail_first(api):
    block_manager = sys.modules["nanovllm.engine.block_manager"]
    manager = block_manager.BlockManager(6, 256, "mru")
    seqs = []
    for token in (1, 2):
        seq = Sequence([token] * 512 + [9], params(api, max_tokens=1))
        manager.allocate(seq, 0)
        seq.num_cached_tokens, seq.num_scheduled_tokens = 0, len(seq)
        manager.hash_blocks(seq)
        seqs.append(seq)
    first, second = ([*seq.block_table] for seq in seqs)
    for seq in seqs:
        manager.deallocate(seq)
    # Unpublished partial tails go first, then the later request's last full block.
    order = [manager._allocate_block() for _ in range(4)]
    assert set(order[:2]) == {first[2], second[2]}
    assert order[2:] == [second[1], second[0]]
    with pytest.raises(ValueError):
        block_manager.BlockManager(2, 256, "fifo")


def test_prefetch_hint_restores_cpu_hits_into_free_blocks(api):
    engine, manager, keys = finite_prefix(api, "cpu", blocks=2, num_blocks=4)
    pool = engine.model_runner.kv_cache
    size = pool[:, :, 0].numel() * pool.element_size()
    expected = [block_bytes(pool, manager.hash_to_block_id[key]) for key in keys]
    cache = engine.create_cpu_prefix_cache(max_bytes=2 * size, automatic=True, prefetch=True)
    drop_gpu_copies(cache, manager, pool, keys)
    hint = Sequence([1] * 512 + [7], params(api, max_tokens=1))
    assert engine.prefetch_prefix(hint) == 2
    assert engine.prefetch_prefix(hint) == 0  # Nothing changed: throttled.
    for key, payload in zip(keys, expected):
        assert torch.equal(block_bytes(pool, manager.hash_to_block_id[key]), payload)
    engine.add_request([9], params(api, max_tokens=1))
    assert engine.prefetch_prefix([1] * 600) == 0  # Never while a request waits.
    assert_pool_consistent(manager)


def test_later_requests_join_the_step_of_a_split_admission_head(api):
    engine, manager, keys = finite_prefix(api, "cpu", [[90, 91], [92]], blocks=3, num_blocks=6)
    pool = engine.model_runner.kv_cache
    size = pool[:, :, 0].numel() * pool.element_size()
    cache = engine.create_cpu_prefix_cache(max_bytes=3 * size, automatic=True)
    cache.hybrid = (1.0, 1 / 256, 0)
    drop_gpu_copies(cache, manager, pool, keys)
    split = engine.add_request([1] * 768 + [5] * 10, params(api, max_tokens=1))
    short = engine.add_request([8] * 10, params(api, max_tokens=1))
    step = engine.step_with_events()
    assert step.num_tokens == 256 + 10  # Head chunk and the short prompt in one batch.
    assert [seq.seq_id for seq in engine.scheduler.waiting] == [split]
    assert [event.seq_id for event in step.token_events] == [short]
    assert engine.step_with_events().num_tokens == 10
    assert engine.is_finished()
    assert_pool_consistent(manager)
