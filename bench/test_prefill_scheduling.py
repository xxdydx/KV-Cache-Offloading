"""Scheduling/progress checks using the real engine and deterministic CPU outputs.

These establish token/work bounds and correctness, not GPU latency improvements.
"""

import sys

import pytest

from bench.test_engine_events import api, make_engine, params  # noqa: F401
from bench.test_engine_handoff import cached_engine
from bench.test_cpu_prefix_cache import assert_pool_consistent


class DeterministicRunner:
    def __init__(self):
        self.calls = []

    def call(self, method, seqs, is_prefill):
        assert method == "run"
        self.calls.append((is_prefill, [(seq.seq_id, seq.num_scheduled_tokens) for seq in seqs]))
        return [10 * seq[0] + seq.num_completion_tokens for seq in seqs]


def engine_for(api, policy, **kwargs):
    engine = make_engine(api, [], prefill_policy=policy, **kwargs)
    engine.model_runner = DeterministicRunner()
    return engine


def drain(engine, limit=2000):
    decisions, events = [], []
    while not engine.is_finished():
        assert len(decisions) < limit, "scheduler failed to make bounded progress"
        result = engine.step_with_events()
        decisions.append(engine.scheduler.last_schedule)
        events.extend(result.token_events)
        assert_pool_consistent(engine.scheduler.block_manager)
    assert not engine.scheduler.block_manager.used_block_ids
    return decisions, events


@pytest.mark.parametrize("policy", ["fixed", "decode_aware"])
def test_long_prefill_yields_to_decode_and_both_finish(api, policy):
    engine = engine_for(api, policy, prefill_chunk_size=4)
    active = engine.add_request([1], params(api, max_tokens=8))
    engine.step_with_events()
    incoming = engine.add_request([2] * 17, params(api, max_tokens=2))
    decisions, events = drain(engine)
    # Decode was due after the first prefill; a long waiting prompt cannot jump it.
    assert decisions[0].phase == "decode" and decisions[0].seq_ids == (active,)
    assert any(item.phase == "prefill" for item in decisions)
    for previous, current in zip(decisions, decisions[1:]):
        if previous.phase == "prefill" and current.decode_requests:
            assert current.phase == "decode"
    assert [event.output_index for event in events if event.seq_id == active] == list(range(1, 8))
    assert [event.output_index for event in events if event.seq_id == incoming] == [0, 1]
    assert all(item.scheduled_tokens <= item.prefill_budget
               for item in decisions if item.phase == "prefill")


def test_legacy_keeps_prefill_priority_for_comparison(api):
    engine = engine_for(api, "legacy", budget=8, prefill_chunk_size=2)
    active = engine.add_request([1], params(api, max_tokens=4))
    engine.step_with_events()
    engine.add_request([2] * 17, params(api, max_tokens=1))
    assert [engine.step_with_events().num_tokens for _ in range(3)] == [8, 8, 1]
    assert engine.scheduler.running[0].seq_id == active
    assert engine.step_with_events().num_tokens == -1


@pytest.mark.parametrize("policy, hard_cap, expected", [
    ("fixed", 50, 12), ("decode_aware", 50, 4),
    ("fixed", 2, 2), ("decode_aware", 2, 2),
])
def test_budget_accounts_for_decode_load_and_hard_token_cap(api, policy, hard_cap, expected):
    engine = engine_for(api, policy, budget=50, prefill_chunk_size=12,
                        prefill_min_chunk_size=3)
    for value in (1, 2):
        engine.add_request([value], params(api, max_tokens=5))
    engine.step_with_events()
    engine.scheduler.max_num_batched_tokens = hard_cap
    engine.add_request([3] * 25, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == -2
    result = engine.step_with_events()
    decision = engine.scheduler.last_schedule
    assert decision.decode_requests == 2
    assert decision.prefill_budget == result.num_tokens == expected
    assert result.token_events == []


def test_adaptive_budget_recovers_when_decode_load_disappears(api):
    engine = engine_for(api, "decode_aware", budget=20, prefill_chunk_size=12,
                        prefill_min_chunk_size=2)
    engine.add_request([1], params(api, max_tokens=3))
    engine.step_with_events()
    engine.add_request([2] * 25, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == -1
    assert engine.step_with_events().num_tokens == 6
    assert engine.step_with_events().num_tokens == -1  # Active decoder finishes.
    assert engine.step_with_events().num_tokens == 12
    assert engine.scheduler.last_schedule.decode_requests == 0
    drain(engine)


@pytest.mark.parametrize("policy", ["fixed", "decode_aware"])
def test_decode_batches_rotate_when_resident_requests_exceed_batch_slots(api, policy):
    engine = engine_for(api, policy, prefill_chunk_size=12)
    ids = [engine.add_request([value], params(api, max_tokens=3)) for value in (1, 2, 3)]
    engine.step_with_events()
    engine.scheduler.max_num_seqs = 1
    observed = [engine.step_with_events().token_events[0].seq_id for _ in range(6)]
    assert observed == ids + ids
    assert engine.is_finished()


@pytest.mark.parametrize("policy", ["legacy", "fixed", "decode_aware"])
def test_policy_preserves_outputs_and_token_indices_across_chunk_shapes(api, policy):
    engine = engine_for(api, policy, budget=8, prefill_chunk_size=3)
    ids = [engine.add_request([value] * length, params(api, max_tokens=3))
           for value, length in ((1, 7), (2, 2), (3, 13))]
    _, events = drain(engine)
    for value, request_id in enumerate(ids, 1):
        observed = [event for event in events if event.seq_id == request_id]
        assert [event.output_index for event in observed] == [0, 1, 2]
        assert [event.token_id for event in observed] == [10 * value + i for i in range(3)]
        assert [event.finished for event in observed] == [False, False, True]


def test_partial_prefill_releases_reservations_for_decode_growth(api):
    engine = engine_for(api, "fixed", budget=512, num_blocks=3, prefill_chunk_size=256)
    active = engine.add_request([1] * 255, params(api, max_tokens=4))
    engine.step_with_events()
    incoming = engine.add_request([2] * 257, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == -1  # Pending token reaches page boundary.
    assert engine.step_with_events().num_tokens == 256
    partial = engine.scheduler.waiting[0]
    assert len(partial.block_table) == 2 and partial.num_cached_tokens == 256
    result = engine.step_with_events()  # Must reclaim partial's slots to grow decode.
    assert result.num_tokens == -1 and result.token_events[0].seq_id == active
    assert not partial.block_table and partial.num_cached_tokens == 0
    decisions, events = drain(engine)
    assert any(event.seq_id == incoming and event.finished for event in events)
    assert decisions


@pytest.mark.parametrize("policy", ["fixed", "decode_aware"])
def test_no_decode_work_means_consecutive_prefill_chunks_can_progress(api, policy):
    engine = engine_for(api, policy, prefill_chunk_size=2)
    request = engine.add_request([1] * 5, params(api, max_tokens=1))
    assert engine.step_with_events().token_events == []
    assert engine.step_with_events().token_events == []
    result = engine.step_with_events()
    assert result.outputs == [(request, [10])]
    assert engine.is_finished()


def test_failed_admission_does_not_advance_policy_phase(api, monkeypatch):
    engine = engine_for(api, "fixed", prefill_chunk_size=4)
    engine.add_request([1], params(api, max_tokens=3))
    engine.step_with_events()
    engine.step_with_events()
    engine.add_request([2] * 5, params(api, max_tokens=1))
    previous = engine.scheduler.last_schedule

    def fail(*args):
        raise RuntimeError("copy failed")

    with monkeypatch.context() as patch:
        patch.setattr(engine.scheduler.block_manager, "allocate", fail)
        with pytest.raises(RuntimeError, match="copy failed"):
            engine.step_with_events()
    assert engine.scheduler.last_schedule is previous
    assert engine.step_with_events().num_tokens == 4  # Retry prefill, not decode.
    drain(engine)


def test_cpu_restoration_and_prefill_budget_compose(api):
    engine = cached_engine(api, [[10], [90], [91], [92], [12]], budget=512, num_blocks=2,
                           prefill_policy="fixed", prefill_chunk_size=512)
    engine.add_request([1] * 257, params(api, max_tokens=1))
    engine.step_with_events()
    pool = engine.model_runner.kv_cache
    cache = engine.create_cpu_prefix_cache(pool[:, :, 0].numel() * pool.element_size(), automatic=True)
    for value, length in ((3, 257), (4, 1)):
        engine.add_request([value] * length, params(api, max_tokens=1))
        engine.step_with_events()
    engine.scheduler.prefill_chunk_size = 8
    engine.add_request([1] * 269, params(api, max_tokens=1))
    assert engine.step_with_events().num_tokens == 8
    assert cache.stats["restored_blocks"] == 1
    assert engine.step_with_events().num_tokens == 5
    assert engine.is_finished()
    assert_pool_consistent(engine.scheduler.block_manager)


@pytest.mark.parametrize("kwargs, message", [
    ({"prefill_policy": "unknown"}, "prefill_policy"),
    ({"prefill_chunk_size": 0}, "prefill_chunk_size"),
    ({"prefill_chunk_size": True}, "prefill_chunk_size"),
    ({"prefill_chunk_size": 1.5}, "prefill_chunk_size"),
    ({"prefill_min_chunk_size": 0}, "prefill_min_chunk_size"),
    ({"prefill_chunk_size": 4, "prefill_min_chunk_size": 5}, "must not exceed"),
    ({"max_num_batched_tokens": 0}, "max_num_batched_tokens"),
    ({"max_num_seqs": 0}, "max_num_seqs"),
])
def test_config_rejects_invalid_policy_before_model_loading(api, kwargs, message):
    config = sys.modules["nanovllm.config"].Config
    with pytest.raises(ValueError, match=message):
        config(model="/not/a/model", **kwargs)


def test_oversized_request_fails_explicitly_instead_of_spinning(api):
    engine = engine_for(api, "fixed", num_blocks=1)
    engine.add_request([1] * 257, params(api, max_tokens=1))
    with pytest.raises(RuntimeError, match="GPU KV capacity"):
        engine.step_with_events()
