"""Check phase accounting against the real scheduler without a GPU."""

from types import SimpleNamespace

import pytest

from bench.baseline import measure_batch
from bench.test_engine_events import api, make_engine, params  # noqa: F401


def test_first_output_is_prefill_and_remaining_outputs_are_decode(api):
    engine = make_engine(api, [[10, 20], [11, 21], [12, 22]], budget=10)
    result = measure_batch(engine, [[1] * 5, [2] * 5],
                           params(api, max_tokens=3, ignore_eos=True), lambda: None)
    assert [(step["phase"], step["tokens"]) for step in result["steps"]] == [
        ("prefill", 10), ("decode", 2), ("decode", 2)]
    assert result["output_token_ids"] == [[10, 11, 12], [20, 21, 22]]
    assert result["decode_tokens_per_s"] == 4 / result["decode_wall_s"]


def test_single_output_has_no_decode_measurement(api):
    engine = make_engine(api, [[10]], budget=8)
    result = measure_batch(engine, [[1] * 5], params(api, max_tokens=1), lambda: None)
    assert result["decode_wall_s"] == 0
    assert result["decode_tokens_per_s"] is None
    assert result["mean_decode_step_ms"] is None


def test_prefill_cache_hit_is_rejected_instead_of_inflating_throughput(api):
    engine = make_engine(api, [[10]], budget=8)
    original = engine.step_with_events

    def cached_step():
        result = original()
        return SimpleNamespace(num_tokens=result.num_tokens - 1, token_events=result.token_events)

    engine.step_with_events = cached_step
    with pytest.raises(RuntimeError, match="prefix-cache hit"):
        measure_batch(engine, [[1] * 5], params(api, max_tokens=1), lambda: None)


def test_missing_output_event_is_rejected(api):
    engine = make_engine(api, [[10]], budget=8)
    original = engine.step_with_events

    def missing_event():
        result = original()
        return SimpleNamespace(num_tokens=result.num_tokens, token_events=[])

    engine.step_with_events = missing_event
    with pytest.raises(RuntimeError, match="Output token count"):
        measure_batch(engine, [[1] * 5], params(api, max_tokens=1), lambda: None)
