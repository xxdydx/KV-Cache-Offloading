"""Deterministic checks for online arrival accounting and measurement integrity."""

import pytest

from bench.prefill_scheduling import Request, make_workload, run_workload
from bench.test_engine_events import api, params  # noqa: F401
from bench.test_prefill_scheduling import engine_for


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def measured_engine(api, policy):
    engine = engine_for(api, policy, budget=32, prefill_chunk_size=2)
    clock = Clock()
    original = engine.model_runner.call

    def run(method, seqs, is_prefill):
        # A test clock for measurement accounting, NOT a model of GPU performance.
        clock.now += 0.001 * (sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else 1)
        return original(method, seqs, is_prefill)

    engine.model_runner.call = run
    return engine, clock


def measure(api, engine, clock, workload, **kwargs):
    return run_workload(engine, workload, lambda count: params(api, max_tokens=count, ignore_eos=True),
                        lambda: None, clock=clock, sleep=clock.sleep, **kwargs)


def test_workload_is_identical_across_policies_and_reproducible():
    first = make_workload(vocab_size=100, seed=7)
    assert first == make_workload(vocab_size=100, seed=7)
    assert first != make_workload(vocab_size=100, seed=8)
    assert len(first) == 12
    assert {item.group for item in first} == {"initial", "short", "long"}
    assert [item.arrival_s for item in first] == sorted(item.arrival_s for item in first)


@pytest.mark.parametrize("policy", ["legacy", "fixed", "decode_aware"])
def test_all_work_is_counted_including_arrivals_during_execution(api, policy):
    engine, clock = measured_engine(api, policy)
    workload = [Request("active", "initial", 0, [1], 8),
                Request("incoming", "long", 0.0005, [2] * 9, 1),
                Request("short", "short", 0.0035, [3], 1)]
    result = measure(api, engine, clock, workload)
    summary = result["summary"]
    assert summary["requests"] == 3
    assert summary["processed_prefill_tokens"] == 11
    assert summary["contended_prefill_steps"] > 0
    assert summary["output_tokens_per_s"] == pytest.approx(10 / summary["wall_s"])
    assert summary["completed_requests_per_s"] == pytest.approx(3 / summary["wall_s"])
    for item in workload:
        record = result["requests"][item.name]
        assert record["submitted_s"] >= item.arrival_s
        assert record["ttft_s"] == pytest.approx(record["token_times_s"][0] - item.arrival_s)
        assert record["prefill_wait_s"] == pytest.approx(record["first_scheduled_s"] - item.arrival_s)
        assert len(record["token_times_s"]) == item.output_tokens
        assert record["finished"]
    # The original arrival counts even though the engine submitted it later.
    incoming = result["requests"]["incoming"]
    assert incoming["submitted_s"] == 0.001
    assert incoming["ttft_s"] > incoming["token_times_s"][0] - incoming["submitted_s"]


def test_interleaving_is_visible_in_test_clock_gaps_and_preserves_outputs(api):
    workload = [Request("active", "initial", 0, [1], 8),
                Request("incoming", "long", 0.0005, [2] * 9, 1)]
    results = {}
    for policy in ("legacy", "fixed", "decode_aware"):
        engine, clock = measured_engine(api, policy)
        results[policy] = measure(api, engine, clock, workload)
    legacy_gap = results["legacy"]["summary"]["groups"]["initial"]["inter_token_ms"]["p99"]
    for policy in ("fixed", "decode_aware"):
        assert results[policy]["summary"]["groups"]["initial"]["inter_token_ms"]["p99"] < legacy_gap
        for item in workload:
            assert results[policy]["requests"][item.name]["token_ids"] == results["legacy"]["requests"][item.name]["token_ids"]


def test_idle_arrival_gaps_count_in_total_wall_time_and_single_tokens_have_no_itl(api):
    engine, clock = measured_engine(api, "fixed")
    result = measure(api, engine, clock, [Request("late", "short", 0.1, [1], 1)])
    assert result["summary"]["wall_s"] == pytest.approx(0.101)
    assert result["summary"]["ttft_ms"]["p50"] == pytest.approx(1)
    assert result["summary"]["inter_token_ms"] == {"p50": None, "p95": None, "p99": None}


def test_timeout_keeps_partial_trace_and_cannot_be_reported_as_success(api):
    engine, clock = measured_engine(api, "fixed")
    trace = []
    workload = [Request("active", "initial", 0, [1], 100)]
    with pytest.raises(TimeoutError, match="timeout"):
        measure(api, engine, clock, workload, timeout_s=0.002, trace=trace)
    assert any(row["kind"] == "arrival" for row in trace)
    assert any(row["kind"] == "token" for row in trace)


@pytest.mark.parametrize("bad", ["index", "finished", "count"])
def test_invalid_event_stream_fails_instead_of_counting_a_throughput_win(api, bad):
    engine, clock = measured_engine(api, "fixed")
    original = engine.step_with_events

    def corrupt():
        result = original()
        if bad == "count":
            result.token_events.clear()
            result.outputs.clear()
        else:
            event = result.token_events[0]
            result.token_events[0] = api.TokenEvent(event.seq_id, 1 if bad == "index" else 0,
                                                   event.token_id, bad != "finished")
        return result

    engine.step_with_events = corrupt
    with pytest.raises(RuntimeError):
        measure(api, engine, clock, [Request("one", "short", 0, [1], 1)])


@pytest.mark.parametrize("workload", [[],
    [Request("one", "short", -1, [1], 1)],
    [Request("one", "short", float("nan"), [1], 1)],
    [Request("one", "short", 0, [], 1)],
    [Request("one", "short", 0, [1], 0)],
    [Request("one", "short", 0, [1], 1), Request("one", "long", 1, [2], 1)],
])
def test_invalid_workload_is_rejected(api, workload):
    engine, clock = measured_engine(api, "fixed")
    with pytest.raises(ValueError):
        measure(api, engine, clock, workload)
