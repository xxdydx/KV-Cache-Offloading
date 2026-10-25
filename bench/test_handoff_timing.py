"""Phase timing for KV handoff: unit coverage and engine-level integration.

Run with the rest of the suite:
    python3 -m pytest -q bench/test_handoff_timing.py
"""

import json
import logging
import sys
from types import SimpleNamespace

import pytest

from bench.test_engine_handoff import cached_engine, prefilled_engine  # noqa: F401
from bench.test_engine_events import api  # noqa: F401


@pytest.fixture
def timing(api):
    """The handoff_timing module loaded by the api fixture, with timing off."""
    module = sys.modules["nanovllm.engine.handoff_timing"]
    module.enable(False)
    yield module
    module.enable(False)


@pytest.fixture
def timing_on(timing):
    timing.enable(True)


@pytest.fixture
def records(caplog):
    caplog.set_level(logging.DEBUG, logger="nanovllm.kv_handoff")
    return caplog


def parsed(record):
    return json.loads(record.message)


def test_disabled_timer_records_nothing(timing):
    timer = timing.PhaseTimer("export_kv", seq_id=1)
    with timer.phase("gather"):
        pass
    assert timer.finish(payload_bytes=10) is None
    assert timer.phases == {}


def test_enabled_timer_logs_structured_record(timing, timing_on, records):
    timer = timing.PhaseTimer("export_kv", seq_id=7)
    with timer.phase("gather"):
        sum(range(1000))
    record = timer.finish(payload_bytes=4096)
    assert record["op"] == "export_kv"
    assert record["seq_id"] == 7
    assert record["payload_bytes"] == 4096
    assert set(record["phases_ms"]) == {"gather"}
    assert record["total_ms"] >= record["phases_ms"]["gather"]
    assert record["unaccounted_ms"] >= 0.0
    assert record["timing_overhead_ms"] >= 0.0
    assert len(records.records) == 1
    assert parsed(records.records[0]) == record


def test_repeated_phase_names_accumulate(timing, timing_on, records):
    timer = timing.PhaseTimer("op")
    for _ in range(3):
        with timer.phase("loop"):
            pass
    record = timer.finish()
    assert set(record["phases_ms"]) == {"loop"}
    assert record["total_ms"] >= record["phases_ms"]["loop"]
    # Three phase records, so the calibrated overhead is charged three times.
    assert record["timing_overhead_ms"] == pytest.approx(
        3 * timing._phase_cost_s() * 1e3, rel=1e-3)


def test_phase_overhead_calibration_is_positive_and_cached(timing):
    first = timing._phase_cost_s()
    second = timing._phase_cost_s()
    assert first > 0
    assert first == second  # computed once per module instance


def test_abort_logs_error_and_returns_record(timing, timing_on, records):
    timer = timing.PhaseTimer("import_kv", num_cached_tokens=5)
    record = timer.abort(ValueError("bad layout"))
    assert record["error"] == "ValueError: bad layout"
    assert record["op"] == "import_kv"
    assert parsed(records.records[0]) == record


def test_lazy_extra_fields_resolve_only_when_enabled(timing, timing_on, records):
    timer = timing.PhaseTimer("import_kv")
    payload = SimpleNamespace(payload_bytes=1234)
    record = timer.finish(payload_bytes=lambda: payload.payload_bytes)
    assert record["payload_bytes"] == 1234

    timing.enable(False)
    timer = timing.PhaseTimer("import_kv")  # disabled: callable must not run
    calls = []
    assert timer.finish(payload_bytes=lambda: calls.append(1)) is None
    assert calls == []


def test_timer_captures_enabled_state_at_construction(timing, timing_on):
    timer = timing.PhaseTimer("export_kv")
    timing.enable(False)
    with timer.phase("gather"):
        pass
    # The enabled flag is snapshotted at construction, so this timer still
    # records and logs even after the global toggle flips off.
    record = timer.finish()
    assert set(record["phases_ms"]) == {"gather"}


def test_detach_and_resume_emit_phase_records(api, timing_on, records):
    source, source_id = prefilled_engine(api, length=5)
    snapshot = source.detach_request(source_id)

    ops = [parsed(record)["op"] for record in records.records]
    assert ops == ["export_kv", "detach_request"]
    export = parsed(records.records[0])
    detach = parsed(records.records[1])
    assert set(export["phases_ms"]) == {"gather_blocks", "pack", "copy_to_host"}
    assert set(detach["phases_ms"]) == {"export", "unschedule"}
    assert export["num_tokens"] == 5
    assert snapshot.payload_bytes == export["payload_bytes"] > 0

    records.records.clear()
    destination = cached_engine(api, [[11], [12]])
    destination.model_runner.kv_cache.fill_(-1)
    destination.resume_request(snapshot)

    ops = [parsed(record)["op"] for record in records.records]
    assert ops == ["import_kv", "resume_request"]
    import_record = parsed(records.records[0])
    resume = parsed(records.records[1])
    assert set(import_record["phases_ms"]) == {
        "validate", "rebuild_sequence", "allocate_blocks",
        "copy_to_device", "scatter_blocks", "publish_cache"}
    # CPU-only test engine: the sync phase runs but contains no CUDA work.
    assert set(resume["phases_ms"]) == {"import", "cuda_sync", "admit"}
    assert resume["seq_id"] == destination.scheduler.running[0].seq_id
    # Nested timers: the import record's total is contained in the resume phase.
    assert import_record["total_ms"] <= resume["phases_ms"]["import"] + 1e-3
