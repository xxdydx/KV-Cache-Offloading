"""Optional phase timing for KV handoff paths.

Enable with the environment variable NANOVLLM_HANDOFF_DEBUG=1 (read at
import time) or by calling enable() from Python. When enabled, every
export/import/detach/resume call emits one JSON line at DEBUG level on the
"nanovllm.kv_handoff" logger with:

- per-phase wall times in milliseconds (cumulative when a phase repeats)
- total wall time and unaccounted time (gaps between phases plus
  instrumentation cost)
- a calibrated estimate of the instrumentation overhead itself, measured
  once per process by timing empty phase records

enable() also attaches a stderr handler at DEBUG level when none exists, so
NANOVLLM_HANDOFF_DEBUG=1 is turnkey. When disabled, recording reduces to a
boolean check and a nullcontext per phase; no logs are emitted and
finish() returns None. Nested timers each emit their own line, so an
import inside resume_request produces two related records.
"""

import json
import logging
import os
import time
from contextlib import nullcontext

logger = logging.getLogger("nanovllm.kv_handoff")

_ENABLED = False
_PHASE_COST_S = None


def enable(enabled: bool = True):
    """Turn handoff phase timing on or off for the rest of the process."""
    global _ENABLED
    _ENABLED = bool(enabled)
    if _ENABLED and logger.level == logging.NOTSET and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


# Environment activation must go through enable() so the stderr handler and
# DEBUG level are attached; a bare module flag alone would swallow records.
enable(os.environ.get("NANOVLLM_HANDOFF_DEBUG", "") not in ("", "0"))


def is_enabled() -> bool:
    return _ENABLED


def _ms(seconds: float) -> float:
    # Six decimals: sub-microsecond CPU phases and the timing overhead itself
    # would vanish at three.
    return round(seconds * 1e3, 6)


def _phase_cost_s() -> float:
    """Calibrate one empty phase record (enter + exit + accumulate)."""
    global _PHASE_COST_S
    if _PHASE_COST_S is None:
        timer = PhaseTimer.__new__(PhaseTimer)
        timer.enabled = True
        timer.phases = {}
        timer.phase_records = 0
        iterations = 2000
        start = time.perf_counter()
        for _ in range(iterations):
            with timer.phase("calibration"):
                pass
        _PHASE_COST_S = max((time.perf_counter() - start) / iterations, 1e-9)
    return _PHASE_COST_S


class _Phase:
    __slots__ = ("_timer", "_name", "_start")

    def __init__(self, timer: "PhaseTimer", name: str):
        self._timer = timer
        self._name = name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._start
        phases = self._timer.phases
        phases[self._name] = phases.get(self._name, 0.0) + elapsed
        self._timer.phase_records += 1
        return False


class PhaseTimer:
    """Record named phases of one handoff operation and log them as JSON.

    Create the timer at the start of an operation, wrap each stage in
    `with timer.phase("name"):`, and call finish(**extra) on success or
    abort(error, **extra) on failure. finish() returns the record dict so
    callers and tests can consume it programmatically. Extra fields may be
    zero-argument callables (e.g. lambda: payload.numel()) which are then
    evaluated only when timing is enabled; cheap constants can be passed
    directly.
    """

    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.enabled = _ENABLED
        self.phases: dict[str, float] = {}
        self.phase_records = 0
        self._start = time.perf_counter() if self.enabled else None

    def phase(self, name: str):
        if not self.enabled:
            return nullcontext()
        return _Phase(self, name)

    def _record(self, **extra) -> dict:
        if self.enabled:
            # Resolve lazy values (zero-arg callables) only when recording, so
            # extra fields cost nothing on the disabled path.
            extra = {key: value() if callable(value) else value
                     for key, value in extra.items()}
        total = time.perf_counter() - self._start
        accounted = sum(self.phases.values())
        record = {
            "op": self.operation,
            **self.context,
            **extra,
            "total_ms": _ms(total),
            "phases_ms": {name: _ms(elapsed) for name, elapsed in self.phases.items()},
            "unaccounted_ms": _ms(total - accounted),
            "timing_overhead_ms": _ms(self.phase_records * _phase_cost_s()),
        }
        logger.debug(json.dumps(record))
        return record

    def finish(self, **extra) -> dict | None:
        if not self.enabled:
            return None
        return self._record(**extra)

    def abort(self, error: BaseException, **extra) -> dict | None:
        if not self.enabled:
            return None
        return self._record(error=f"{type(error).__name__}: {error}", **extra)
