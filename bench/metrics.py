"""Latency metrics from per-token timestamps.

Timestamps are in seconds and all come from one clock per request (the client's),
so differences between them are valid without cross-machine clock sync.
"""

import math


def ttft(sent: float, token_times: list[float]) -> float:
    """Time to first token."""
    return token_times[0] - sent


def mean_tpot(token_times: list[float]) -> float:
    """Mean time per output token after the first, for one request."""
    if len(token_times) < 2:
        raise ValueError("need at least two tokens")
    return (token_times[-1] - token_times[0]) / (len(token_times) - 1)


def inter_token_latencies(token_times: list[float]) -> list[float]:
    """Gap between each pair of consecutive tokens."""
    return [b - a for a, b in zip(token_times, token_times[1:])]


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (0-100) with linear interpolation."""
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def p99_mean_tpot(requests: list[list[float]]) -> float:
    """p99 across requests of each request's mean TPOT."""
    return percentile([mean_tpot(t) for t in requests if len(t) >= 2], 99)


def p99_itl(requests: list[list[float]]) -> float:
    """p99 over every inter-token gap pooled across all requests."""
    return percentile([gap for t in requests for gap in inter_token_latencies(t)], 99)
