"""Run from the repo root: python3 -m bench.test_metrics"""

import math

from bench import metrics


def steady(n=100, gap=0.01, start=0.0):
    return [start + i * gap for i in range(n)]


def test_ttft_is_first_token_minus_send_time():
    assert metrics.ttft(1.0, [1.5, 1.6]) == 0.5


def test_mean_tpot_of_steady_stream():
    assert math.isclose(metrics.mean_tpot(steady()), 0.01)


def test_inter_token_latencies_are_consecutive_differences():
    gaps = metrics.inter_token_latencies([0.0, 0.1, 0.4])
    assert math.isclose(gaps[0], 0.1) and math.isclose(gaps[1], 0.3)


def test_percentile_interpolates():
    assert metrics.percentile([1, 2, 3, 4], 50) == 2.5
    assert metrics.percentile([5], 99) == 5


def test_mean_tpot_hides_stalls_that_p99_itl_shows():
    # 97 normal gaps and two 0.5 s freezes: a 50 ms SLO on mean TPOT passes,
    # while the p99 inter-token gap shows the freezes.
    times = [0.0]
    for gap in [0.01] * 48 + [0.5] + [0.01] * 49 + [0.5]:
        times.append(times[-1] + gap)
    assert metrics.mean_tpot(times) < 0.05
    assert math.isclose(metrics.p99_itl([times]), 0.5)


def test_pooled_itl_ignores_a_single_stall_among_many_gaps():
    # One 0.5 s gap among ~2000 is under 1% of gaps, so p99 ITL does not see it.
    requests = [steady() for _ in range(20)]
    requests[0][50:] = [t + 0.5 for t in requests[0][50:]]
    assert metrics.p99_itl(requests) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
