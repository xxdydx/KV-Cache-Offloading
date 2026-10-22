"""Sanity checks for the cost model. Run from the repo root: python3 -m costmodel.test_model"""

import math

from costmodel import model


def test_kv_bytes_per_token_matches_spec():
    assert model.kv_bytes_per_token() == 147_456  # ~144 KB


def test_compression_scales_transfer_time():
    base = model.t_net(8192)
    assert math.isclose(model.t_net(8192, compression=2) * 2, base)
    assert math.isclose(model.t_net(8192, compression=4) * 4, base)


def test_8k_uncompressed_transfer_is_about_one_second():
    assert 0.9 < model.t_net(8192) < 1.1


def test_fully_cached_prompt_sends_nothing():
    assert model.t_net(0) == 0


def test_exposed_time_is_zero_when_prefill_covers_transfer():
    # A very fast link makes the transfer shorter than prefill.
    assert model.t_exposed(8192, 8192, "h100_nvl", bw_bytes_per_s=1e12) == 0


def test_exposed_time_never_below_final_chunk():
    assert model.t_exposed(8192, 8192, "h100_nvl", t_chunk=0.01, bw_bytes_per_s=1e12) == 0.01


def test_h100_prefills_faster_than_a100():
    assert model.t_prefill(8192, "h100_nvl") < model.t_prefill(8192, "a100_80gb")


def test_more_compression_never_hurts():
    for gpu in model.GPUS:
        results = [model.should_disaggregate(8192, 8192, gpu, c) for c in (1, 2, 4)]
        assert results == sorted(results)


def test_no_active_decoders_means_nothing_to_protect():
    assert not model.should_disaggregate(8192, 8192, "h100_nvl", compression=4, n_decoding=0)


def test_more_decoders_never_hurts():
    results = [model.should_disaggregate(8192, 8192, "h100_nvl", n_decoding=n) for n in range(0, 20)]
    assert results == sorted(results)


def test_break_even_decoders_is_the_exact_threshold():
    kwargs = dict(prompt_len=8192, l_new=8192, gpu="h100_nvl", compression=2)
    n = model.break_even_decoders(**kwargs)
    assert model.should_disaggregate(**kwargs, n_decoding=n)
    assert not model.should_disaggregate(**kwargs, n_decoding=n - 1)


def test_zero_weight_on_new_request_always_disaggregates():
    assert model.should_disaggregate(8192, 8192, "h100_nvl", n_decoding=1, new_request_weight=0)


def test_unchunked_stall_is_the_full_prefill():
    assert model.t_stall(8192, "h100_nvl") == model.t_prefill(8192, "h100_nvl")


def test_chunking_shrinks_the_longest_pause_but_not_the_total():
    assert model.t_stall(8192, "h100_nvl", prefill_chunk_tokens=512) < model.t_stall(8192, "h100_nvl")
    assert model.t_stall_total(8192, "h100_nvl") == model.t_prefill(8192, "h100_nvl")


def test_chunk_larger_than_prompt_is_the_full_prefill():
    assert model.t_stall(1000, "h100_nvl", prefill_chunk_tokens=4096) == model.t_prefill(1000, "h100_nvl")


def test_tail_objective_with_no_decoders_never_disaggregates():
    assert not model.should_disaggregate(8192, 8192, "h100_nvl", compression=4, n_decoding=0, objective="tail")


def test_chunked_baseline_is_harder_to_beat_under_tail_objective():
    kwargs = dict(prompt_len=8192, l_new=8192, gpu="h100_nvl", compression=2, objective="tail")
    assert model.should_disaggregate(**kwargs)
    assert not model.should_disaggregate(**kwargs, prefill_chunk_tokens=512)


def test_mean_tpot_objective_ignores_chunking():
    kwargs = dict(prompt_len=8192, l_new=8192, gpu="h100_nvl", compression=2, objective="mean_tpot")
    assert model.should_disaggregate(**kwargs) == model.should_disaggregate(**kwargs, prefill_chunk_tokens=512)


def test_mean_tpot_objective_with_no_decoders_never_disaggregates():
    assert not model.should_disaggregate(8192, 8192, "h100_nvl", compression=4, n_decoding=0, objective="mean_tpot")


def test_kv_size_follows_the_model():
    assert model.kv_bytes_per_token(model="llama-3.1-8b") == 131_072
    assert model.kv_bytes_per_token(model="qwen3-0.6b") == 114_688


def test_custom_model_spec_matches_a_registered_one():
    spec = model.ModelSpec(n_layers=36, n_kv_heads=8, d_head=128, n_params=8.2e9)
    assert model.t_prefill(8192, "h100_nvl", spec) == model.t_prefill(8192, "h100_nvl", "qwen3-8b")


def test_unknown_model_name_is_rejected():
    try:
        model.kv_bytes_per_token(model="not-a-model")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_decode_gpu_defaults_to_the_prefill_gpu():
    kwargs = dict(prompt_len=8192, l_new=8192, gpu="h100_nvl", compression=2)
    assert model.should_disaggregate(**kwargs) == model.should_disaggregate(**kwargs, decode_gpu="h100_nvl")


def test_stall_is_charged_to_the_decode_gpu_not_the_prefill_gpu():
    # Prefill on the H100 hides the transfer; the colocated alternative would
    # run on the slower A100, so the stall avoided is larger than on an H100 pair.
    kwargs = dict(prompt_len=8192, l_new=8192, gpu="h100_nvl", compression=2, new_request_weight=2)
    assert not model.should_disaggregate(**kwargs)
    assert model.should_disaggregate(**kwargs, decode_gpu="a100_80gb")


def test_break_even_decoders_uses_the_decode_gpu():
    kwargs = dict(prompt_len=8192, l_new=8192, gpu="h100_nvl", compression=1)
    assert model.break_even_decoders(**kwargs, decode_gpu="a100_80gb") < model.break_even_decoders(**kwargs)


def test_unknown_objective_is_rejected():
    try:
        model.should_disaggregate(8192, 8192, "h100_nvl", objective="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
