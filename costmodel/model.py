"""Analytical cost model for prefill/decode disaggregation. Units: bytes, seconds, FLOPs."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Architecture figures the cost model needs. Add to MODELS or pass one directly."""
    n_layers: int
    n_kv_heads: int
    d_head: int
    n_params: float


# From each model's config.json; parameter counts are approximate.
MODELS = {
    "qwen3-0.6b": ModelSpec(n_layers=28, n_kv_heads=8, d_head=128, n_params=0.6e9),
    "qwen3-8b": ModelSpec(n_layers=36, n_kv_heads=8, d_head=128, n_params=8.2e9),
    "llama-3.1-8b": ModelSpec(n_layers=32, n_kv_heads=8, d_head=128, n_params=8.03e9),
}
DEFAULT_MODEL = "qwen3-8b"

# Approximate vendor peak figures; replace with measured values.
GPUS = {
    "rtx6000_ada": {"bw_bytes_per_s": 0.96e12, "bf16_flops": 360e12},
    "a100_80gb":   {"bw_bytes_per_s": 1.90e12, "bf16_flops": 312e12},
    "h100_nvl":    {"bw_bytes_per_s": 3.90e12, "bf16_flops": 835e12},
}

# Assumed fraction of peak BF16 FLOPS that prefill achieves.
ACHIEVED_FLOPS_FRACTION = 0.6

# Measured TCP bandwidth, H100 -> A100-80 on the SoC cluster (10 GbE).
NET_BW_BYTES_PER_S = 1.23e9


def _spec(model: str | ModelSpec) -> ModelSpec:
    return MODELS[model] if isinstance(model, str) else model


def kv_bytes_per_token(bytes_per_elem: float = 2.0, model: str | ModelSpec = DEFAULT_MODEL) -> float:
    """Bytes of KV cache per token, K and V across all layers."""
    spec = _spec(model)
    return 2 * spec.n_layers * spec.n_kv_heads * spec.d_head * bytes_per_elem


def t_net(l_new: int, compression: float = 1.0, bw_bytes_per_s: float = NET_BW_BYTES_PER_S,
          model: str | ModelSpec = DEFAULT_MODEL) -> float:
    """Seconds to send the KV for the `l_new` tokens the decode side lacks."""
    return l_new * kv_bytes_per_token(model=model) / (compression * bw_bytes_per_s)


def t_prefill(prompt_len: int, gpu: str, model: str | ModelSpec = DEFAULT_MODEL) -> float:
    """Seconds to prefill `prompt_len` tokens; linear term only, ignoring attention's quadratic cost."""
    flops = 2 * _spec(model).n_params * prompt_len
    return flops / (GPUS[gpu]["bf16_flops"] * ACHIEVED_FLOPS_FRACTION)


def t_exposed(l_new: int, prompt_len: int, gpu: str, compression: float = 1.0,
              t_chunk: float = 0.0, bw_bytes_per_s: float = NET_BW_BYTES_PER_S,
              model: str | ModelSpec = DEFAULT_MODEL) -> float:
    """Transfer time not hidden behind prefill on `gpu`, plus the final chunk `t_chunk`."""
    unhidden = t_net(l_new, compression, bw_bytes_per_s, model) - t_prefill(prompt_len, gpu, model)
    return max(0.0, unhidden) + t_chunk


def t_stall(prompt_len: int, gpu: str, prefill_chunk_tokens: int | None = None,
            model: str | ModelSpec = DEFAULT_MODEL) -> float:
    """Longest pause a decoder sees if this prefill runs on its GPU, `gpu`."""
    tokens = prompt_len if prefill_chunk_tokens is None else min(prefill_chunk_tokens, prompt_len)
    return t_prefill(tokens, gpu, model)


def t_stall_total(prompt_len: int, gpu: str, model: str | ModelSpec = DEFAULT_MODEL) -> float:
    """Total delay a decoder accumulates over the whole prefill on `gpu`, chunked or not."""
    return t_prefill(prompt_len, gpu, model)


def should_disaggregate(prompt_len: int, l_new: int, gpu: str, compression: float = 1.0,
                        t_queue: float = 0.0, t_chunk: float = 0.0,
                        bw_bytes_per_s: float = NET_BW_BYTES_PER_S,
                        n_decoding: int = 1, new_request_weight: float = 1.0,
                        prefill_chunk_tokens: int | None = None, objective: str = "total",
                        decoder_remaining_tokens: int = 128, new_output_tokens: int = 256,
                        decode_gpu: str | None = None, model: str | ModelSpec = DEFAULT_MODEL) -> bool:
    """True if the delay avoided for active decoders exceeds the delay added to the new request.

    `gpu` runs the prefill when disaggregating. `decode_gpu` (default `gpu`) is where
    decoders run and where a colocated prefill would execute. `objective` is "total"
    (summed delay), "tail" (longest pause, the p99 inter-token latency) or "mean_tpot"
    (worst per-request increase in mean TPOT).
    """
    stall_gpu = decode_gpu or gpu
    added = new_request_weight * (t_exposed(l_new, prompt_len, gpu, compression, t_chunk,
                                            bw_bytes_per_s, model) + t_queue)
    if objective == "total":
        avoided = n_decoding * t_stall_total(prompt_len, stall_gpu, model)
    elif objective == "tail":
        avoided = t_stall(prompt_len, stall_gpu, prefill_chunk_tokens, model) if n_decoding > 0 else 0.0
    elif objective == "mean_tpot":
        avoided = t_stall_total(prompt_len, stall_gpu, model) / decoder_remaining_tokens if n_decoding > 0 else 0.0
        added /= new_output_tokens
    else:
        raise ValueError(f"unknown objective: {objective!r}")
    return avoided > added


def break_even_decoders(prompt_len: int, l_new: int, gpu: str, compression: float = 1.0,
                        t_queue: float = 0.0, t_chunk: float = 0.0,
                        bw_bytes_per_s: float = NET_BW_BYTES_PER_S,
                        new_request_weight: float = 1.0, decode_gpu: str | None = None,
                        model: str | ModelSpec = DEFAULT_MODEL) -> int:
    """Fewest active decoders for which the "total" objective favours disaggregating."""
    added = new_request_weight * (t_exposed(l_new, prompt_len, gpu, compression, t_chunk,
                                            bw_bytes_per_s, model) + t_queue)
    return math.floor(added / t_stall_total(prompt_len, decode_gpu or gpu, model)) + 1


def break_even_length(gpu: str, compression: float = 1.0, max_len: int = 65536,
                      t_queue: float = 0.0, t_chunk: float = 0.0,
                      bw_bytes_per_s: float = NET_BW_BYTES_PER_S, step: int = 128,
                      **kwargs) -> int | None:
    """First prompt length where `should_disaggregate` holds with no prefix reuse, or None.

    Extra keyword arguments pass through to `should_disaggregate`. Under "tail" with
    chunking this is not a threshold: the rule can stop holding at longer lengths.
    """
    for length in range(step, max_len + 1, step):
        if should_disaggregate(length, length, gpu, compression, t_queue=t_queue, t_chunk=t_chunk,
                               bw_bytes_per_s=bw_bytes_per_s, **kwargs):
            return length
    return None
