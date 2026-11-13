from functools import lru_cache
import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        yarn: tuple[float, int] | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        mscale = 1.0
        if yarn is not None:
            inv_freq, mscale = yarn_inv_freq(inv_freq, rotary_dim, base, *yarn)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def yarn_inv_freq(inv_freq, rotary_dim, base, factor, original_max_position,
                  beta_fast=32, beta_slow=1):
    """YaRN frequencies and attention scale (as transformers' "yarn" rope_type).

    High-frequency dimensions keep their original frequency, low-frequency
    ones are interpolated by `factor`, with a linear ramp between.
    """
    def correction_dim(rotations):
        return rotary_dim * math.log(original_max_position / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), rotary_dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(rotary_dim // 2, dtype=torch.float) - low) / (high - low)).clamp(0, 1)
    keep = 1 - ramp  # 1: extrapolate (original frequency), 0: interpolate.
    inv_freq = inv_freq / factor * (1 - keep) + inv_freq * keep
    mscale = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return inv_freq, mscale


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    yarn: tuple[float, int] | None = None,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, yarn)
    return rotary_emb
