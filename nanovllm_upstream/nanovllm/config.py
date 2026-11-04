import os
from dataclasses import dataclass
from transformers import AutoConfig


PREFILL_POLICIES = ("legacy", "fixed", "decode_aware")


def validate_scheduling_options(policy, chunk_size, min_chunk_size, batch_tokens, batch_seqs):
    if policy not in PREFILL_POLICIES:
        raise ValueError(f"prefill_policy must be one of {PREFILL_POLICIES}")
    for name, value in (("prefill_chunk_size", chunk_size),
                        ("prefill_min_chunk_size", min_chunk_size),
                        ("max_num_batched_tokens", batch_tokens),
                        ("max_num_seqs", batch_seqs)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_chunk_size > chunk_size:
        raise ValueError("prefill_min_chunk_size must not exceed prefill_chunk_size")


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    distributed_init_method: str = "tcp://localhost:2333"
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    prefill_policy: str = "legacy"
    prefill_chunk_size: int = 512
    prefill_min_chunk_size: int = 1

    def __post_init__(self):
        validate_scheduling_options(self.prefill_policy, self.prefill_chunk_size,
                                    self.prefill_min_chunk_size,
                                    self.max_num_batched_tokens, self.max_num_seqs)
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
