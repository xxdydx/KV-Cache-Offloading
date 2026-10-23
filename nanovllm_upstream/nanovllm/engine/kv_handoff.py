"""Reference, uncompressed KV handoff for a single-GPU worker (TP=1).

The payload contains written tokens only, in logical order, on the CPU. This
deliberately synchronous implementation is the correctness/O0 baseline; packing,
pinned buffers and pipelining can replace its copies independently later.
"""

from dataclasses import dataclass

import torch

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


@dataclass
class KVSnapshot:
    token_ids: list[int]
    num_prompt_tokens: int
    num_cached_tokens: int
    block_size: int
    temperature: float
    max_tokens: int
    ignore_eos: bool
    kv: torch.Tensor  # [K/V, layer, written token, KV head, head dim], CPU

    @property
    def payload_bytes(self):
        return self.kv.numel() * self.kv.element_size()


@torch.inference_mode()
def export_kv(runner, seq: Sequence) -> KVSnapshot:
    """Snapshot after a completed prefill/decode step, before the next decode."""
    if runner.world_size != 1:
        raise NotImplementedError("KV handoff currently requires tensor_parallel_size=1")
    if seq.status != SequenceStatus.RUNNING or seq.num_scheduled_tokens:
        raise ValueError("export requires a running sequence between steps")
    count = seq.num_cached_tokens
    if count <= 0 or count != len(seq) - 1:
        raise ValueError("export requires exactly one pending, uncached token")
    num_blocks = (count + runner.block_size - 1) // runner.block_size
    block_ids = torch.tensor(seq.block_table[:num_blocks], dtype=torch.int64,
                             device=runner.kv_cache.device)
    if block_ids.numel() != num_blocks:
        raise ValueError("sequence is missing KV blocks")
    packed = runner.kv_cache.index_select(2, block_ids).flatten(2, 3)
    payload = packed[:, :, :count].contiguous().cpu()
    return KVSnapshot(seq.token_ids.copy(), seq.num_prompt_tokens, count,
                      runner.block_size, seq.temperature, seq.max_tokens,
                      seq.ignore_eos, payload)


@torch.inference_mode()
def verify_imported_kv(runner, seq: Sequence, snapshot: KVSnapshot) -> bool:
    """Compare exact bytes using at most one block of CPU readback storage.

    Reading the entire imported cache back at once needlessly doubles CPU RAM
    usage for long prompts. A block contains 256 tokens by default.
    """
    for logical, physical in enumerate(seq.block_table):
        start = logical * runner.block_size
        end = min(start + runner.block_size, snapshot.num_cached_tokens)
        actual = runner.kv_cache[:, :, physical, :end-start].cpu()
        expected = snapshot.kv[:, :, start:end]
        if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            return False
        del actual
    return True


@torch.inference_mode()
def import_kv(runner, block_manager, snapshot: KVSnapshot) -> Sequence:
    """Allocate independent physical blocks and reconstruct a decode-ready sequence.

    Both workers must use the same model weights and KV dtype. The caller owns
    admission into its scheduler and releasing the source after acknowledgement.
    """
    if runner.world_size != 1:
        raise NotImplementedError("KV handoff currently requires tensor_parallel_size=1")
    cache = runner.kv_cache
    expected = (2, cache.shape[1], snapshot.num_cached_tokens, *cache.shape[-2:])
    if (snapshot.block_size != runner.block_size or block_manager.block_size != runner.block_size
            or tuple(snapshot.kv.shape) != expected or snapshot.kv.dtype != cache.dtype
            or snapshot.kv.device.type != "cpu"):
        raise ValueError("snapshot layout, dtype or block size does not match this worker")
    if (not 0 < snapshot.num_prompt_tokens < len(snapshot.token_ids)
            or snapshot.num_cached_tokens != len(snapshot.token_ids) - 1
            or len(snapshot.token_ids) - snapshot.num_prompt_tokens >= snapshot.max_tokens):
        raise ValueError("snapshot must contain an unfinished completion and one uncached token")
    seq = Sequence(snapshot.token_ids[:snapshot.num_prompt_tokens], SamplingParams(
        temperature=snapshot.temperature, max_tokens=snapshot.max_tokens,
        ignore_eos=snapshot.ignore_eos))
    for token in snapshot.token_ids[snapshot.num_prompt_tokens:]:
        seq.append_token(token)
    block_manager.allocate_for_resume(seq, snapshot.num_cached_tokens)
    try:
        payload = snapshot.kv.to(device=cache.device)
        for logical, physical in enumerate(seq.block_table):
            start = logical * runner.block_size
            end = min(start + runner.block_size, snapshot.num_cached_tokens)
            cache[:, :, physical, :end-start].copy_(payload[:, :, start:end])
        # Prefix metadata becomes valid only once the payload has been copied.
        seq.num_cached_tokens = 0
        seq.num_scheduled_tokens = snapshot.num_cached_tokens
        block_manager.hash_blocks(seq)
        seq.num_scheduled_tokens = 0
        seq.num_cached_tokens = snapshot.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        return seq
    except Exception:
        block_manager.deallocate(seq)
        raise
