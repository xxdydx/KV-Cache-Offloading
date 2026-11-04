"""Synchronous CPU storage with optional automatic prefix-cache preservation."""

from dataclasses import dataclass
from math import prod

import torch

from nanovllm.engine.block_manager import BlockManager


@dataclass(frozen=True)
class _CPUBlock:
    token_ids: tuple[int, ...]
    kv: torch.Tensor  # [K/V, layer, token, KV head, head dim], owned CPU storage


class CPUCacheFullError(MemoryError):
    """The configured payload budget cannot admit another block."""


class CPUBlockCache:
    """One engine's bounded payload store, with optional automatic integration.

    Call from the engine-owning thread between inference steps. Keys are the
    engine's chained prefix hashes, not physical block IDs. The byte budget
    covers tensor payloads, excluding Python metadata. Restored GPU entries are
    recyclable, not pinned; the CPU copy survives until discard/clear/close.
    """

    def __init__(self, runner, block_manager: BlockManager, max_bytes: int, *, automatic: bool = False):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if runner.world_size != 1:
            raise NotImplementedError("CPU prefix blocks currently require TP=1")
        cache = runner.kv_cache
        if (cache.ndim != 6 or cache.shape[0] != 2 or min(cache.shape) <= 0
                or cache.shape[2] != len(block_manager.blocks)
                or cache.shape[3] != block_manager.block_size
                or runner.block_size != block_manager.block_size):
            raise ValueError("KV pool layout does not match the block manager")
        if cache.device.type not in ("cpu", "cuda"):
            raise ValueError("KV pool must be on CPU or CUDA")
        self._cache = cache
        self._manager = block_manager
        self._shape = (2, cache.shape[1], *cache.shape[3:])
        self._block_bytes = prod(self._shape) * cache.element_size()
        self._max_bytes = max_bytes
        self._entries: dict[int, _CPUBlock] = {}
        self._saved_blocks = 0
        self._restored_blocks = 0
        self._budget_skips = 0
        if automatic:
            if block_manager._cpu_prefix_cache is not None:
                raise RuntimeError("block manager already has an automatic CPU cache")
            block_manager._cpu_prefix_cache = self

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def block_bytes(self) -> int:
        return self._block_bytes

    @property
    def bytes_used(self) -> int:
        return len(self._entries) * self._block_bytes

    @property
    def closed(self) -> bool:
        return self._cache is None

    @property
    def automatic(self) -> bool:
        return not self.closed and self._manager._cpu_prefix_cache is self

    @property
    def stats(self) -> dict[str, int]:
        return {"saved_blocks": self._saved_blocks, "restored_blocks": self._restored_blocks,
                "budget_skips": self._budget_skips}

    def __len__(self):
        return len(self._entries)

    def __contains__(self, prefix_hash: int):
        return prefix_hash in self._entries

    def _require_open(self):
        if self.closed:
            raise RuntimeError("CPU prefix cache is closed")

    @torch.inference_mode()
    def save_block(self, prefix_hash: int) -> bool:
        """Save a published full block; return False if already saved.

        A full budget raises before allocation. Copy/allocation failure leaves
        CPU entries and GPU metadata unchanged. Partial tails cannot be saved.
        """
        self._require_open()
        entry = self._entries.get(prefix_hash)
        block_id = self._manager.hash_to_block_id.get(prefix_hash)
        if block_id is None:
            if entry is not None:
                return False
            raise KeyError(f"GPU prefix block {prefix_hash} is not cached")
        block = self._manager.blocks[block_id]
        tokens = tuple(block.token_ids)
        if block.hash != prefix_hash or len(tokens) != self._manager.block_size:
            raise ValueError("source must be a published full prefix block")
        if entry is not None:
            if entry.token_ids != tokens:
                raise ValueError("prefix hash has conflicting token metadata")
            return False
        if self.bytes_used + self.block_bytes > self.max_bytes:
            raise CPUCacheFullError("CPU prefix cache payload budget exceeded")
        # Own contiguous storage even on CPU: never save a view into the pool.
        payload = torch.empty(self._shape, dtype=self._cache.dtype, device="cpu")
        if self._cache.device.type == "cuda":
            torch.cuda.synchronize(self._cache.device)
        payload.copy_(self._cache[:, :, block_id], non_blocking=False)
        self._entries[prefix_hash] = _CPUBlock(tokens, payload)
        self._saved_blocks += 1
        return True

    def _preserve_before_recycle(self, prefix_hash: int):
        # Only a configured-budget miss is an expected, safe fallback. Device
        # errors and unexpected allocation/copy failures must remain visible.
        try:
            self.save_block(prefix_hash)
        except CPUCacheFullError:
            self._budget_skips += 1

    def _entry_for(self, prefix_hash: int, token_ids):
        """Pure lookup used while planning admission; hold entries during copies."""
        entry = self._entries.get(prefix_hash)
        return entry if entry is not None and entry.token_ids == tuple(token_ids) else None

    @torch.inference_mode()
    def _copy_entry(self, entry: _CPUBlock, block_id: int):
        self._require_open()
        self._cache[:, :, block_id].copy_(entry.kv, non_blocking=False)
        if self._cache.device.type == "cuda":
            torch.cuda.synchronize(self._cache.device)
        self._restored_blocks += 1

    @torch.inference_mode()
    def restore_block(self, prefix_hash: int) -> int:
        """Restore a CPU hit; return the GPU cache's physical block ID.

        GPU hits are reused without copying. On copy failure the destination is
        freed and the new hash stays unpublished; any recycled idle victim is
        already evicted. The CPU entry remains available for retry.
        """
        self._require_open()
        entry = self._entries[prefix_hash]

        def copy_to_block(block_id):
            self._copy_entry(entry, block_id)

        return self._manager.restore_cached_block(prefix_hash, entry.token_ids, copy_to_block)

    def discard(self, prefix_hash: int) -> bool:
        """Release one CPU payload without changing GPU cache state."""
        self._require_open()
        return self._entries.pop(prefix_hash, None) is not None

    def clear(self):
        self._entries.clear()

    def close(self):
        """Release payloads and pool references; idempotent."""
        if self.automatic:
            self._manager._cpu_prefix_cache = None
        self.clear()
        self._cache = None
        self._manager = None
