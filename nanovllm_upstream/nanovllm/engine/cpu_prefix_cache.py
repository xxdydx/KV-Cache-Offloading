"""Bounded CPU storage with optional automatic prefix-cache preservation."""

from dataclasses import dataclass
from math import prod

import torch

from nanovllm.engine.block_manager import BlockManager


# Pinned memory is allocated in chunks of about this size, so one host
# allocation (and its page-locking cost) covers many blocks.
_SLAB_CHUNK_BYTES = 1 << 30


# Largest finite float8_e4m3fn magnitude.
_FP8_MAX = 448.0


def fp8_quantize(source: torch.Tensor, payload: torch.Tensor, scale: torch.Tensor):
    """float8_e4m3fn payload with one scale (scale's dtype) per vector along the last dim."""
    # Divide by the stored (rounded) scale so dequantization inverts it exactly.
    scale.copy_((source.abs().amax(-1, keepdim=True).float() / _FP8_MAX).clamp_min(1e-30))
    payload.copy_((source.float() / scale.float()).clamp_(-_FP8_MAX, _FP8_MAX))


def fp8_dequantize(payload: torch.Tensor, scale: torch.Tensor, target: torch.Tensor):
    torch.mul(payload.to(target.dtype), scale, out=target)


@dataclass(frozen=True)
class _CPUBlock:
    token_ids: tuple[int, ...]
    kv: torch.Tensor  # [K/V, layer, token, KV head, head dim], a slab slot view
    slot: int
    scale: torch.Tensor | None = None  # FP8 only: [K/V, layer, token, KV head, 1]
    saved: torch.cuda.Event | None = None  # CUDA only: the save into `slot` finished


class CPUCacheFullError(MemoryError):
    """The configured payload budget cannot admit another block."""


class CPUBlockCache:
    """One engine's bounded payload store, with optional automatic integration.

    Call from the engine-owning thread between inference steps. Keys are the
    engine's chained prefix hashes, not physical block IDs. The byte budget
    covers tensor payloads, excluding Python metadata. Restored GPU entries are
    recyclable, not pinned; the CPU copy survives until discard/clear/close.

    On CUDA, payloads live in reusable pinned slab slots and copies run on a
    dedicated stream (split_streams=True: one per direction, so saves and
    restores share full-duplex PCIe instead of queueing behind each other). CUDA events order them against compute: a block is not
    overwritten before its pending save finishes, and work issued on the
    current stream after a restore waits for it. Call synchronize() before
    reading payloads on the host. With save_on_release=True, the automatic
    cache also saves a published block as soon as its last owner releases it
    while admission is blocked, so the copy overlaps later steps instead of
    delaying the recycling allocation. With prefetch=True, the scheduler restores a blocked waiting
    request's CPU hits into spare GPU blocks in the background; readers wait
    for such a restore only when they first take the block.

    fp8=True stores payloads as float8_e4m3fn with one bfloat16 scale per
    token and KV head, halving PCIe traffic. It is lossy: restored KV differs
    from the original. Quantization runs on the copy streams.
    hybrid=(restore_s_per_block, recompute_s_per_token, recompute_s_per_token2)
    lets admission recompute the head of a CPU-only prefix while restoring
    the rest (see block_manager.plan_split).
    """

    def __init__(self, runner, block_manager: BlockManager, max_bytes: int, *,
                 automatic: bool = False, save_on_release: bool = False, preallocate: bool = False,
                 prefetch: bool = False, split_streams: bool = False, fp8: bool = False,
                 hybrid: tuple[float, float, float] | None = None):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if runner.world_size != 1:
            raise NotImplementedError("CPU prefix blocks currently require TP=1")
        if (save_on_release or prefetch or hybrid) and not automatic:
            raise ValueError("save_on_release, prefetch and hybrid require automatic=True")
        if hybrid is not None and (len(hybrid) != 3 or min(hybrid) < 0):
            raise ValueError("hybrid needs three nonnegative cost coefficients")
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
        self._fp8 = fp8
        self._dtype = torch.float8_e4m3fn if fp8 else cache.dtype
        self._scale_shape = (*self._shape[:-1], 1)
        self._block_bytes = prod(self._shape) * self._dtype.itemsize
        if fp8:
            self._block_bytes += prod(self._scale_shape) * cache.element_size()
        self._max_bytes = max_bytes
        self._entries: dict[int, _CPUBlock] = {}
        self._capacity = max_bytes // self._block_bytes
        self._slabs: list[torch.Tensor] = []
        self._scale_slabs: list[torch.Tensor] = []
        self._free_slots: list[int] = []
        self._num_slots = 0
        # Pinned slots enable true async DMA; a CPU-simulated pool (CPU-only
        # tests) has no CUDA runtime, so it uses plain synchronous copies.
        self._on_cuda = cache.device.type == "cuda"
        self._copy_stream = torch.cuda.Stream(cache.device) if self._on_cuda else None
        self._restore_stream = (torch.cuda.Stream(cache.device) if self._on_cuda and split_streams
                                else self._copy_stream)
        self._slot_reads: dict[int, torch.cuda.Event] = {}  # slot -> last restore read it
        self._staging: dict[torch.cuda.Stream, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hybrid = hybrid
        self._pending_saves: dict[int, torch.cuda.Event] = {}  # GPU block ID -> save done
        self._pending_restores: dict[int, torch.cuda.Event] = {}  # GPU block ID -> prefetch done
        self._save_on_release = save_on_release
        self._prefetch = prefetch
        self._prefetched_blocks = 0
        self._saved_blocks = 0
        self._restored_blocks = 0
        self._budget_skips = 0
        if automatic:
            if block_manager._cpu_prefix_cache is not None:
                raise RuntimeError("block manager already has an automatic CPU cache")
            block_manager._cpu_prefix_cache = self
        if preallocate:
            while self._num_slots < self._capacity:
                self._grow()

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
    def reserved_bytes(self) -> int:
        return self._num_slots * self._block_bytes

    @property
    def closed(self) -> bool:
        return self._cache is None

    @property
    def automatic(self) -> bool:
        return not self.closed and self._manager._cpu_prefix_cache is self

    @property
    def prefetch_enabled(self) -> bool:
        return self._prefetch

    @property
    def hybrid(self) -> tuple[float, float, float] | None:
        return self._hybrid

    @hybrid.setter
    def hybrid(self, costs: tuple[float, float, float] | None):
        """Set split-admission costs, e.g. after measure_restore_s_per_block()."""
        if costs is not None:
            if not self.automatic:
                raise ValueError("hybrid requires an automatic cache")
            if len(costs) != 3 or min(costs) < 0:
                raise ValueError("hybrid needs three nonnegative cost coefficients")
            costs = tuple(float(c) for c in costs)
        self._hybrid = costs

    @property
    def fp8(self) -> bool:
        return self._fp8

    @property
    def prefetched_blocks(self) -> int:
        return self._prefetched_blocks

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

    def _grow(self):
        count = min(max(1, _SLAB_CHUNK_BYTES // self._block_bytes), self._capacity - self._num_slots)
        slab = torch.empty((count, *self._shape), dtype=self._dtype, device="cpu",
                           pin_memory=self._on_cuda)
        self._slabs.append(slab)
        if self._fp8:
            self._scale_slabs.append(torch.empty((count, *self._scale_shape), dtype=self._cache.dtype,
                                                 device="cpu", pin_memory=self._on_cuda))
        self._free_slots.extend(range(self._num_slots + count - 1, self._num_slots - 1, -1))
        self._num_slots += count

    def _slot_view(self, slot: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        for i, slab in enumerate(self._slabs):
            if slot < len(slab):
                return slab[slot], self._scale_slabs[i][slot] if self._fp8 else None
            slot -= len(slab)
        raise IndexError(slot)

    def _stage(self, stream) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-stream contiguous GPU buffers for FP8 payloads; stream order protects reuse."""
        if stream not in self._staging:
            self._staging[stream] = (
                torch.empty(self._shape, dtype=self._dtype, device=self._cache.device),
                torch.empty(self._scale_shape, dtype=self._cache.dtype, device=self._cache.device))
        return self._staging[stream]

    def _quantize(self, source: torch.Tensor, payload: torch.Tensor, scale: torch.Tensor):
        """Write FP8 `source` into payload/scale (any devices; issue on the copy stream)."""
        if source.is_cuda and payload.device.type == "cpu":
            staged, staged_scale = self._stage(torch.cuda.current_stream(source.device))
        else:
            staged, staged_scale = payload, scale
        fp8_quantize(source, staged, staged_scale)
        if staged is not payload:
            payload.copy_(staged, non_blocking=True)
            scale.copy_(staged_scale, non_blocking=True)

    def _dequantize(self, entry: _CPUBlock, target: torch.Tensor):
        if target.is_cuda:
            staged, staged_scale = self._stage(torch.cuda.current_stream(target.device))
            staged.copy_(entry.kv, non_blocking=True)
            staged_scale.copy_(entry.scale, non_blocking=True)
        else:
            staged, staged_scale = entry.kv, entry.scale
        fp8_dequantize(staged, staged_scale, target)

    @torch.inference_mode()
    def measure_restore_s_per_block(self, repeats: int = 8) -> float:
        """Time this cache's restore path (H2D copy, plus FP8 dequantization) on scratch buffers."""
        self._require_open()
        if not self._on_cuda:
            raise RuntimeError("restore timing needs a CUDA KV pool")
        payload = torch.zeros((repeats, *self._shape), dtype=self._dtype, pin_memory=True)
        scale = (torch.ones((repeats, *self._scale_shape), dtype=self._cache.dtype, pin_memory=True)
                 if self._fp8 else None)
        target = torch.empty(self._shape, dtype=self._cache.dtype, device=self._cache.device)
        stream = self._restore_stream
        stream.wait_stream(torch.cuda.current_stream(self._cache.device))
        for _ in range(2):  # The first pass warms up kernels and clocks.
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                for i in range(repeats):
                    if self._fp8:
                        self._dequantize(_CPUBlock((), payload[i], -1, scale[i]), target)
                    else:
                        target.copy_(payload[i], non_blocking=True)
                end.record(stream)
            end.synchronize()
        return start.elapsed_time(end) / 1000 / repeats

    def synchronize(self):
        """Wait for every pending host<->device copy (needed before host reads)."""
        for stream in {self._copy_stream, self._restore_stream} - {None}:
            stream.synchronize()

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
        if not self._free_slots:
            self._grow()
        slot = self._free_slots.pop()
        done = None
        try:
            payload, scale = self._slot_view(slot)
            source = self._cache[:, :, block_id]
            if self._on_cuda:
                # Start after the kernels that wrote this block; finish before
                # anything overwrites it (see before_block_reuse). With split
                # streams, also after earlier restores that read this slot.
                current = torch.cuda.current_stream(self._cache.device)
                self._copy_stream.wait_stream(current)
                read = self._slot_reads.pop(slot, None)
                if read is not None and self._restore_stream is not self._copy_stream:
                    self._copy_stream.wait_event(read)
                with torch.cuda.stream(self._copy_stream):
                    if self._fp8:
                        self._quantize(source, payload, scale)
                    else:
                        payload.copy_(source, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(self._copy_stream)
                self._pending_saves[block_id] = done
            elif self._fp8:
                self._quantize(source, payload, scale)
            else:
                payload.copy_(source)
        except BaseException:
            self._free_slots.append(slot)
            raise
        self._entries[prefix_hash] = _CPUBlock(tokens, payload, slot, scale, done)
        self._saved_blocks += 1
        return True

    def _preserve(self, prefix_hash: int):
        # Only a configured-budget miss is an expected, safe fallback. Device
        # errors and unexpected allocation/copy failures must remain visible.
        try:
            self.save_block(prefix_hash)
        except CPUCacheFullError:
            self._budget_skips += 1

    def _preserve_before_recycle(self, prefix_hash: int):
        self._preserve(prefix_hash)

    def _preserve_on_release(self, prefix_hash: int):
        # Only when someone is waiting for blocks: otherwise the block will
        # likely stay a GPU hit, and the D2H traffic delays small per-step
        # copies (sampled tokens) queued behind it.
        if self._save_on_release and self._manager.admission_blocked:
            self._preserve(prefix_hash)

    def before_block_reuse(self, block_id: int):
        """Order later writes to a GPU block after its pending copies, without a host sync."""
        for pending in (self._pending_saves, self._pending_restores):
            done = pending.pop(block_id, None)
            if done is not None:
                torch.cuda.current_stream(self._cache.device).wait_event(done)

    def before_block_use(self, block_id: int):
        """Order later reads of a prefetched GPU block after its restore."""
        done = self._pending_restores.pop(block_id, None)
        if done is not None:
            torch.cuda.current_stream(self._cache.device).wait_event(done)

    def _entry_for(self, prefix_hash: int, token_ids):
        """Pure lookup used while planning admission; hold entries during copies."""
        entry = self._entries.get(prefix_hash)
        return entry if entry is not None and entry.token_ids == tuple(token_ids) else None

    @torch.inference_mode()
    def _copy_entry(self, entry: _CPUBlock, block_id: int, wait: bool = True):
        self._require_open()
        target = self._cache[:, :, block_id]
        if self._on_cuda:
            # The destination may still be read by queued kernels, and the
            # payload may still be arriving from its own save.
            current = torch.cuda.current_stream(self._cache.device)
            stream = self._restore_stream
            stream.wait_stream(current)
            if entry.saved is not None and stream is not self._copy_stream:
                stream.wait_event(entry.saved)
            with torch.cuda.stream(stream):
                if self._fp8:
                    self._dequantize(entry, target)
                else:
                    target.copy_(entry.kv, non_blocking=True)
                done = torch.cuda.Event()
                done.record(stream)
            self._slot_reads[entry.slot] = done
            if wait:
                current.wait_event(done)
            else:
                self._pending_restores[block_id] = done
        elif self._fp8:
            self._dequantize(entry, target)
        else:
            target.copy_(entry.kv)
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
        entry = self._entries.pop(prefix_hash, None)
        if entry is None:
            return False
        # A later save into this slot is ordered after pending copies on the copy stream.
        self._free_slots.append(entry.slot)
        return True

    def clear(self):
        for entry in self._entries.values():
            self._free_slots.append(entry.slot)
        self._entries.clear()

    def close(self):
        """Release payloads and pool references; idempotent."""
        if self.closed:
            return
        if self.automatic:
            self._manager._cpu_prefix_cache = None
        self.synchronize()
        self.clear()
        self._pending_saves.clear()
        self._pending_restores.clear()
        self._slot_reads.clear()
        self._staging.clear()
        self._slabs.clear()
        self._scale_slabs.clear()
        self._free_slots.clear()
        self._num_slots = 0
        self._cache = None
        self._manager = None
