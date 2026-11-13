from collections import deque
from collections.abc import Callable, Sequence as TokenSequence
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


def plan_split(cpu_blocks: int, block_size: int, restore_s_per_block: float,
               recompute_s_per_token: float, recompute_s_per_token2: float) -> int:
    """How many leading CPU-hit blocks to recompute while the rest restore.

    The head of a prefix depends on nothing after it, so recomputing it on
    the GPU can overlap restoring the remainder over PCIe. Picks the split
    that minimizes the slower of the two, using
    recompute(t) = t * recompute_s_per_token + t^2 * recompute_s_per_token2.
    """
    def cost(k):
        tokens = k * block_size
        recompute = tokens * recompute_s_per_token + tokens * tokens * recompute_s_per_token2
        return max(recompute, (cpu_blocks - k) * restore_s_per_block)

    return min(range(cpu_blocks + 1), key=cost)


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        # Last owner release (BlockManager.deallocate call) and position there; for MRU eviction.
        self.release_gen = 0
        self.seq_index = 0

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, eviction: str = "lru"):
        if eviction not in ("lru", "mru"):
            raise ValueError("eviction must be 'lru' or 'mru'")
        self.block_size = block_size
        self.eviction = eviction
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self._cpu_prefix_cache = None
        self._last_prefetch = None
        self._release_gen = 0
        # Idle blocks a blocked request's prefetch is keeping for it (MRU only;
        # LRU moves them to the back of the free list instead).
        self._protected: set[int] = set()
        # Set by the scheduler while the head waiting request cannot be admitted.
        self.admission_blocked = False

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _is_published(self, block: Block) -> bool:
        return block.hash != -1 and self.hash_to_block_id.get(block.hash) == block.block_id

    def _victim(self) -> int:
        if self.eviction == "lru":
            return self.free_block_ids[0]
        # Unpublished blocks first, then the most recently released request's
        # blocks, tail first. With similar tool waits, the agent that finished
        # last returns last, which is the case where LRU evicts exactly the
        # context needed next.
        best = None
        for block_id in self.free_block_ids:
            block = self.blocks[block_id]
            if not self._is_published(block):
                return block_id
            if block_id in self._protected:
                continue
            key = (block.release_gen, block.seq_index)
            if best is None or key > best[0]:
                best = (key, block_id)
        return best[1] if best is not None else self.free_block_ids[0]

    def _allocate_block(self, order_writes: bool = True) -> int:
        # Preserve before removing the free slot or invalidating its metadata.
        # Unexpected copy failures leave this candidate available and indexed.
        block_id = self._victim()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            if self._cpu_prefix_cache is not None:
                self._cpu_prefix_cache._preserve_before_recycle(block.hash)
            del self.hash_to_block_id[block.hash]
        if order_writes and self._cpu_prefix_cache is not None:
            self._cpu_prefix_cache.before_block_reuse(block_id)
        if self.free_block_ids[0] == block_id:
            self.free_block_ids.popleft()
        else:
            self.free_block_ids.remove(block_id)
        self._protected.discard(block_id)
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        # Optionally start saving idle published blocks now, while the
        # copy can overlap later steps, rather than when they are recycled.
        if (self._cpu_prefix_cache is not None and block.hash != -1
                and self.hash_to_block_id.get(block.hash) == block_id):
            self._cpu_prefix_cache._preserve_on_release(block.hash)

    def _acquire_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        if self._cpu_prefix_cache is not None:
            # Also when shared: another request's restore span may still be copying it.
            self._cpu_prefix_cache.before_block_use(block_id)
        block.ref_count += 1

    def _release_block(self, block_id: int):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)

    def restore_cached_block(
        self, prefix_hash: int, token_ids: TokenSequence[int], copy_to_block: Callable[[int], None],
    ) -> int:
        """Reserve, populate, then publish a full recyclable prefix-cache block.

        The callback must finish copying, or order the copy after pending
        copies on its own stream and make later readers wait (as the CPU
        cache's copy stream does). Failure frees the destination without
        publishing its hash; previous idle contents may already be
        overwritten. Call between steps.
        """
        tokens = list(token_ids)
        if (type(prefix_hash) is not int or not 0 <= prefix_hash < 2**64
                or len(tokens) != self.block_size):
            raise ValueError("restore requires a valid prefix hash and one full block")
        existing = self.hash_to_block_id.get(prefix_hash)
        if existing is not None:
            block = self.blocks[existing]
            if block.hash != prefix_hash or block.token_ids != tokens:
                raise ValueError("prefix hash has conflicting token metadata")
            return existing
        if not self.free_block_ids:
            raise MemoryError("not enough free KV blocks to restore a prefix")
        block_id = self._allocate_block(order_writes=False)
        block = self.blocks[block_id]
        try:
            copy_to_block(block_id)
            block.update(prefix_hash, tokens)
            self.hash_to_block_id[prefix_hash] = block_id
        except BaseException:
            if self.hash_to_block_id.get(prefix_hash) == block_id:
                del self.hash_to_block_id[prefix_hash]
            block.hash = -1
            block.token_ids = []
            raise
        finally:
            block.ref_count = 0
            self._deallocate_block(block_id)
        return block_id

    def _full_block_hashes(self, seq: Sequence, count: int):
        """(chained hash, token list, token tuple) for a sequence's leading blocks.

        Tokens are append-only, so results stay valid. Memoizing keeps the
        scheduler's per-step admission retry from re-hashing a long waiting
        prompt (about 0.6 ms per step for a 17K-token prompt).
        """
        memo = getattr(seq, "_block_hash_memo", None)
        if memo is None:
            memo = seq._block_hash_memo = []
        h = memo[-1][0] if memo else -1
        for i in range(len(memo), count):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            memo.append((h, token_ids, tuple(token_ids)))
        return memo

    def prefetch(self, seq: Sequence, reserve: int = 0) -> int:
        """Start background restores of a waiting request's CPU hits; return how many.

        Restored blocks become ordinary idle GPU hits, so admission can skip
        their copies. Uses only free blocks beyond `reserve` (kept for running
        requests' growth) and never recycles this request's own idle GPU hits.
        Requires a CPU cache created with prefetch=True. Call between steps.
        """
        cache = self._cpu_prefix_cache
        if cache is None or not cache.prefetch_enabled:
            return 0
        key = (seq.seq_id, len(self.free_block_ids))
        if key == self._last_prefetch:
            return 0  # Nothing changed since the last attempt.
        self._last_prefetch = key
        prefix = self._prefix_blocks(seq)
        own_idle = [block_id for _, _, block_id, _ in prefix
                    if block_id is not None and self.blocks[block_id].ref_count == 0]
        budget = len(self.free_block_ids) - reserve - len(own_idle)
        if budget <= 0 or all(entry is None for *_, entry in prefix):
            return 0
        # Recycle other idle blocks first: move this request's hits to the back.
        for block_id in own_idle:
            self.free_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)
        self._protected = set(own_idle)
        started = 0
        for h, token_ids, block_id, entry in prefix:
            if started == budget:
                break
            if entry is not None:
                self._protected.add(self.restore_cached_block(
                    h, token_ids, lambda target, e=entry: cache._copy_entry(e, target, wait=False)))
                started += 1
        cache._prefetched_blocks += started
        return started

    def _prefix_blocks(self, seq: Sequence):
        """Contiguous GPU/CPU hits; no copies or pool mutations during lookup."""
        prefix = []
        memo = self._full_block_hashes(seq, seq.num_blocks - 1)
        for i in range(seq.num_blocks - 1):
            h, token_ids, token_tuple = memo[i]
            block_id = self.hash_to_block_id.get(h)
            if block_id is not None:
                if self.blocks[block_id].hash != h or self.blocks[block_id].token_ids != token_ids:
                    break
                prefix.append((h, token_ids, block_id, None))
                continue
            entry = (self._cpu_prefix_cache._entry_for(h, token_tuple)
                     if self._cpu_prefix_cache is not None else None)
            if entry is None:
                break
            prefix.append((h, token_ids, None, entry))
        return prefix

    def can_allocate(self, seq: Sequence) -> int:
        prefix = self._prefix_blocks(seq)
        num_new_blocks = seq.num_blocks - sum(block_id in self.used_block_ids
                                               for _, _, block_id, _ in prefix)
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return len(prefix)

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        prefix = self._prefix_blocks(seq)
        if not 0 <= num_cached_blocks <= len(prefix):
            raise ValueError("requested prefix is no longer available")
        prefix = prefix[:num_cached_blocks]
        needed = seq.num_blocks - sum(block_id in self.used_block_ids
                                     for _, _, block_id, _ in prefix)
        if len(self.free_block_ids) < needed:
            raise MemoryError("not enough free KV blocks to admit the request")
        split = self._plan_split(prefix)
        if split == len(prefix):
            prefix, num_cached_blocks, split = [], 0, 0  # Recomputing everything is fastest.
        self._protected.clear()
        table = [-1] * seq.num_blocks
        owned = []
        try:
            # Protect *all* GPU hits first, including hits after a CPU miss.
            # Neither CPU restores nor suffix allocations can then recycle them.
            for i, (_, _, block_id, _) in enumerate(prefix):
                if block_id is not None:
                    self._acquire_block(block_id)
                    owned.append(block_id)
                    table[i] = block_id
            for i, (h, token_ids, block_id, entry) in enumerate(prefix):
                if block_id is None and i >= split:
                    block_id = self._allocate_block()
                    owned.append(block_id)
                    table[i] = block_id
                    # With a split, the head prefill overlaps these copies;
                    # finish_restore_span orders the first read after them.
                    if split:
                        self._cpu_prefix_cache._copy_entry(entry, block_id, wait=False)
                    else:
                        self._cpu_prefix_cache._copy_entry(entry, block_id)
                    self.blocks[block_id].update(h, token_ids)
                    self.hash_to_block_id[h] = block_id
            for i in [*range(split), *range(num_cached_blocks, seq.num_blocks)]:
                block_id = self._allocate_block()
                owned.append(block_id)
                table[i] = block_id
        except BaseException:
            for block_id in reversed(owned):
                self._release_block(block_id)
            raise
        seq.block_table = table
        if split:
            seq.num_cached_tokens = 0
            seq.restore_span = (split * self.block_size, num_cached_blocks * self.block_size)
        else:
            seq.num_cached_tokens = num_cached_blocks * self.block_size

    def _plan_split(self, prefix) -> int:
        """Leading blocks to recompute instead of restore (0: restore all hits)."""
        cache = self._cpu_prefix_cache
        if cache is None or cache.hybrid is None:
            return 0
        lead = next((i for i, (_, _, block_id, _) in enumerate(prefix) if block_id is not None),
                    len(prefix))
        split = plan_split(lead, self.block_size, *cache.hybrid)
        # Splitting before a GPU hit only helps if something still restores.
        return split if split < lead or lead == len(prefix) else 0

    def finish_restore_span(self, seq: Sequence):
        """Skip a split admission's restored span once its head is prefilled."""
        start, end = seq.restore_span
        assert seq.num_cached_tokens == start
        seq.restore_span = None
        seq.num_cached_tokens = end
        for block_id in seq.block_table[start // self.block_size:end // self.block_size]:
            self._cpu_prefix_cache.before_block_use(block_id)

    def deallocate(self, seq: Sequence):
        self._release_gen += 1
        for i, block_id in enumerate(seq.block_table):
            self.blocks[block_id].release_gen = self._release_gen
            self.blocks[block_id].seq_index = i
        for block_id in reversed(seq.block_table):
            self._release_block(block_id)
        seq.num_cached_tokens = 0
        seq.restore_span = None
        seq.block_table.clear()

    def allocate_for_resume(self, seq: Sequence, num_cached_tokens: int):
        """Reserve fresh blocks for written KV, leaving the pending token to decode.

        At a block boundary, may_append() will allocate the pending token's block
        on the next decode step. Do not allocate that extra block twice here.
        Imported blocks are published to the prefix cache only after their copy.
        """
        if seq.block_table or not 0 < num_cached_tokens == len(seq) - 1:
            raise ValueError("resume needs an empty block table and one uncached token")
        num_blocks = (num_cached_tokens + self.block_size - 1) // self.block_size
        if len(self.free_block_ids) < num_blocks:
            raise MemoryError("not enough free KV blocks to resume")
        try:
            for _ in range(num_blocks):
                seq.block_table.append(self._allocate_block())
        except BaseException:
            self.deallocate(seq)
            raise
        seq.num_cached_tokens = num_cached_tokens

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
