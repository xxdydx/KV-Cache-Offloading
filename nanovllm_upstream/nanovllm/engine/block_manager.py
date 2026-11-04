from collections import deque
from collections.abc import Callable, Sequence as TokenSequence
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self._cpu_prefix_cache = None

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        # Preserve before removing the free slot or invalidating its metadata.
        # Unexpected copy failures leave this candidate available and indexed.
        block_id = self.free_block_ids[0]
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            if self._cpu_prefix_cache is not None:
                self._cpu_prefix_cache._preserve_before_recycle(block.hash)
            del self.hash_to_block_id[block.hash]
        self.free_block_ids.popleft()
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _acquire_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
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

        The callback must finish copying and establishing readiness before
        returning. Failure frees the destination without publishing its hash;
        previous idle contents may already be overwritten. Call between steps.
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
        block_id = self._allocate_block()
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

    def _prefix_blocks(self, seq: Sequence):
        """Contiguous GPU/CPU hits; no copies or pool mutations during lookup."""
        h = -1
        prefix = []
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h)
            if block_id is not None:
                if self.blocks[block_id].hash != h or self.blocks[block_id].token_ids != token_ids:
                    break
                prefix.append((h, token_ids, block_id, None))
                continue
            entry = (self._cpu_prefix_cache._entry_for(h, token_ids)
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
                if block_id is None:
                    block_id = self._allocate_block()
                    owned.append(block_id)
                    table[i] = block_id
                    self._cpu_prefix_cache._copy_entry(entry, block_id)
                    self.blocks[block_id].update(h, token_ids)
                    self.hash_to_block_id[h] = block_id
            for i in range(num_cached_blocks, seq.num_blocks):
                block_id = self._allocate_block()
                owned.append(block_id)
                table[i] = block_id
        except BaseException:
            for block_id in reversed(owned):
                self._release_block(block_id)
            raise
        seq.block_table = table
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._release_block(block_id)
        seq.num_cached_tokens = 0
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
