from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config, validate_scheduling_options
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(frozen=True)
class ScheduleDecision:
    policy: str
    phase: str
    prefill_budget: int
    decode_requests: int  # Runnable decoders before admission/preemption.
    scheduled_tokens: int
    seq_ids: tuple[int, ...]


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.prefill_policy = getattr(config, "prefill_policy", "legacy")
        self.prefill_chunk_size = getattr(config, "prefill_chunk_size", 512)
        self.prefill_min_chunk_size = getattr(config, "prefill_min_chunk_size", 1)
        validate_scheduling_options(self.prefill_policy, self.prefill_chunk_size,
                                    self.prefill_min_chunk_size,
                                    self.max_num_batched_tokens, self.max_num_seqs)
        self.last_schedule: ScheduleDecision | None = None
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Time-slice homogeneous prefill/decode batches; keep runner APIs intact.

        Opt-in policies allow at most one prefill batch between decode batches
        when both phases can progress. This is a work budget, not a latency SLO.
        """
        decode_requests = len(self.running)
        budget = self.max_num_batched_tokens
        if self.prefill_policy != "legacy":
            chunk = self.prefill_chunk_size
            if self.prefill_policy == "decode_aware":
                chunk = max(self.prefill_min_chunk_size, chunk // (1 + decode_requests))
            budget = min(budget, chunk)

        decode_due = (self.prefill_policy != "legacy" and self.running
                      and self.last_schedule is not None
                      and self.last_schedule.phase == "prefill")
        phases = (False, True) if decode_due else (True, False)
        for is_prefill in phases:
            seqs = self._schedule_prefill(budget) if is_prefill else self._schedule_decode()
            if seqs:
                self.last_schedule = ScheduleDecision(
                    self.prefill_policy, "prefill" if is_prefill else "decode", budget,
                    decode_requests, sum(seq.num_scheduled_tokens for seq in seqs),
                    tuple(seq.seq_id for seq in seqs))
                return seqs, is_prefill

        # Decode may preempt every candidate, making a formerly blocked prefill
        # admissible. Retry once rather than returning an empty engine step.
        seqs = self._schedule_prefill(budget)
        if seqs:
            self.last_schedule = ScheduleDecision(
                self.prefill_policy, "prefill", budget, decode_requests,
                sum(seq.num_scheduled_tokens for seq in seqs),
                tuple(seq.seq_id for seq in seqs))
            return seqs, True
        raise RuntimeError("no request can progress: check GPU KV capacity and request lengths")

    def _schedule_prefill(self, budget: int) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # Prefill copies may fail before model execution. Keep all requests
        # and their references retryable, including earlier admissions in a batch.
        prefill_states = []
        removed_waiting = []
        try:
            while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.waiting[0]
                remaining = budget - num_batched_tokens
                if remaining == 0:
                    break
                prefill_states.append((seq, seq.num_cached_tokens, seq.num_scheduled_tokens,
                                       seq.status, bool(seq.block_table)))
                if not seq.block_table:
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        break
                    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                else:
                    num_tokens = seq.num_tokens - seq.num_cached_tokens
                if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                    break
                if not seq.block_table:
                    self.block_manager.allocate(seq, num_cached_blocks)
                seq.num_scheduled_tokens = min(num_tokens, remaining)
                num_batched_tokens += seq.num_scheduled_tokens
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    removed_waiting.append(seq)
                    self.running.append(seq)
                scheduled_seqs.append(seq)

        except BaseException:
            for seq, cached, scheduled, status, had_blocks in reversed(prefill_states):
                if not had_blocks and seq.block_table:
                    self.block_manager.deallocate(seq)
                seq.num_cached_tokens = cached
                seq.num_scheduled_tokens = scheduled
                seq.status = status
            for seq in removed_waiting:
                self.running.remove(seq)
            self.waiting.extendleft(reversed(removed_waiting))
            raise

        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        # Decode growth can also recycle a cached block. Roll back unpublished
        # tail reservations if preservation fails anywhere in this batch.
        decode_states = []
        try:
            while self.running and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.running.popleft()
                while not self.block_manager.can_append(seq):
                    # Interleaving leaves partial prefills resident in waiting.
                    # Reclaim those reservations before evicting live decoders;
                    # otherwise a long partial prompt can prevent all progress.
                    partial = next((item for item in reversed(self.waiting) if item.block_table), None)
                    if self.prefill_policy != "legacy" and partial is not None:
                        self.block_manager.deallocate(partial)
                        continue
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    decode_states.append((seq, len(seq.block_table), seq.num_scheduled_tokens, seq.is_prefill))
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
        except BaseException:
            for seq, old_length, scheduled, is_prefill in reversed(decode_states):
                for block_id in reversed(seq.block_table[old_length:]):
                    self.block_manager._release_block(block_id)
                del seq.block_table[old_length:]
                seq.num_scheduled_tokens = scheduled
                seq.is_prefill = is_prefill
            self.running.extendleft(reversed([seq for seq, *_ in decode_states]))
            raise
        if self.prefill_policy == "legacy":
            self.running.extendleft(reversed(scheduled_seqs))
        else:
            # More resident requests than batch slots must all get decode time.
            self.running.extend(scheduled_seqs)
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
