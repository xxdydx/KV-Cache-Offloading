import atexit
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


@dataclass(frozen=True)
class TokenEvent:
    """One new output token; seq_id is local to this engine process."""

    seq_id: int
    output_index: int  # Zero-based position within the completion.
    token_id: int
    finished: bool


@dataclass(frozen=True)
class StepResult:
    """Completed outputs, signed work count, and new tokens from one step.

    num_tokens counts scheduled prefill tokens positively and decode sequences
    negatively, matching the legacy step() interface.
    """

    outputs: list[tuple[int, list[int]]]
    num_tokens: int
    token_events: list[TokenEvent]


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        """Enqueue a request and return its process-local sequence ID."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        """Run one batch, preserving the original (outputs, num_tokens) API."""
        result = self._step(emit_events=False)
        return result.outputs, result.num_tokens

    def step_with_events(self) -> StepResult:
        """Run one batch and report each newly generated token exactly once.

        Intermediate prefill chunks emit no tokens. The final prefill chunk
        emits output_index=0 for a fresh request; decode continues that index.
        finished is true on the token that reaches EOS or the output limit.
        """
        return self._step(emit_events=True)

    def _step(self, *, emit_events: bool) -> StepResult:
        seqs, is_prefill = self.scheduler.schedule()
        previous_counts = [seq.num_completion_tokens for seq in seqs] if emit_events else []
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        token_events = [
            TokenEvent(seq.seq_id, seq.num_completion_tokens - 1, seq.last_token, seq.is_finished)
            for seq, previous in zip(seqs, previous_counts)
            if seq.num_completion_tokens > previous
        ]
        return StepResult(outputs, num_tokens, token_events)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
