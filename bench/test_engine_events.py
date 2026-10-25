"""CPU checks using the real scheduler and scripted model outputs, without CUDA.

Run: python3 -m pytest -q bench/test_engine_events.py
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1] / "nanovllm_upstream" / "nanovllm"


@pytest.fixture
def api(monkeypatch):
    # Skip the package initializer and CUDA-only ModelRunner import. Load actual
    # engine/scheduler/sequence/block-manager code; restore all modules afterward.
    for name, path in (("nanovllm", ROOT), ("nanovllm.engine", ROOT / "engine")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("nanovllm.sampling_params", "sampling_params.py")
    load("nanovllm.config", "config.py")
    load("nanovllm.engine.sequence", "engine/sequence.py")
    load("nanovllm.engine.block_manager", "engine/block_manager.py")
    load("nanovllm.engine.scheduler", "engine/scheduler.py")
    load("nanovllm.engine.kv_handoff", "engine/kv_handoff.py")
    runner_module = ModuleType("nanovllm.engine.model_runner")
    runner_module.ModelRunner = object
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    return load("nanovllm.engine.llm_engine", "engine/llm_engine.py")


class ScriptedRunner:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = []

    def call(self, method, seqs, is_prefill):
        assert method == "run"
        tokens = next(self.batches)
        assert len(tokens) == len(seqs)
        self.calls.append((is_prefill, [seq.seq_id for seq in seqs]))
        return tokens


def make_engine(api, batches, budget=8, num_blocks=16):
    engine = api.LLMEngine.__new__(api.LLMEngine)
    engine.scheduler = api.Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=budget, eos=99,
        kvcache_block_size=256, num_kvcache_blocks=num_blocks,
    ))
    engine._detached_requests = {}
    engine.model_runner = ScriptedRunner(batches)
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: [int(token) for token in text.split()],
        decode=lambda tokens: " ".join(map(str, tokens)),
    )
    return engine


def params(api, **kwargs):
    return api.SamplingParams(temperature=0, **kwargs)


def test_add_request_returns_distinct_ids_for_text_and_token_prompts(api):
    engine = make_engine(api, [])
    first = engine.add_request("1 2", params(api, max_tokens=2))
    second = engine.add_request([3, 4], params(api, max_tokens=2))
    queued = list(engine.scheduler.waiting)
    assert first != second
    assert [seq.seq_id for seq in queued] == [first, second]
    assert [seq.prompt_token_ids for seq in queued] == [[1, 2], [3, 4]]


def test_chunked_prefill_emits_only_actual_outputs_then_finishes_once(api):
    engine = make_engine(api, [[90], [91], [10], [11]], budget=2)
    request_id = engine.add_request([1, 2, 3, 4, 5], params(api, max_tokens=2))

    for _ in range(2):
        chunk = engine.step_with_events()
        assert chunk.token_events == []
        assert chunk.outputs == []
        assert chunk.num_tokens == 2

    first = engine.step_with_events()
    assert first.token_events == [api.TokenEvent(request_id, 0, 10, False)]
    assert first.outputs == []
    assert first.num_tokens == 1

    last = engine.step_with_events()
    assert last.token_events == [api.TokenEvent(request_id, 1, 11, True)]
    assert last.outputs == [(request_id, [10, 11])]
    assert last.num_tokens == -1
    assert engine.is_finished()
    assert not engine.scheduler.block_manager.used_block_ids
    assert [prefill for prefill, _ in engine.model_runner.calls] == [True, True, True, False]


def test_batched_requests_keep_independent_indices_and_finish_states(api):
    engine = make_engine(api, [[10, 20], [21], [22]])
    first = engine.add_request([1, 2], params(api, max_tokens=1))
    second = engine.add_request([3, 4], params(api, max_tokens=3))

    prefill = engine.step_with_events()
    assert prefill.token_events == [
        api.TokenEvent(first, 0, 10, True),
        api.TokenEvent(second, 0, 20, False),
    ]
    assert prefill.outputs == [(first, [10])]
    assert prefill.num_tokens == 4
    assert engine.step_with_events().token_events == [api.TokenEvent(second, 1, 21, False)]
    final = engine.step_with_events()
    assert final.token_events == [api.TokenEvent(second, 2, 22, True)]
    assert final.outputs == [(second, [20, 21, 22])]
    assert engine.is_finished()


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_eos_event_respects_ignore_eos(api, ignore_eos):
    engine = make_engine(api, [[99], [10]])
    request_id = engine.add_request([1], params(api, max_tokens=2, ignore_eos=ignore_eos))
    first = engine.step_with_events()
    assert first.token_events == [api.TokenEvent(request_id, 0, 99, not ignore_eos)]
    if ignore_eos:
        assert first.outputs == []
        final = engine.step_with_events()
        assert final.token_events == [api.TokenEvent(request_id, 1, 10, True)]
        assert final.outputs == [(request_id, [99, 10])]
    else:
        assert first.outputs == [(request_id, [99])]
    assert engine.is_finished()
    assert not engine.scheduler.block_manager.used_block_ids


def test_step_and_event_api_share_progress_without_replaying_tokens(api):
    engine = make_engine(api, [[10], [11], [12]])
    request_id = engine.add_request([1], params(api, max_tokens=3))
    assert engine.step_with_events().token_events == [api.TokenEvent(request_id, 0, 10, False)]
    assert engine.step() == ([], -1)
    final = engine.step_with_events()
    assert final.token_events == [api.TokenEvent(request_id, 2, 12, True)]
    assert final.outputs == [(request_id, [10, 11, 12])]
    assert len(engine.model_runner.calls) == 3


def test_legacy_step_still_returns_completed_outputs_and_signed_work(api):
    engine = make_engine(api, [[10], [11]])
    request_id = engine.add_request([1, 2], params(api, max_tokens=2))
    assert engine.step() == ([], 2)
    assert engine.step() == ([(request_id, [10, 11])], -1)
    assert engine.is_finished()


def test_generate_preserves_output_order_and_matches_streamed_tokens(api):
    prompts = [[1], [2, 3]]
    sampling = params(api, max_tokens=2)
    engine = make_engine(api, [[10, 20], [11, 21]])
    generated = engine.generate(prompts, sampling, use_tqdm=False)
    assert generated == [
        {"text": "10 11", "token_ids": [10, 11]},
        {"text": "20 21", "token_ids": [20, 21]},
    ]

    streaming = make_engine(api, [[10, 20], [11, 21]])
    request_ids = [streaming.add_request(prompt, sampling) for prompt in prompts]
    tokens = {request_id: [] for request_id in request_ids}
    while not streaming.is_finished():
        for event in streaming.step_with_events().token_events:
            assert event.output_index == len(tokens[event.seq_id])
            tokens[event.seq_id].append(event.token_id)
    assert [tokens[request_id] for request_id in request_ids] == [item["token_ids"] for item in generated]
