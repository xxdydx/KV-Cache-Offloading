"""vLLM replay driver against a fake AsyncLLM: turn chaining, timing records, and summary shape."""

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from bench.agent_replay import make_agent_workload
from bench.vllm_replay import replay


class FakeEngine:
    """Streams one token per output, all with id 5; reports a fixed cached-token count."""

    def __init__(self):
        self.prompts = []

    async def generate(self, prompt, sampling, request_id):
        self.prompts.append((request_id, list(prompt["prompt_token_ids"])))
        for i in range(sampling):
            await asyncio.sleep(0)
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5])], num_cached_tokens=7,
                                  finished=i == sampling - 1)


@pytest.fixture(autouse=True)
def fake_vllm_inputs(monkeypatch):
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


def test_turns_carry_context_and_outputs_forward():
    agents = make_agent_workload(vocab_size=100, seed=1, num_agents=2, turns_per_agent=3,
                                 initial_prompt=8, tool_result_tokens=2, output_tokens=3, tool_wait_s=0.0)
    engine = FakeEngine()
    result = asyncio.run(replay(engine, agents, lambda count: count))

    prompts = dict(engine.prompts)
    turns = agents["agent-0"]["turns"]
    first = turns[0].tool_result_tokens
    second = first + [5, 5, 5] + turns[1].tool_result_tokens
    assert prompts["agent-0-0"] == first
    assert prompts["agent-0-1"] == second
    assert prompts["agent-0-2"] == second + [5, 5, 5] + turns[2].tool_result_tokens

    records = result["turns"]
    assert [(r["agent"], r["turn"]) for r in records] == [(f"agent-{a}", t) for a in range(2) for t in range(3)]
    assert all(len(r["token_times_s"]) == 3 and r["ttft_s"] <= r["turn_latency_s"] for r in records)
    summary = result["summary"]
    assert summary["completed_trajectories"] == 2
    assert summary["prompt_tokens_total"] == sum(r["prompt_len"] for r in records) == 2 * (8 + 13 + 18)
    assert summary["prompt_tokens_reused"] == 7 * 6
    assert result["steps"] == []


def test_short_output_is_an_error():
    agents = make_agent_workload(vocab_size=100, num_agents=1, turns_per_agent=1, initial_prompt=4,
                                 output_tokens=3, tool_wait_s=0.0)
    with pytest.raises(RuntimeError, match="expected 3"):
        asyncio.run(replay(FakeEngine(), agents, lambda count: count - 1))
