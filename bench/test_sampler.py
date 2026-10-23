"""Sampler checks without loading the CUDA-only engine package.

Run: python3 -m pytest -q bench/test_sampler.py
The GPU handoff benchmark additionally exercises the default compiled sampler.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "nanovllm_upstream" / "nanovllm"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SamplingParams = load_file("sampler_test_params", ROOT / "sampling_params.py").SamplingParams
Sampler = load_file("sampler_test_impl", ROOT / "layers" / "sampler.py").Sampler


def sample(logits, temperatures):
    return Sampler.forward.__wrapped__(Sampler(), logits, temperatures)


def test_greedy_matches_argmax_independently_of_seed_and_preserves_logits():
    logits = torch.tensor([[1.2, 4.8, 2.1], [9.0, 2.0, 1.0]])
    original = logits.clone()
    for seed in (0, 7, 123):
        torch.manual_seed(seed)
        assert sample(logits, torch.zeros(2)).tolist() == [1, 0]
        assert torch.equal(logits, original)


def test_positive_temperature_retains_existing_sampling_rule():
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    temperatures = torch.tensor([0.7, 1.2])
    torch.manual_seed(123)
    probs = torch.softmax(logits / temperatures[:, None], dim=-1)
    expected = (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)
    torch.manual_seed(123)
    assert torch.equal(sample(logits, temperatures), expected)


def test_mixed_batch_can_be_captured_as_one_graph():
    fn = torch.compile(Sampler.forward.__wrapped__, backend="eager", fullgraph=True)
    logits = torch.tensor([[0.0, 10.0, 1.0], [float("-inf"), 0.0, float("-inf")], [8.0, 1.0, 0.0]])
    temperatures = torch.tensor([0.0, 0.8, 0.0])
    assert fn(Sampler(), logits, temperatures).tolist() == [1, 1, 0]


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
def test_valid_temperatures(temperature):
    assert SamplingParams(temperature=temperature).temperature == temperature


@pytest.mark.parametrize("temperature", [-1.0, 1e-12, float("nan")])
def test_invalid_temperatures(temperature):
    with pytest.raises(AssertionError):
        SamplingParams(temperature=temperature)
