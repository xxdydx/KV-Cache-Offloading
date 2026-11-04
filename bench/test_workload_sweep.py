"""Verify memory estimates and context limits without treating them as timing."""

import pytest
import sys

from bench.workload_sweep import plan_workloads, run_cell


CONFIG = {"model_type": "qwen3", "num_hidden_layers": 28, "num_key_value_heads": 8,
          "num_attention_heads": 16, "hidden_size": 1024, "head_dim": 128,
          "max_position_embeddings": 40960}


def test_qwen3_explicit_head_dimension_and_gqa_determine_cache_size():
    cell = plan_workloads(CONFIG, [16], [8192], 128)["cells"][0]
    assert cell["kv_bytes_per_token_per_request"] == 112 * 1024
    assert cell["prompt_kv_bytes"] == 14 * 1024**3
    assert cell["required_kv_blocks"] == 16 * 33
    assert cell["peak_block_kv_bytes"] >= cell["peak_valid_kv_bytes"]
    assert cell["peak_valid_kv_bytes"] > cell["prompt_kv_bytes"]


def test_output_growth_can_exceed_context_limit_even_when_prompt_fits():
    cell = plan_workloads(CONFIG, [1], [40900], 128)["cells"][0]
    assert not cell["within_model_context"]


def test_boundary_final_generated_token_is_not_yet_cached():
    config = {**CONFIG, "max_position_embeddings": 512}
    cell = plan_workloads(config, [1], [256], 1)["cells"][0]
    assert cell["required_kv_blocks"] == 1
    assert cell["peak_valid_kv_bytes"] == cell["prompt_kv_bytes"]


def test_nonpositive_workloads_rejected():
    with pytest.raises(ValueError, match="positive"):
        plan_workloads(CONFIG, [0], [512], 128)


def test_execution_preserves_failure_code_and_diagnostics(tmp_path):
    log = tmp_path / "cell.log"
    result = run_cell([sys.executable, "-c", "print('dependency unavailable'); raise SystemExit(3)"], log, 5)
    assert result == {"exit_code": 3, "timed_out": False}
    assert "dependency unavailable" in log.read_text()


def test_execution_timeout_does_not_leave_trial_running(tmp_path):
    result = run_cell([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "cell.log", 0.1)
    assert result == {"exit_code": None, "timed_out": True}
