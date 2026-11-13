"""Paired-delta summary over synthetic agent-replay run directories."""

import json

from bench.summarize_agent_replay import load, markdown, summarize


def write_run(root, name, traj, ttft=100.0, outputs=2):
    run = root / name
    run.mkdir()
    steps = [{"phase": "prefill", "start_s": 0, "end_s": 1}, {"phase": "decode", "start_s": 1, "end_s": 3}]
    summary = {"wall_s": 60 / traj, "completed_trajectories_per_min": traj,
               "turn_ttft_ms": {"p50": ttft, "p95": 2 * ttft},
               "prompt_tokens_reused": 1, "prompt_tokens_total": 4}
    turns = [{"token_times_s": [0.0] * outputs}]
    (run / "result.json").write_text(json.dumps({"summary": summary, "steps": steps, "turns": turns}))


def test_pairs_by_seed_and_pools_merged_configs(tmp_path):
    for seed, (gpu, cpu) in {7: (10, 11), 11: (10, 12)}.items():
        write_run(tmp_path, f"ctx-gpu_only-s{seed}-r0", gpu)
        write_run(tmp_path, f"ctx-cpu_full-s{seed}-r0", cpu, ttft=50)
    write_run(tmp_path, "ctx-rep-gpu_only-s7-r0", 10)
    write_run(tmp_path, "ctx-rep-cpu_full-s7-r0", 9, ttft=50, outputs=1)
    (tmp_path / "unrelated").mkdir()
    table = summarize(load(tmp_path), ["ctx", "ctx-rep"])
    row = next(r for r in table if r["policy"] == "cpu_full")
    assert row["config"] == "ctx" and row["n"] == row["paired_n"] == 3
    assert abs(row["traj_delta_pct"] - (10 + 20 - 10) / 3) < 1e-9
    assert row["wins"] == 2 and row["traj_delta_ci95"] > 0
    assert row["ttft_p50_delta_pct"] == -50
    assert [r["policy"] for r in table] == ["gpu_only", "cpu_full"]
    assert row["work_mismatches"] == 1
    assert "1 unequal work" in markdown(table)


def test_attached_ablation_pairs_with_target_control(tmp_path):
    write_run(tmp_path, "ctx-gpu_only-s7-r0", 10)
    write_run(tmp_path, "ablate-cpu_release-s7-r0", 12)
    write_run(tmp_path, "lonely-cpu_full-s7-r0", 12)
    table = summarize(load(tmp_path), attach={"ablate": "ctx"})
    row = next(r for r in table if r["policy"] == "cpu_release")
    assert row["config"] == "ctx" and abs(row["traj_delta_pct"] - 20) < 1e-9
    lonely = next(r for r in table if r["config"] == "lonely")
    assert "traj_delta_pct" not in lonely and "`cpu_full`" in markdown(table)


def test_multi_word_policies_and_custom_baseline(tmp_path):
    write_run(tmp_path, "ctx-gpu_only-s7-r0", 10)
    write_run(tmp_path, "ctx-gpu_only_mru-s7-r0", 11)
    write_run(tmp_path, "ctx-cpu_full-s7-r0", 12)
    write_run(tmp_path, "ctx-cpu_exact_hint-s7-r0", 15)
    table = summarize(load(tmp_path), baseline="cpu_full")
    assert [r["policy"] for r in table][0] == "cpu_full"
    hint = next(r for r in table if r["policy"] == "cpu_exact_hint")
    assert abs(hint["traj_delta_pct"] - 25) < 1e-9
    assert {r["policy"] for r in table} == {"gpu_only", "gpu_only_mru", "cpu_full", "cpu_exact_hint"}
    assert "vs cpu_full" in markdown(table, "cpu_full")


def test_per_turn_summary_pairs_turn_indices(tmp_path):
    from bench.summarize_turns import load_turns, markdown as turn_markdown, summarize_turns

    def write(name, ttfts):
        run = tmp_path / name
        run.mkdir()
        turns = [{"turn": i, "prompt_len": 100 * (i + 1), "num_cached_tokens": 50 * i,
                  "ttft_s": t, "turn_latency_s": 2 * t} for i, t in enumerate(ttfts)]
        (run / "result.json").write_text(json.dumps({"turns": turns}))

    write("h-gpu_only-s7-r0", [1.0, 2.0])
    write("h-cpu_full-s7-r0", [1.0, 1.0])
    write("other-cpu_full-s7-r0", [9.0, 9.0])
    rows = summarize_turns(load_turns(tmp_path, "h"))
    assert [row["turn"] for row in rows] == [0, 1]
    assert rows[1]["cpu_full"]["ttft_delta_pct"] == -50
    assert rows[1]["cpu_full"]["reused"] == 0.25
    assert "1000 (-50%), 25%" in turn_markdown(rows, ["gpu_only", "cpu_full"])
