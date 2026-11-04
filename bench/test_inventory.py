"""Check sandbox limits rather than assuming host resources are allocated."""

from bench.inventory import cgroup_v2


def test_nested_cgroup_inherits_tighter_parent_limits(tmp_path):
    mount = tmp_path / "cgroup"
    leaf = mount / "sandbox" / "worker"
    leaf.mkdir(parents=True)
    (leaf / "cpu.max").write_text("max 100000")
    (leaf / "memory.max").write_text("max")
    (leaf.parent / "cpu.max").write_text("200000 100000")
    (leaf.parent / "memory.max").write_text(str(8 * 1024**3))
    membership = tmp_path / "membership"
    membership.write_text("0::/sandbox/worker\n")
    mounts = tmp_path / "mountinfo"
    mounts.write_text(f"1 0 0:1 / {mount} rw - cgroup2 cgroup rw\n")
    result = cgroup_v2(membership, mounts)
    assert result["tightest_visible_cpu_quota_cores"] == 2
    assert result["tightest_visible_memory_limit_bytes"] == 8 * 1024**3
    assert len(result["visible_ancestors"]) == 3


def test_cgroup_namespace_root_and_unknown_limits(tmp_path):
    mount = tmp_path / "cgroup"
    mount.mkdir()
    (mount / "cpu.max").write_text("150000 100000")
    membership = tmp_path / "membership"
    membership.write_text("0::/\n")
    mounts = tmp_path / "mountinfo"
    mounts.write_text(f"1 0 0:1 /provider/sandbox {mount} rw - cgroup2 cgroup rw\n")
    result = cgroup_v2(membership, mounts)
    assert result["tightest_visible_cpu_quota_cores"] == 1.5
    assert result["tightest_visible_memory_limit_bytes"] is None


def test_missing_cgroup_data_does_not_claim_unlimited_resources(tmp_path):
    result = cgroup_v2(tmp_path / "missing", tmp_path / "missing")
    assert result["visible_ancestors"] == []
    assert result["tightest_visible_cpu_quota_cores"] is None
    assert result["tightest_visible_memory_limit_bytes"] is None
