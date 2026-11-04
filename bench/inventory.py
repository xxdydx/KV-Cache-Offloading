"""Save a hardware/limits manifest without importing CUDA or ML packages."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def cgroup_v2(proc_cgroup="/proc/self/cgroup", mountinfo="/proc/self/mountinfo"):
    """Resolve the process cgroup and inspect all ancestors visible in its mount.

    Hidden parent limits and cgroup v1 require separate provider-side inspection.
    Raw membership is retained so unavailable data cannot be mistaken for no limit.
    """
    membership = read_text(proc_cgroup)
    result = {"membership": membership, "visible_ancestors": [],
              "limitation": "Visible cgroup v2 limits only; hidden parents and v1 limits are unknown."}
    process_path = next((line[3:] for line in (membership or "").splitlines()
                         if line.startswith("0::")), None)
    for line in (read_text(mountinfo) or "").splitlines():
        left, _, right = line.partition(" - ")
        if not right.startswith("cgroup2 ") or process_path is None:
            continue
        fields = left.split()
        unescape = lambda value: value.replace("\\040", " ").replace("\\134", "\\")
        mount_root, mount = Path(unescape(fields[3])), Path(unescape(fields[4]))
        try:
            relative = Path(process_path).relative_to(mount_root)
        except ValueError:
            if process_path != "/":
                continue
            relative = Path(".")  # Cgroup namespace rooted at this mount.
        current = mount / relative
        if ".." in relative.parts:
            continue
        while True:
            values = {name: read_text(current / name) for name in (
                "cpu.max", "cpu.stat", "cpuset.cpus.effective", "memory.max",
                "memory.current", "memory.peak", "memory.events")}
            result["visible_ancestors"].append({"path": str(current), **values})
            if current == mount:
                break
            current = current.parent
        break
    quotas, memory_limits = [], []
    for level in result["visible_ancestors"]:
        quota, _, period = (level["cpu.max"] or "").partition(" ")
        if quota.isdigit() and period.isdigit() and int(period):
            quotas.append(int(quota) / int(period))
        if (level["memory.max"] or "").isdigit():
            memory_limits.append(int(level["memory.max"]))
    result["tightest_visible_cpu_quota_cores"] = min(quotas, default=None)
    result["tightest_visible_memory_limit_bytes"] = min(memory_limits, default=None)
    return result


def collect():
    versions = {}
    for name in ("torch", "transformers", "triton", "flash-attn", "vllm", "nano-vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    disk = shutil.disk_usage(Path.cwd())
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "python_executable": sys.executable,
        "host_logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_v2": cgroup_v2(),
        "host_cpu": command(["lscpu"]),
        "host_memory": read_text("/proc/meminfo"),
        "process_status": read_text("/proc/self/status"),
        "gpu": command(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version,pci.bus_id", "--format=csv"]),
        "tool_paths": {name: shutil.which(name) for name in ("nvcc", "ncu", "nsys")},
        "versions": versions, "disk_free_bytes": disk.free,
        "git_revision": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "limitations": ["Inventory only; does not validate CUDA kernels or profiler counter permissions.",
                         "Host CPU/RAM and filesystem free space are not sandbox entitlements.",
                         "CPU sharing, hidden quotas, storage quota and GPU exclusivity need provider confirmation."],
    }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New JSON file; existing files are not overwritten")
    args = parser.parse_args()
    save_json(args.out, collect())
    print(f"Saved inventory to {args.out}")


if __name__ == "__main__":
    main()
