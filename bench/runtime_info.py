"""Read container RAM limits and process RSS without importing GPU libraries."""

from pathlib import Path


def memory_info():
    result = {}
    for name in ("memory.max", "memory.current", "memory.peak", "memory.events"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            value = path.read_text().strip()
            if name == "memory.events":
                result[name] = {k:int(v) for k,v in (line.split() for line in value.splitlines())}
            else:
                result[name] = int(value) if value.isdigit() else value
    path = Path("/proc/self/status")
    if path.exists():
        result["process_rss_kib"] = next((int(line.split()[1]) for line in path.read_text().splitlines()
                                          if line.startswith("VmRSS:")), None)
    return result
