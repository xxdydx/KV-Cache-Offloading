"""Per-block CPU prefix-cache copy costs and copy/compute overlap on one GPU.

Uses the real CPUBlockCache and BlockManager over a synthetic KV pool shaped
like Qwen3-0.6B (28 layers, 8 KV heads, head dim 128, BF16, 256-token blocks;
28 MiB per block) and compares against the earlier implementations and the
contiguous pinned-memcpy ceiling. No model weights are loaded.

Run: python3 -m bench.cpu_cache_microbench [--blocks 64] [--out result.json]
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from bench.baseline import positive_int

ROOT = Path(__file__).resolve().parents[1] / "nanovllm_upstream" / "nanovllm"


def load_engine_modules():
    # Only the cache/block-manager code; skips the package's CUDA model imports.
    for name, path in (("nanovllm", ROOT), ("nanovllm.engine", ROOT / "engine")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    modules = {}
    for name, relative in (("nanovllm.sampling_params", "sampling_params.py"),
                           ("nanovllm.engine.sequence", "engine/sequence.py"),
                           ("nanovllm.engine.block_manager", "engine/block_manager.py"),
                           ("nanovllm.engine.cpu_prefix_cache", "engine/cpu_prefix_cache.py")):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name.rsplit(".", 1)[1]] = module
    return modules["block_manager"].BlockManager, modules["cpu_prefix_cache"].CPUBlockCache


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    issued = time.perf_counter()
    torch.cuda.synchronize()
    return issued - start, time.perf_counter() - start


def per_block(issue_s, total_s, blocks, block_bytes):
    return {"issue_ms_per_block": 1000 * issue_s / blocks, "ms_per_block": 1000 * total_s / blocks,
            "gb_per_s": blocks * block_bytes / total_s / 1e9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=positive_int, default=64, help="Blocks copied per measurement")
    parser.add_argument("--layers", type=positive_int, default=28)
    parser.add_argument("--kv-heads", type=positive_int, default=8)
    parser.add_argument("--head-dim", type=positive_int, default=128)
    parser.add_argument("--block-size", type=positive_int, default=256)
    parser.add_argument("--matmul-iters", type=positive_int, default=200)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    BlockManager, CPUBlockCache = load_engine_modules()

    n = args.blocks
    num_blocks = 2 * n
    pool = torch.randn((2, args.layers, num_blocks, args.block_size, args.kv_heads, args.head_dim),
                       dtype=torch.bfloat16, device="cuda")
    block_shape = (2, args.layers, args.block_size, args.kv_heads, args.head_dim)
    block_bytes = pool[:, :, 0].numel() * pool.element_size()
    manager = BlockManager(num_blocks, args.block_size)
    runner = SimpleNamespace(world_size=1, block_size=args.block_size, kv_cache=pool)
    hashes = []
    for i in range(n):
        tokens = [i] * args.block_size
        h = BlockManager.compute_hash(tokens)
        manager.blocks[i].update(h, tokens)
        manager.hash_to_block_id[h] = i
        hashes.append(h)
    result = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "blocks": n, "block_bytes": block_bytes}

    # Ceiling: one contiguous pinned buffer each way.
    flat_gpu = torch.empty(n * block_bytes // 2, dtype=torch.bfloat16, device="cuda")
    flat_host = torch.empty_like(flat_gpu, device="cpu", pin_memory=True)
    flat_host.copy_(flat_gpu)
    result["pinned_memcpy_d2h"] = per_block(*timed(lambda: flat_host.copy_(flat_gpu, non_blocking=True)),
                                            n, block_bytes)
    result["pinned_memcpy_h2d"] = per_block(*timed(lambda: flat_gpu.copy_(flat_host, non_blocking=True)),
                                            n, block_bytes)
    flat_gpu = flat_host = None

    # The cost each save paid before the slab: a fresh pinned host allocation.
    def pinned_allocs():
        keep = [torch.empty(block_shape, dtype=pool.dtype, pin_memory=True) for _ in range(n)]
        return keep
    issue, total = timed(pinned_allocs)
    result["per_save_pinned_alloc"] = {"ms_per_block": 1000 * total / n}

    # Original path: pageable host memory, synchronous copies.
    def pageable_saves():
        keep = []
        for i in range(n):
            payload = torch.empty(block_shape, dtype=pool.dtype)
            torch.cuda.synchronize()
            payload.copy_(pool[:, :, i], non_blocking=False)
            keep.append(payload)
    result["pageable_sync_save"] = per_block(*timed(pageable_saves), n, block_bytes)

    # Current path: preallocated slab, dedicated copy stream.
    start = time.perf_counter()
    cache = CPUBlockCache(runner, manager, n * block_bytes, preallocate=True)
    result["slab_preallocate_s"] = time.perf_counter() - start
    result["slab_save"] = per_block(*timed(lambda: [cache.save_block(h) for h in hashes]), n, block_bytes)
    for h in hashes:
        block = manager.blocks[manager.hash_to_block_id.pop(h)]
        block.hash, block.token_ids = -1, []
    result["slab_restore"] = per_block(*timed(lambda: [cache.restore_block(h) for h in hashes]),
                                       n, block_bytes)
    cache.close()

    # Overlap: do saves on the copy stream slow down concurrent compute?
    x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")

    def matmuls():
        for _ in range(args.matmul_iters):
            torch.mm(x, x)
    matmuls()  # cuBLAS warmup
    _, compute_s = timed(matmuls)
    manager = BlockManager(num_blocks, args.block_size)
    for i, h in enumerate(hashes):
        manager.blocks[i].update(h, [i] * args.block_size)
        manager.hash_to_block_id[h] = i
    cache = CPUBlockCache(runner, manager, n * block_bytes, preallocate=True)
    _, copy_s = timed(lambda: [cache.save_block(h) for h in hashes])
    cache.clear()

    def both():
        # Saves first: each waits for prior current-stream work before starting.
        for h in hashes:
            cache.save_block(h)
        matmuls()
    _, both_s = timed(both)
    cache.close()
    result["overlap"] = {"compute_alone_s": compute_s, "saves_alone_s": copy_s, "together_s": both_s,
                         "serial_sum_s": compute_s + copy_s,
                         "hidden_fraction": (compute_s + copy_s - both_s) / min(compute_s, copy_s)}

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
