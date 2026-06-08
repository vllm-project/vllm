#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demo: NIXL TTL-based engine eviction prevents host memory leaks.

Simulates many P (prefill) engines registering with one D (decode) worker,
tracking RSS to show that memory stays bounded with TTL eviction vs grows
linearly without it.

Usage:
    CUDA_VISIBLE_DEVICES="" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python tests/v1/kv_connector/unit/demo_ttl_eviction_memory.py \
        [--num-engines 500] [--num-blocks 100] [--measure-every 25]
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (  # noqa: E402
    NixlAgentMetadata,
    NixlConnectorWorker,
)
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
)

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__)),
)
from test_nixl_connector import FakeNixlWrapper  # noqa: E402
from utils import create_vllm_config, make_kv_cache_config  # noqa: E402


@contextlib.contextmanager
def _init_distributed():
    """Initialize single-rank distributed env (gloo, no GPU)."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def get_rss_mib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def create_decode_worker(
    engine_ttl: float,
    num_blocks: int,
    vllm_config=None,
) -> NixlConnectorWorker:
    if vllm_config is None:
        vllm_config = create_vllm_config(
            kv_connector_extra_config={"engine_ttl": engine_ttl},
        )
    block_size = 16
    kv_cache_config = make_kv_cache_config(block_size=block_size, num_blocks=num_blocks)

    with set_current_vllm_config(vllm_config):
        worker = NixlConnectorWorker(
            vllm_config,
            vllm_config.kv_transfer_config.engine_id,
            kv_cache_config,
        )

    num_kv_heads = 4
    head_size = 16
    dtype_size = 2  # float16
    block_len = block_size * num_kv_heads * head_size * dtype_size
    worker.block_len_per_layer = [block_len, block_len]

    from vllm.distributed.kv_transfer.kv_connector.utils import (
        TransferTopology,
    )

    worker.transfer_topo = TransferTopology(
        tp_rank=0,
        tp_size=1,
        block_size=block_size,
        engine_id=worker.engine_id,
        is_mla=False,
        is_mamba=False,
        total_num_kv_heads=num_kv_heads,
        attn_backends=[FlashAttentionBackend],
    )
    worker.compat_hash = "demo-hash"
    worker.kv_caches_base_addr[worker.engine_id][0] = [0x1000, 0x2000]
    worker.num_regions = 2
    worker.num_descs = 2 * worker.num_blocks
    worker.src_blocks_data = [(0x1000, block_len, 0)] * worker.num_blocks + [
        (0x2000, block_len, 0)
    ] * worker.num_blocks
    worker.src_xfer_handles_by_block_size[block_size] = 0xDEAD

    return worker


def make_p_metadata(
    index: int,
    worker: NixlConnectorWorker,
    num_blocks: int,
) -> NixlAgentMetadata:
    return NixlAgentMetadata(
        engine_id=f"prefill-{index}",
        agent_metadata=b"fake",
        kv_caches_base_addr=[0xA000 + index * 0x100, 0xB000 + index * 0x100],
        device_id=0,
        num_blocks=num_blocks,
        block_lens=list(worker.block_len_per_layer),
        kv_cache_layout=worker.kv_cache_layout,
        block_size=worker.block_size,
        ssm_sizes=(0, 0),
        attn_backend_name=worker.backend_name,
        physical_blocks_per_logical_kv_block=1,
    )


def run_scenario(
    name: str,
    engine_ttl: float,
    num_engines: int,
    num_blocks: int,
    measure_every: int,
    vllm_config=None,
) -> list[tuple[int, float]]:
    print(f"\n--- {name} (engine_ttl={engine_ttl}s) ---")

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker.NixlWrapper",
        FakeNixlWrapper,
    ):
        worker = create_decode_worker(engine_ttl, num_blocks, vllm_config)

    gc.collect()
    baseline = get_rss_mib()
    measurements: list[tuple[int, float]] = [(0, baseline)]
    print(f"{'Engines':>10}  {'RSS (MiB)':>12}  {'Delta (MiB)':>12}")
    print(f"{'─' * 10}  {'─' * 12}  {'─' * 12}")
    print(f"{0:>10}  {baseline:>12.1f}  {0.0:>+12.1f}")

    t0 = time.perf_counter()
    for i in range(num_engines):
        meta = make_p_metadata(i, worker, num_blocks)
        worker.add_remote_agent(meta, remote_tp_rank=0, remote_tp_size=1)

        if engine_ttl > 0:
            for eid in list(worker._engine_last_active):
                worker._engine_last_active[eid] = time.perf_counter() - engine_ttl - 1
            worker._evict_stale_engines()

        if (i + 1) % measure_every == 0 or i == num_engines - 1:
            gc.collect()
            rss = get_rss_mib()
            measurements.append((i + 1, rss))
            print(f"{i + 1:>10}  {rss:>12.1f}  {rss - baseline:>+12.1f}")

    elapsed = time.perf_counter() - t0
    worker.shutdown()
    print(f"  ({elapsed:.1f}s elapsed)")
    return measurements


def print_summary(
    with_ttl: list[tuple[int, float]],
    no_ttl: list[tuple[int, float]],
) -> None:
    ttl_baseline = with_ttl[0][1]
    no_ttl_baseline = no_ttl[0][1]
    ttl_peak_delta = max(rss - ttl_baseline for _, rss in with_ttl)
    no_ttl_peak_delta = max(rss - no_ttl_baseline for _, rss in no_ttl)

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  With TTL:    peak delta = {ttl_peak_delta:+.1f} MiB (bounded)")
    print(f"  Without TTL: peak delta = {no_ttl_peak_delta:+.1f} MiB (unbounded)")
    print(f"  Ratio:       {no_ttl_peak_delta / max(ttl_peak_delta, 0.01):.1f}x")
    print("=" * 50)


def plot_results(
    with_ttl: list[tuple[int, float]],
    no_ttl: list[tuple[int, float]],
    output_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not available, skipping plot ({output_path})")
        return

    ttl_baseline = with_ttl[0][1]
    no_ttl_baseline = no_ttl[0][1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        [e for e, _ in no_ttl],
        [rss - no_ttl_baseline for _, rss in no_ttl],
        "r-o",
        markersize=3,
        label="Without TTL (engine_ttl=0)",
    )
    ax.plot(
        [e for e, _ in with_ttl],
        [rss - ttl_baseline for _, rss in with_ttl],
        "g-o",
        markersize=3,
        label="With TTL (eviction enabled)",
    )
    ax.set_xlabel("Number of P engines registered")
    ax.set_ylabel("RSS delta (MiB)")
    ax.set_title("NIXL TTL Eviction: Host Memory Usage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Demo: NIXL TTL eviction prevents memory leaks"
    )
    parser.add_argument(
        "--num-engines", type=int, default=500, help="P engines to simulate"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=100,
        help="KV cache blocks per P engine",
    )
    parser.add_argument(
        "--measure-every",
        type=int,
        default=25,
        help="Measure RSS every N engines",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="ttl_eviction_memory.png",
        help="Output plot path (empty to skip)",
    )
    parser.add_argument(
        "--engine-ttl",
        type=float,
        default=2.0,
        help="TTL value for the with-TTL scenario",
    )
    args = parser.parse_args()

    print("=== NIXL TTL Eviction Memory Demo ===")
    print(
        f"Config: {args.num_engines} engines, {args.num_blocks} blocks/engine, "
        f"TTL={args.engine_ttl}s"
    )

    with _init_distributed():
        vllm_config_ttl = create_vllm_config(
            kv_connector_extra_config={"engine_ttl": args.engine_ttl},
        )
        vllm_config_no_ttl = create_vllm_config(
            kv_connector_extra_config={"engine_ttl": 0.0},
        )
        with set_current_vllm_config(vllm_config_ttl):
            with_ttl = run_scenario(
                "With TTL",
                engine_ttl=args.engine_ttl,
                num_engines=args.num_engines,
                num_blocks=args.num_blocks,
                measure_every=args.measure_every,
                vllm_config=vllm_config_ttl,
            )

        with set_current_vllm_config(vllm_config_no_ttl):
            no_ttl = run_scenario(
                "Without TTL",
                engine_ttl=0.0,
                num_engines=args.num_engines,
                num_blocks=args.num_blocks,
                measure_every=args.measure_every,
                vllm_config=vllm_config_no_ttl,
            )

    print_summary(with_ttl, no_ttl)

    if args.plot:
        plot_results(with_ttl, no_ttl, args.plot)


if __name__ == "__main__":
    main()
