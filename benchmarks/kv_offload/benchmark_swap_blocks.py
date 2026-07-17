# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM KV Cache Offload Benchmark — Block Copy Performance

This script benchmarks the throughput (bandwidth) of copying fixed-size blocks
between CPU (host) and GPU (device) memory under different transfer strategies:

  - naive     : Single element‑wise copy via PyTorch tensor indexing.
  - batch     : vLLM `swap_blocks_batch`.
  - swap      : vLLM `swap_blocks`.
  - triton    : Custom Triton kernel.

For each strategy, a range of block sizes (from 2**min_block_exp to 2**max_block_exp
bytes) is tested, and the achieved bandwidth (GB/s) is reported.
After all measurements, a results table is printed and a bandwidth plot is generated.
"""

import argparse
import random

import torch

try:
    from vllm import _custom_ops as ops

    HAS_VLLM_OPS = True
except ImportError:
    HAS_VLLM_OPS = False
    print(
        "Warning: vllm._custom_ops.swap_blocks_batch/swap_blocks not available, "
        "batch/swap modes will be skipped."
    )

try:
    from vllm.v1.kv_offload.cpu.swap_blocks_triton import NUM_SMS, _swap_blocks_kernel

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton kernel not found – triton mode will be skipped.")

try:
    from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
except ImportError:

    def pin_tensor(tensor):
        return tensor


try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available – plot will be skipped.")


def swap_blocks_triton(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    *,
    bytes_per_chunk: int,
):
    """Launch the Triton-based block copy kernel."""
    n = src_addrs.numel()
    _swap_blocks_kernel[(min(NUM_SMS, n),)](
        src_addrs.to("cuda", non_blocking=True),
        dst_addrs.to("cuda", non_blocking=True),
        sizes.to("cuda", non_blocking=True),
        n,
        BYTES_PER_CHUNK=bytes_per_chunk,
    )


def format_size(
    num_bytes: int,
    decimal_places: int = 4,
    use_binary: bool = True,
    target_unit: str = None,
) -> str:
    """Format a byte count as a human-readable string."""
    if num_bytes == 0:
        return f"0 {target_unit or 'B'}"
    units = ["B", "KB", "MB", "GB"]
    base = 1024 if use_binary else 1000
    if target_unit is not None:
        target_exp = units.index(target_unit)
        size = num_bytes / (base**target_exp)
        return f"{size:.{decimal_places}f} {target_unit}"
    exponent = 0
    size = num_bytes
    while size >= base and exponent < len(units) - 1:
        size /= base
        exponent += 1
    return f"{size:.{decimal_places}f} {units[exponent]}"


def format_bandwidth(bytes_per_sec: float, decimal_places: int = 4) -> str:
    """Format bandwidth in GB/s."""
    return (
        format_size(int(bytes_per_sec), decimal_places=decimal_places, target_unit="GB")
        + "/s"
    )


def benchmark_swap_blocks(
    host: torch.Tensor,
    device: torch.Tensor,
    block_sizes: list,
    n_iters: int = 100,
    warmup_iters: int = 3,
    direction: str = "H2D",
    mode: str = "naive",
    triton_chunk_size: int = 8192,
):
    """
    Benchmark block swapping for a range of block sizes.

    Args:
        host: Pinned CPU tensor.
        device: GPU tensor.
        block_sizes: List of block sizes (in bytes) to test.
        n_iters: Number of timed iterations.
        warmup_iters: Number of warmup iterations.
        direction: "H2D" (host->device) or "D2H" (device->host).
        mode: "naive", "batch", "swap", or "triton".
        triton_chunk_size: Bytes per chunk for the Triton kernel.
    """
    # Check availability and skip if necessary
    if mode in ("batch", "swap") and not HAS_VLLM_OPS:
        print(f"Skipping {mode} mode (vLLM ops unavailable).")
        return [None] * len(block_sizes)
    if mode == "triton" and not HAS_TRITON:
        print("Skipping triton mode (Triton kernel unavailable).")
        return [None] * len(block_sizes)

    print(f"\n--- {direction} ({mode} mode) ---")
    bandwidths = []

    # Create CUDA events once for timing (reused across runs)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for block_size in block_sizes:
            # Reshape memory into (num_blocks, block_size) views
            host_view = host.view(-1, block_size)
            device_view = device.view(-1, block_size)
            num_blocks = host_view.size(0)

            # Pre‑generate random (src_idx, dst_idx) pairs
            tasks = [
                (random.randint(0, num_blocks - 1), random.randint(0, num_blocks - 1))
                for _ in range(n_iters)
            ]

            if mode == "naive":

                def run_naive(
                    iterations: int,
                    _tasks=tasks,
                    _host_view=host_view,
                    _device_view=device_view,
                    _direction=direction,
                ):
                    for i in range(iterations):
                        src_idx, dst_idx = _tasks[i]
                        if _direction == "H2D":
                            _device_view[dst_idx] = _host_view[src_idx]
                        else:
                            _host_view[dst_idx] = _device_view[src_idx]

                run_copy = run_naive

            elif mode == "batch":
                if direction == "H2D":
                    src_addrs = torch.tensor(
                        [host_view[i].data_ptr() for i, j in tasks], dtype=torch.int64
                    )
                    dst_addrs = torch.tensor(
                        [device_view[j].data_ptr() for i, j in tasks], dtype=torch.int64
                    )
                else:
                    src_addrs = torch.tensor(
                        [device_view[i].data_ptr() for i, j in tasks], dtype=torch.int64
                    )
                    dst_addrs = torch.tensor(
                        [host_view[j].data_ptr() for i, j in tasks], dtype=torch.int64
                    )
                sizes = torch.full((n_iters,), block_size, dtype=torch.int64)

                def run_batch(
                    iterations: int,
                    _src_addrs=src_addrs,
                    _dst_addrs=dst_addrs,
                    _sizes=sizes,
                ):
                    with torch.cuda.stream(torch.cuda.Stream()):
                        ops.swap_blocks_batch(_src_addrs, _dst_addrs, _sizes)

                run_copy = run_batch

            elif mode == "swap":
                block_mapping = torch.tensor(tasks, dtype=torch.int64, device="cpu")
                if direction == "H2D":
                    src_tensor = host_view
                    dst_tensor = device_view
                else:
                    src_tensor = device_view
                    dst_tensor = host_view

                def run_swap(
                    iterations: int,
                    _src=src_tensor,
                    _dst=dst_tensor,
                    _block_size=block_size,
                    _mapping=block_mapping,
                ):
                    ops.swap_blocks(_src, _dst, _block_size, _mapping)

                run_copy = run_swap

            elif mode == "triton":
                if direction == "H2D":
                    src_addrs = torch.tensor(
                        [host_view[i].data_ptr() for i, j in tasks], dtype=torch.int64
                    ).cuda()
                    dst_addrs = torch.tensor(
                        [device_view[j].data_ptr() for i, j in tasks], dtype=torch.int64
                    ).cuda()
                else:
                    src_addrs = torch.tensor(
                        [device_view[i].data_ptr() for i, j in tasks], dtype=torch.int64
                    ).cuda()
                    dst_addrs = torch.tensor(
                        [host_view[j].data_ptr() for i, j in tasks], dtype=torch.int64
                    ).cuda()
                sizes = torch.full((n_iters,), block_size, dtype=torch.int64).cuda()

                def run_triton(
                    iterations: int,
                    _src_addrs=src_addrs,
                    _dst_addrs=dst_addrs,
                    _sizes=sizes,
                    _chunk=triton_chunk_size,
                ):
                    swap_blocks_triton(
                        _src_addrs, _dst_addrs, _sizes, bytes_per_chunk=_chunk
                    )

                run_copy = run_triton

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Warmup:
            for _ in range(warmup_iters):
                run_copy(1)
            torch.cuda.synchronize()

            # Timed run:
            start_event.record()
            run_copy(n_iters)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed_s = elapsed_ms / 1000.0

            bw = (block_size * n_iters) / elapsed_s
            bandwidths.append(bw)
            print(
                f"  block size: {format_size(block_size):>12s}"
                f"  bandwidth: {format_bandwidth(bw)}"
            )

    return bandwidths


def print_results_table(block_sizes, results):
    """
    Print a Markdown formatted table of bandwidths (GB/s) for all directions and modes.
    """
    directions = ["H2D", "D2H"]
    modes = ["naive", "batch", "swap", "triton"]

    # Build header
    header = ["size"] + [f"{d}-{m}" for d in directions for m in modes]

    # Print Markdown table header
    print("\n### Bandwidth Results (GB/s)")
    print("| " + " | ".join(header) + " |")
    # Separator row
    print("|" + "|".join([" --- " for _ in header]) + "|")

    for i, bs in enumerate(block_sizes):
        row = [format_size(bs)]
        for d in directions:
            for m in modes:
                bw = results.get(d, {}).get(m, [None] * len(block_sizes))[i]
                if bw is None:
                    row.append("N/A")
                else:
                    # Convert bytes/s to GB/s
                    gb_s = bw / 1e9
                    row.append(f"{gb_s:.4f}")
        print("| " + " | ".join(row) + " |")


def plot_results(block_sizes, results, output_file=None):
    """
    Create a bandwidth plot (GB/s vs block size in bytes) with:
      - X‑axis: logarithmic base‑2, tick labels in human‑readable format
        (using format_size with 0 decimal places, e.g., '256B', '1K', '2M').
      - Y‑axis: linear, starting from 0, with standard decimal formatting.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available.")
        return

    directions = ["H2D", "D2H"]
    modes = ["naive", "batch", "swap", "triton"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    line_styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each (direction, mode) combination
    for d_idx, direction in enumerate(directions):
        for m_idx, mode in enumerate(modes):
            bw_list = results.get(direction, {}).get(mode, [None] * len(block_sizes))
            x_vals, y_vals = [], []
            for bs, bw in zip(block_sizes, bw_list):
                if bw is not None:
                    x_vals.append(bs)
                    y_vals.append(bw / 1e9)  # Convert to GB/s
            if not x_vals:
                continue
            label = f"{direction}-{mode}"
            color = colors[(d_idx * len(modes) + m_idx) % len(colors)]
            style = line_styles[m_idx % len(line_styles)]
            ax.plot(
                x_vals,
                y_vals,
                label=label,
                color=color,
                linestyle=style,
                marker="o",
                markersize=4,
            )

    # Configure X‑axis (log2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes)
    # Generate labels
    labels = [format_size(bs, decimal_places=0) for bs in block_sizes]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Block Size")

    # Configure Y‑axis (linear)
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(
        plt.ScalarFormatter(useOffset=False, useMathText=False)
    )
    ax.set_ylabel("Bandwidth (GB/s)")

    ax.set_title("Block Copy Bandwidth")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark block copy bandwidth between host and "
        "device using different vLLM copy strategies."
    )
    parser.add_argument(
        "--total-size",
        type=int,
        default=2**34,
        help="Total memory allocated (in bytes) for the host and device tensors. "
        "Default: 2**34 (16 GiB).",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=100,
        help="Number of timed iterations per block size. Default: 100.",
    )
    parser.add_argument(
        "--min-block-exp",
        type=int,
        default=8,
        help="Exponent for the smallest block size (2**exp bytes). Default: 8 (256 B).",
    )
    parser.add_argument(
        "--max-block-exp",
        type=int,
        default=31,
        help="Exponent for the largest block size (2**exp bytes). Default: 31 (2 GiB).",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["H2D", "D2H"],
        default=["H2D", "D2H"],
        help="Transfer directions to benchmark. Default: H2D D2H.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["naive", "batch", "swap", "triton"],
        default=["naive", "batch", "swap", "triton"],
        help="Copy modes to benchmark. Default: all four modes.",
    )
    parser.add_argument(
        "--triton-chunk-size",
        type=int,
        default=8192,
        help="Bytes per chunk for the Triton kernel. Default: 8192.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display or save the bandwidth plot.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="benchmark_swap_blocks.png",
        help="Save the plot to the given file path (e.g., benchmark_swap_blocks.png). "
        "Implies --no-plot is ignored.",
    )
    args = parser.parse_args()

    total_size = args.total_size
    n_iters = args.n_iters
    block_sizes = [2**n for n in range(args.min_block_exp, args.max_block_exp + 1)]
    directions = args.directions
    modes = args.modes
    triton_chunk_size = args.triton_chunk_size

    # Allocate pinned host memory and device memory
    dtype = torch.uint8
    host_raw = torch.randn(total_size // 4, dtype=torch.float32, device="cpu").view(
        dtype
    )
    device_raw = torch.randn(total_size // 4, dtype=torch.float32, device="cuda").view(
        dtype
    )
    pin_tensor(host_raw)

    total_bytes = host_raw.nelement() * host_raw.element_size()
    print(f"Total memory allocated: {format_size(total_bytes)}")

    # Store results: results[direction][mode] = list of bandwidths
    results = {}
    for direction in directions:
        results[direction] = {}
        for mode in modes:
            bw_list = benchmark_swap_blocks(
                host_raw,
                device_raw,
                block_sizes,
                n_iters,
                direction=direction,
                mode=mode,
                triton_chunk_size=triton_chunk_size,
            )
            results[direction][mode] = bw_list

    # Print summary table
    print_results_table(block_sizes, results)

    # Plot if requested
    if not args.no_plot or args.save_plot:
        plot_results(block_sizes, results, output_file=args.save_plot)


if __name__ == "__main__":
    main()
