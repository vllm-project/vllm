# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the native checkpoint iterator with a bounded CPU copy buffer."""

import argparse
import json
import os
import time
from pathlib import Path

from vllm.entrypoints.cli.types import CLISubcommand


def consume_tensor(tensor, buffer) -> int:
    """Touch every payload byte, including lazily mapped safetensors pages."""
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Expected a contiguous CPU checkpoint tensor")
    raw = tensor.reshape(-1).view(buffer.dtype)
    for offset in range(0, raw.numel(), buffer.numel()):
        chunk = raw[offset : offset + buffer.numel()]
        buffer[: chunk.numel()].copy_(chunk)
    return raw.numel()


def prefetch_only_result(config, loader, source) -> dict:
    """Prefetch the selected checkpoint files and join all I/O workers."""
    from vllm.model_executor.model_loader.weight_utils import (
        _natural_sort_key,
        prefetch_checkpoints,
    )

    # Reuse the loader's exact file selection, including safetensors index
    # filtering and inference-only file filtering.
    _, weight_files, _, _ = loader._prepare_weights(
        source.model_or_path,
        source.subfolder,
        source.revision,
        source.fall_back_to_pt,
        source.allow_patterns_overrides,
    )
    weight_files = sorted(weight_files, key=_natural_sort_key)
    checkpoint_bytes = sum(os.path.getsize(path) for path in weight_files)
    started = time.perf_counter()
    file_groups, _ = prefetch_checkpoints(
        weight_files,
        io_threads_per_file=config.safetensors_streaming_io_threads_per_file,
        block_size=config.safetensors_prefetch_block_size,
        prefetch_file_window=config.safetensors_streaming_prefetch_file_window,
    )
    seconds = time.perf_counter() - started
    return {
        "benchmark": "checkpoint_prefetch_only",
        "model_path": str(source.model_or_path),
        "load_format": config.load_format,
        "safetensors_load_strategy": config.safetensors_load_strategy,
        "safetensors_prefetch_num_threads": (config.safetensors_prefetch_num_threads),
        "safetensors_prefetch_block_size": (config.safetensors_prefetch_block_size),
        "prefetch_file_window": config.safetensors_streaming_prefetch_file_window,
        "io_threads_per_file": config.safetensors_streaming_io_threads_per_file,
        "load_workers": file_groups,
        "files": len(weight_files),
        "checkpoint_bytes": checkpoint_bytes,
        "file_parallelism": file_groups,
        "prefetch_seconds": seconds,
        "prefetch_gib_per_second": (
            checkpoint_bytes / 1024**3 / seconds if seconds else 0.0
        ),
        "note": (
            "Grouped whole-file reads joined before return; each active file "
            "uses a dedicated multi-thread executor. Measures storage/page-cache "
            "prefetch and excludes tensor mapping, CPU copies and model construction."
        ),
    }


def run(args: argparse.Namespace) -> dict:
    import torch

    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    config_kwargs = {
        "load_format": args.load_format,
        "model_loader_extra_config": args.model_loader_extra_config,
    }
    for name in (
        "safetensors_load_strategy",
        "safetensors_prefetch_num_threads",
        "safetensors_prefetch_block_size",
    ):
        value = getattr(args, name)
        if value is not None:
            config_kwargs[name] = value
    if args.prefetch_file_window is not None:
        config_kwargs["safetensors_streaming_prefetch_file_window"] = (
            args.prefetch_file_window
        )
    if args.io_threads_per_file is not None:
        config_kwargs["safetensors_streaming_io_threads_per_file"] = (
            args.io_threads_per_file
        )
    config = LoadConfig(**config_kwargs)
    loader = DefaultModelLoader(config)
    source = DefaultModelLoader.Source(
        model_or_path=str(Path(args.model_path).resolve()), revision=None
    )
    if args.prefetch_only:
        return prefetch_only_result(config, loader, source)
    # Allocate and touch the destination before timing. Reusing this buffer
    # forces mmap reads without retaining a second copy of the whole model.
    buffer = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device="cpu")
    tensor_bytes = 0
    tensors = 0
    started = time.perf_counter()
    weights = loader._get_weights_iterator(source)
    try:
        with torch.inference_mode():
            for _, weight in weights:
                tensor_bytes += consume_tensor(weight, buffer)
                tensors += 1
                del weight
    finally:
        weights.close()
    seconds = time.perf_counter() - started
    return {
        "benchmark": "weights_to_cpu_buffer",
        "model_path": str(source.model_or_path),
        "load_format": config.load_format,
        "safetensors_load_strategy": config.safetensors_load_strategy,
        "safetensors_prefetch_num_threads": config.safetensors_prefetch_num_threads,
        "safetensors_prefetch_block_size": config.safetensors_prefetch_block_size,
        "safetensors_streaming_prefetch_file_window": (
            config.safetensors_streaming_prefetch_file_window
        ),
        "safetensors_streaming_io_threads_per_file": (
            config.safetensors_streaming_io_threads_per_file
        ),
        "model_loader_extra_config": config.model_loader_extra_config,
        "load_workers": (
            config.model_loader_extra_config.get(
                "num_threads", loader.DEFAULT_NUM_THREADS
            )
            if config.model_loader_extra_config.get("enable_multithread_load")
            else 1
        ),
        "torch_num_threads": torch.get_num_threads(),
        "copy_buffer_mib": 16,
        "tensors": tensors,
        "tensor_bytes": tensor_bytes,
        "load_seconds": seconds,
        "load_gib_per_second": tensor_bytes / 1024**3 / seconds,
        "note": (
            "Logical payload throughput including page cache and CPU copies; "
            "excludes model construction, rank-specific selection, quantization "
            "and GPU transfer. Original background prefetch is not joined."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--load-format", choices=("auto", "safetensors", "pt"), default="auto"
    )
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=("lazy", "prefetch", "streaming", "eager"),
    )
    parser.add_argument("--safetensors-prefetch-num-threads", type=int)
    parser.add_argument("--safetensors-prefetch-block-size", type=int)
    parser.add_argument(
        "--prefetch-file-window",
        type=int,
        help="Streaming prefetch files",
    )
    parser.add_argument(
        "--io-threads-per-file",
        type=int,
        help="Streaming IO threads per active checkpoint file",
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help=(
            "Only prefetch selected checkpoint files into the page cache; "
            "join all workers and report prefetch throughput"
        ),
    )
    parser.add_argument("--model-loader-extra-config", type=json.loads, default={})


class LoadBenchSubcommand(CLISubcommand):
    name = "load-bench"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if not Path(args.model_path).is_dir():
            raise ValueError("--model-path must be a local checkpoint directory")
        print(json.dumps(run(args), indent=2))

    def subparser_init(self, subparsers):
        parser = subparsers.add_parser(
            self.name, help="Benchmark local weight loading without a GPU."
        )
        add_arguments(parser)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [LoadBenchSubcommand()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vllm load-bench",
        description="Load local checkpoint weights into a reusable CPU buffer.",
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    if not Path(args.model_path).is_dir():
        parser.error("--model-path must be an existing local checkpoint directory")
    LoadBenchSubcommand.cmd(args)


if __name__ == "__main__":
    main()
