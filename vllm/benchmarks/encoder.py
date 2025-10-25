# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone benchmark for multimodal encoder forward latency."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.benchmarks.lib.utils import write_to_json
from vllm.config import LoadConfig, ModelConfig, MultiModalConfig, VllmConfig
from vllm.config.multimodal import ImageDummyOptions
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


def _parse_image_size(size_str: str) -> ImageSize:
    if "x" in size_str:
        width_str, height_str = size_str.lower().split("x", maxsplit=1)
        width = int(width_str)
        height = int(height_str)
    else:
        width = height = int(size_str)

    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {size_str!r}.")

    return ImageSize(width=width, height=height)


def _parse_mm_processor_kwargs(raw: str | None) -> Mapping[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--mm-processor-kwargs must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("--mm-processor-kwargs must decode to a JSON object")
    return value


def _check_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False"
        )


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", type=str, required=True, help="Model (HF repo or local path)."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer to use (defaults to --model).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype (auto, float16, bfloat16, float32, ...).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the encoder on (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes (number of images) to benchmark.",
    )
    parser.add_argument(
        "--image-sizes",
        type=str,
        nargs="+",
        default=["224x224", "336x336"],
        help="Image sizes to benchmark, e.g. 224x224 or 384.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
        help="Number of timed iterations per configuration.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=16,
        help="Dummy prompt length used when constructing fake inputs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of remote code when loading configs/processors.",
    )
    parser.add_argument(
        "--mm-processor-kwargs",
        type=str,
        default=None,
        help="JSON string of additional kwargs passed to the model's HF processor.",
    )
    parser.add_argument(
        "--measure-preprocessing",
        action="store_true",
        help="Measure and report preprocessing latency in addition to encoder latency.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to dump results as JSON.",
    )


def _create_model_config(
    args: argparse.Namespace,
    max_batch: int,
    max_width: int,
    max_height: int,
) -> ModelConfig:
    limit = ImageDummyOptions(count=max_batch, width=max_width, height=max_height)
    mm_config = MultiModalConfig(limit_per_prompt={"image": limit})

    tokenizer = args.tokenizer or args.model

    return ModelConfig(
        model=args.model,
        tokenizer=tokenizer,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        multimodal_config=mm_config,
    )


def _load_model(
    model_config: ModelConfig,
    device: torch.device,
) -> torch.nn.Module:
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format="dummy"),
    )
    model = get_model(vllm_config=vllm_config)
    model.eval()
    model.to(device)
    return model


def _build_processor(model_config: ModelConfig):
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _build_mm_kwargs(
    processor,
    prompt_len: int,
    batch_size: int,
    image_size: ImageSize,
    mm_processor_kwargs: Mapping[str, Any],
    device: torch.device,
    include_preproc: bool,
) -> tuple[dict[str, Any], float | None]:
    mm_counts = {"image": batch_size}
    mm_options = {
        "image": ImageDummyOptions(
            count=batch_size, width=image_size.width, height=image_size.height
        )
    }

    inputs = processor.dummy_inputs.get_dummy_processor_inputs(
        seq_len=prompt_len,
        mm_counts=mm_counts,
        mm_options=mm_options,
    )

    start = time.perf_counter() if include_preproc else None
    mm_inputs = processor.apply(
        prompt=inputs.prompt,
        mm_data=inputs.mm_data,
        hf_processor_mm_kwargs=mm_processor_kwargs,
        tokenization_kwargs=inputs.tokenization_kwargs,
    )
    preprocess_ms = None
    if include_preproc:
        preprocess_ms = (time.perf_counter() - start) * 1000.0

    mm_kwargs_items = mm_inputs["mm_kwargs"]
    mm_kwargs = mm_kwargs_items.get_data()

    moved = MultiModalKwargs.as_kwargs(mm_kwargs, device=device)
    return dict(moved), preprocess_ms


def _cuda_synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_encoder(
    model: torch.nn.Module,
    mm_kwargs: Mapping[str, Any],
    warmup_iters: int,
    num_iters: int,
    device: torch.device,
) -> Sequence[float]:
    times_ms: list[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.cuda.device(device):
                _cuda_synchronize_if_needed(device)
                for _ in range(warmup_iters):
                    model.get_multimodal_embeddings(**mm_kwargs)
                _cuda_synchronize_if_needed(device)

                for _ in range(num_iters):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    model.get_multimodal_embeddings(**mm_kwargs)
                    end_event.record()
                    end_event.synchronize()
                    times_ms.append(start_event.elapsed_time(end_event))
        else:
            _cuda_synchronize_if_needed(device)
            for _ in range(warmup_iters):
                model.get_multimodal_embeddings(**mm_kwargs)
            _cuda_synchronize_if_needed(device)

            for _ in range(num_iters):
                start = time.perf_counter()
                model.get_multimodal_embeddings(**mm_kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                times_ms.append(elapsed_ms)

    return times_ms


def _summarize(times_ms: Sequence[float]) -> dict[str, float]:
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "avg": float(arr.mean()),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "stdev": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
    }


def _print_results(
    results: Sequence[dict[str, Any]],
    include_preproc: bool,
) -> None:
    print("Standalone encoder benchmark results (latency in ms):")
    for entry in results:
        summary = entry["summary"]
        batch = entry["batch_size"]
        width = entry["image_width"]
        height = entry["image_height"]
        line = (
            f"  batch={batch:>3}, size={width}x{height}: "
            f"avg={summary['avg']:.3f}, p50={summary['p50']:.3f}, "
            f"p90={summary['p90']:.3f}, p99={summary['p99']:.3f}"
        )
        print(line)
        if include_preproc and entry.get("preprocessing_ms") is not None:
            print(f"      preprocessing={entry['preprocessing_ms']:.3f} ms")


def _dump_json(
    path: str,
    args: argparse.Namespace,
    device: torch.device,
    results: Sequence[dict[str, Any]],
) -> None:
    payload = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "device": str(device),
        "warmup_iters": args.warmup_iters,
        "num_iters": args.num_iters,
        "measure_preprocessing": args.measure_preprocessing,
        "results": results,
    }
    write_to_json(path, payload)


def main(args: argparse.Namespace) -> None:
    if not args.batch_sizes:
        raise ValueError("--batch-sizes must contain at least one value")
    if not args.image_sizes:
        raise ValueError("--image-sizes must contain at least one value")

    image_sizes = [_parse_image_size(size) for size in args.image_sizes]
    max_batch = max(args.batch_sizes)
    max_width = max(size.width for size in image_sizes)
    max_height = max(size.height for size in image_sizes)

    device = torch.device(args.device)
    _check_device(device)

    mm_processor_kwargs = _parse_mm_processor_kwargs(args.mm_processor_kwargs)

    model_config = _create_model_config(args, max_batch, max_width, max_height)
    model = _load_model(model_config, device)
    processor = _build_processor(model_config)

    results: list[dict[str, Any]] = []

    for batch in args.batch_sizes:
        for size in image_sizes:
            mm_kwargs, preprocess_ms = _build_mm_kwargs(
                processor=processor,
                prompt_len=args.prompt_len,
                batch_size=batch,
                image_size=size,
                mm_processor_kwargs=mm_processor_kwargs,
                device=device,
                include_preproc=args.measure_preprocessing,
            )

            times_ms = _time_encoder(
                model=model,
                mm_kwargs=mm_kwargs,
                warmup_iters=args.warmup_iters,
                num_iters=args.num_iters,
                device=device,
            )

            summary = _summarize(times_ms)
            result_entry = {
                "batch_size": batch,
                "image_width": size.width,
                "image_height": size.height,
                "latency_ms": times_ms,
                "summary": summary,
            }
            if args.measure_preprocessing:
                result_entry["preprocessing_ms"] = preprocess_ms
            results.append(result_entry)

    _print_results(results, include_preproc=args.measure_preprocessing)

    if args.output_json:
        _dump_json(args.output_json, args, device, results)
