#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the forward pass of a multimodal (vision) encoder only.

The script will:
 - Load the HuggingFace image processor for `--model`.
 - Instantiate the corresponding HF model from its config with random
   weights (no pretrained weights are downloaded/loaded for the model
   parameters themselves).
 - Convert dummy PIL images to model inputs via the HF processor.
 - Run warmup iterations and measure per-batch latency for the encoder
   forward pass for combinations of batch sizes and image sizes.

Usage example:
  python benchmarks/benchmark_encoder_forward.py \
    --model "openai/clip-vit-base-patch32" \
    --batch-sizes 1,4,8 --image-sizes 224,384 --num-iters 200 --warmup 10 --device cuda

Notes:
 - The script attempts a few common ways to call the vision encoder:
   `encoder(pixel_values=...)`, `encoder(pixel_values)`, `encoder.encode_image(...)`,
   `encoder.encode(...)`. If none of these work the script will error.
 - The model is instantiated from config with random weights using
   `AutoModel.from_config`. This avoids downloading large pretrained
   weights while allowing us to measure forward-pass performance.
"""

import argparse
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from benchmark_utils import TimeCollector
from vllm.config import ModelConfig
from vllm.transformers_utils.processor import cached_image_processor_from_config


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x != ""]


def find_encoder_module(model: torch.nn.Module) -> torch.nn.Module:
    """Try to heuristically locate a vision encoder submodule inside HF model.

    Falls back to the model object itself if no better candidate is found.
    """
    candidates = [
        "vision_model",
        "vision_encoder",
        "encoder",
        "vision",
        "backbone",
        "model",
    ]

    for name in candidates:
        if hasattr(model, name):
            attr = getattr(model, name)
            if isinstance(attr, torch.nn.Module):
                return attr

    # Nothing obvious found, return the model and rely on flexible calling
    return model


def run_encoder_call(encoder: torch.nn.Module, pixel_values: torch.Tensor) -> Any:
    """Try a few call styles for vision encoders and return the output."""
    # Prefer methods that many HF vision models expose
    if hasattr(encoder, "encode_image"):
        return encoder.encode_image(pixel_values)
    if hasattr(encoder, "encode"):
        try:
            return encoder.encode(pixel_values)
        except TypeError:
            # Some encode implementations expect kwargs
            return encoder.encode(pixel_values=pixel_values)

    # Try calling the module directly with and without kwarg
    try:
        return encoder(pixel_values=pixel_values)
    except Exception:
        return encoder(pixel_values)


def make_dummy_images(batch: int, size: int) -> list[Image.Image]:
    # Create white RGB PIL images
    images = [
        Image.fromarray(np.ones((size, size, 3), dtype=np.uint8) * 255)
        for _ in range(batch)
    ]
    return images


def benchmark(
    model_name: str,
    batch_sizes: Iterable[int],
    image_sizes: Iterable[int],
    num_iters: int,
    warmup: int,
    device: str,
    trust_remote_code: bool,
) -> None:
    device_obj = torch.device(
        device if torch.cuda.is_available() or device == "cpu" else "cpu"
    )

    print(
        "Constructing vLLM ModelConfig and loading processor for "
        f"model={model_name} (trust_remote_code={trust_remote_code})"
    )
    model_config = ModelConfig(model=model_name, trust_remote_code=trust_remote_code)
    # Try to use vLLM cached image processor when the model is multimodal.
    try:
        processor = cached_image_processor_from_config(model_config)
    except ValueError as e:
        # Not a multimodal model or processor unavailable; fall back to HF
        # AutoImageProcessor to preserve original behavior.
        msg = str(e)
        if "multimodal" in msg or "not multimodal" in msg:
            processor = AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
        else:
            # Re-raise other unexpected errors
            raise

    print(
        "Instantiating model encoder with random weights from vLLM hf "
        "config (no pretrained weights)."
    )
    # Use vision_config when present (multimodal hf configs expose vision_config)
    hf_cfg = model_config.hf_config
    if hasattr(hf_cfg, "vision_config") and hf_cfg.vision_config is not None:
        model = AutoModel.from_config(hf_cfg.vision_config)
    else:
        model = AutoModel.from_config(hf_cfg)
    model.eval()
    model.to(device_obj)

    encoder = find_encoder_module(model)
    encoder.eval()
    encoder.to(device_obj)

    # Test call once to verify input signature / warm up JIT/backends
    for bs in batch_sizes:
        for im_sz in image_sizes:
            print(f"Preparing dummy batch: batch_size={bs}, image_size={im_sz}")

            images = make_dummy_images(bs, im_sz)
            # Use HF processor to create pixel_values tensor
            inputs = processor(images, return_tensors="pt")
            # Common HF name is 'pixel_values'
            if "pixel_values" not in inputs:
                # Fall back to raw tensors
                tensors = [
                    torch.from_numpy(np.array(i).transpose(2, 0, 1)) for i in images
                ]
                pixel_values = torch.stack(tensors).float()
            else:
                pixel_values = inputs["pixel_values"].float()

            pixel_values = pixel_values.to(device_obj)

            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = run_encoder_call(encoder, pixel_values)

            # Measure
            collector = TimeCollector(TimeCollector.US)
            with torch.no_grad():
                for _ in range(num_iters):
                    start = time.monotonic_ns()
                    _ = run_encoder_call(encoder, pixel_values)
                    if device_obj.type == "cuda":
                        torch.cuda.synchronize(device=device_obj)
                    end = time.monotonic_ns()
                    collector.collect(end - start)

            avg_us, max_us = collector.dump_avg_max()
            print(f"batch={bs} size={im_sz}: avg={avg_us:.3f} us, max={max_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark multimodal encoder forward pass (dummy weights, HF processor)"
        )
    )
    parser.add_argument("--model", required=True, help="HuggingFace model identifier")
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8",
        help="Comma-separated list of batch sizes to benchmark",
    )
    parser.add_argument(
        "--image-sizes",
        default="224",
        help="Comma-separated list of square image sizes (pixels)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of measured iterations per config",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before measuring",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run the encoder on: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom processors/feature-extractors from remote code",
    )

    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    image_sizes = parse_int_list(args.image_sizes)

    benchmark(
        model_name=args.model,
        batch_sizes=batch_sizes,
        image_sizes=image_sizes,
        num_iters=args.num_iters,
        warmup=args.warmup,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
