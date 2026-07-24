# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark renderer-stage queueing under concurrent multimodal preprocessing.

The renderer offloads tokenization and multimodal preprocessing to thread-pool
executors. This benchmark keeps several multimodal preprocessing jobs in flight
and measures how long unrelated renderer work has to queue behind them, swept
over item size and ``renderer_num_workers``.

It complements ``vllm bench mm-processor``, which reports the per-stage cost of
a single request; this one reports what concurrent requests do to each other.
No GPU or model weights are required: only the renderer is constructed.

Run:
    python benchmarks/overheads/benchmark_multimodal_preprocessing_concurrency.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct

    python benchmarks/overheads/benchmark_multimodal_preprocessing_concurrency.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct --modality video --video-frames 8 16
"""

import asyncio
import json
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image

from vllm.config import ModelConfig, VllmConfig
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.utils.argparse_utils import FlexibleArgumentParser

IMAGE_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>Describe the image."
VIDEO_PROMPT = "<|vision_start|><|video_pad|><|vision_end|>Describe the video."


def build_renderer(model: str, num_workers: int, cache_gb: float) -> HfRenderer:
    model_config = ModelConfig(
        model,
        runner="generate",
        max_model_len=8192,
        mm_processor_cache_gb=cache_gb,
        renderer_num_workers=num_workers,
    )
    return HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )


def make_image(size: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def make_video(num_frames: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (num_frames, size, size, 3), dtype=np.uint8)


def make_mm_data(modality: str, magnitude: int, frame_size: int, seed: int) -> dict:
    if modality == "image":
        return {"image": [make_image(magnitude, seed)]}
    return {"video": [make_video(magnitude, frame_size, seed)]}


def percentile(values: list[float], p: int) -> float:
    if len(values) < 2:
        return max(values)
    return statistics.quantiles(values, n=100)[p - 1]


async def run_case(
    renderer: HfRenderer,
    prompt: str,
    modality: str,
    magnitude: int,
    frame_size: int,
    concurrency: int,
    duration_s: float,
) -> dict[str, float]:
    loop = asyncio.get_running_loop()
    await renderer._process_multimodal_async(
        prompt, make_mm_data(modality, magnitude, frame_size, 0), None, None, None
    )

    heavy_lat: list[float] = []
    probe_wait: list[float] = []
    stop = time.monotonic() + duration_s
    seed = 0

    async def heavy_loop() -> None:
        nonlocal seed
        while time.monotonic() < stop:
            seed += 1
            mm_data = make_mm_data(modality, magnitude, frame_size, seed)
            t0 = time.perf_counter()
            await renderer._process_multimodal_async(prompt, mm_data, None, None, None)
            heavy_lat.append(time.perf_counter() - t0)

    async def probe_loop() -> None:
        # Submit a no-op to the tokenizer executor: the round trip is the time a
        # concurrent text request would spend queueing before it can tokenize.
        while time.monotonic() < stop:
            t0 = time.perf_counter()
            await loop.run_in_executor(renderer._executor, time.perf_counter)
            probe_wait.append(time.perf_counter() - t0)
            await asyncio.sleep(0.01)

    await asyncio.gather(*(heavy_loop() for _ in range(concurrency)), probe_loop())

    return {
        "mm_ops": len(heavy_lat),
        "mm_avg_ms": round(1000 * statistics.mean(heavy_lat), 1),
        "probes": len(probe_wait),
        "probe_p50_ms": round(1000 * statistics.median(probe_wait), 2),
        "probe_p99_ms": round(1000 * percentile(probe_wait, 99), 2),
        "probe_max_ms": round(1000 * max(probe_wait), 2),
    }


async def run(args: Any) -> dict[str, dict[str, float]]:
    prompt = args.prompt or (IMAGE_PROMPT if args.modality == "image" else VIDEO_PROMPT)
    magnitudes = args.image_sizes if args.modality == "image" else args.video_frames

    results: dict[str, dict[str, float]] = {}
    for num_workers in args.num_workers:
        renderer = build_renderer(args.model, num_workers, args.mm_processor_cache_gb)
        for magnitude in magnitudes:
            unit = "px" if args.modality == "image" else "frames"
            key = f"workers={num_workers},{args.modality}={magnitude}{unit}"
            results[key] = await run_case(
                renderer,
                prompt,
                args.modality,
                magnitude,
                args.frame_size,
                args.concurrency,
                args.duration,
            )
            print(f"{key}: {results[key]}")
        renderer.shutdown()
    return results


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark renderer queueing under concurrent MM preprocessing."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--modality", choices=["image", "video"], default="image")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--video-frames", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, nargs="+", default=[1])
    parser.add_argument("--mm-processor-cache-gb", type=float, default=4.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    results = asyncio.run(run(args))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
