# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic Qwen3-VL text+image HF processor benchmark.

This isolates the image preprocessing subpath that vLLM exercises for
tokenized Qwen3-VL chat requests containing text plus images. By the time this
code runs in the OpenAI server, chat template rendering and text tokenization
have already produced prompt token IDs; the HF processor is used only to turn
image objects into ``pixel_values`` and ``image_grid_thw``. The benchmark keeps
text tokens around the image placeholders so it models the business workload's
text+multi-image prompt shape without loading model weights or fetching network
resources.
"""

import argparse
import concurrent.futures
import statistics
import sys
import time
import types
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor, Qwen3VLVideoProcessor

QWEN36_IMAGE_PROCESSOR_KWARGS: dict[str, object] = {
    "size": {
        "shortest_edge": 65536,
        "longest_edge": 16777216,
    },
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}

DEFAULT_PDF_PAGE_WIDTH = 1786
DEFAULT_PDF_PAGE_HEIGHT = 2526



def stub_flash_attention_modules() -> None:
    """Allow importing Qwen3-VL model code in CPU-only benchmark envs."""
    module = types.ModuleType("vllm.vllm_flash_attn")
    module.flash_attn_varlen_func = lambda *args, **kwargs: None
    module.get_scheduler_metadata = lambda *args, **kwargs: None
    sys.modules.setdefault("vllm.vllm_flash_attn", module)


stub_flash_attention_modules()

from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor  # noqa: E402
from vllm.multimodal.parse import MultiModalDataParser  # noqa: E402
from vllm.multimodal.processing.context import TimingContext  # noqa: E402
from vllm.multimodal.processing.inputs import ProcessorInputs  # noqa: E402


@dataclass(frozen=True)
class RequestInputs:
    prompt_token_ids: list[int]
    images: list[Image.Image]
    mm_items: Any


@dataclass(frozen=True)
class ApplyResult:
    latency_s: float
    stage_secs: dict[str, float]
    prompt_tokens: int
    image_placeholders: int


class FakeProcessingContext:
    def __init__(self, mm_config: Any) -> None:
        self._mm_config = mm_config

    def call_hf_processor(
        self,
        processor: Any,
        data: dict[str, object],
        kwargs: dict[str, object],
    ) -> Any:
        kwargs = dict(kwargs)
        kwargs.setdefault("return_tensors", "pt")
        return processor(**data, **kwargs)

    def get_mm_config(self) -> Any:
        return self._mm_config


class FakeProcessingInfo:
    model_id = "deterministic-qwen3vl-hf-processor-benchmark"

    def __init__(self, hf_processor: Qwen3VLProcessor) -> None:
        self._hf_processor = hf_processor
        self.ctx = FakeProcessingContext(SimpleNamespace(video_pruning_rate=None))
        self._hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(
                spatial_merge_size=hf_processor.image_processor.merge_size,
            ),
            image_token_id=hf_processor.image_token_id,
            video_token_id=hf_processor.video_token_id,
            vision_start_token_id=hf_processor.vision_start_token_id,
            vision_end_token_id=hf_processor.vision_end_token_id,
        )

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser()

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self._hf_processor

    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessor:
        return self._hf_processor.image_processor

    def get_tokenizer(self) -> PreTrainedTokenizerFast:
        return self._hf_processor.tokenizer

    def get_hf_config(self) -> Any:
        return self._hf_config


class FakeDummyInputs:
    def __init__(self, info: FakeProcessingInfo) -> None:
        self.info = info

    def get_dummy_text(self, mm_counts: dict[str, int]) -> str:
        return "".join(
            "<|vision_start|><|image_pad|><|vision_end|>"
            for _ in range(mm_counts.get("image", 0))
        )


class BenchmarkProcessor:
    def __init__(self) -> None:
        tokenizer = make_tokenizer()
        hf_processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(**QWEN36_IMAGE_PROCESSOR_KWARGS),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(),
        )
        info = FakeProcessingInfo(hf_processor)
        self.hf_processor = hf_processor
        self.processor = Qwen3VLMultiModalProcessor(
            info,
            FakeDummyInputs(info),
            cache=None,
        )
        self.tokenizer = tokenizer

    def make_prompt_token_ids(
        self,
        image_count: int,
        text_tokens: int,
    ) -> list[int]:
        token_ids: list[int] = []
        text_token_id = self.tokenizer.unk_token_id
        text_chunks = image_count + 1
        base_chunk = text_tokens // text_chunks
        remainder = text_tokens % text_chunks

        for index in range(image_count):
            chunk_size = base_chunk + (1 if index < remainder else 0)
            token_ids.extend([text_token_id] * chunk_size)
            token_ids.extend(
                [
                    self.tokenizer.vision_start_token_id,
                    self.tokenizer.image_token_id,
                    self.tokenizer.vision_end_token_id,
                ]
            )

        tail_size = base_chunk + (1 if image_count < remainder else 0)
        token_ids.extend([text_token_id] * tail_size)
        return token_ids

    def make_prompt_text(self, image_count: int, text_tokens: int) -> str:
        text_chunks = image_count + 1
        base_chunk = text_tokens // text_chunks
        remainder = text_tokens % text_chunks
        parts = []
        for index in range(image_count):
            chunk_size = base_chunk + (1 if index < remainder else 0)
            parts.append(" ".join(["benchmark"] * chunk_size))
            parts.append("<|vision_start|><|image_pad|><|vision_end|>")
        tail_size = base_chunk + (1 if image_count < remainder else 0)
        parts.append(" ".join(["benchmark"] * tail_size))
        return " ".join(parts)


def make_tokenizer() -> PreTrainedTokenizerFast:
    special_tokens = [
        "<unk>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    tokenizer_impl = Tokenizer(
        WordLevel(
            vocab={token: index for index, token in enumerate(special_tokens)},
            unk_token="<unk>",
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="<unk>",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens[1:]}
    )
    tokenizer.image_token = "<|image_pad|>"
    tokenizer.video_token = "<|video_pad|>"
    tokenizer.vision_start_token = "<|vision_start|>"
    tokenizer.vision_end_token = "<|vision_end|>"
    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    tokenizer.video_token_id = tokenizer.convert_tokens_to_ids(tokenizer.video_token)
    tokenizer.vision_start_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.vision_start_token
    )
    tokenizer.vision_end_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.vision_end_token
    )
    return tokenizer


def make_image(width: int, height: int, image_index: int) -> Image.Image:
    y = np.arange(height, dtype=np.uint16)[:, None]
    x = np.arange(width, dtype=np.uint16)[None, :]
    base = (x * 3 + y * 5 + image_index * 17) & 0xFF
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = base.astype(np.uint8)
    rgb[:, :, 1] = ((base + image_index * 11) & 0xFF).astype(np.uint8)
    rgb[:, :, 2] = ((base * 2 + 31) & 0xFF).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def make_requests(
    bench: BenchmarkProcessor,
    *,
    request_count: int,
    images_per_request: int,
    width: int,
    height: int,
    text_tokens: int,
) -> list[RequestInputs]:
    requests: list[RequestInputs] = []
    image_index = 0
    prompt = bench.make_prompt_token_ids(images_per_request, text_tokens)
    for _ in range(request_count):
        images = []
        for _ in range(images_per_request):
            images.append(make_image(width, height, image_index))
            image_index += 1
        mm_items = bench.processor.data_parser.parse_mm_data({"image": images})
        requests.append(RequestInputs(prompt, images, mm_items))
    return requests


def run_vllm_apply(
    bench: BenchmarkProcessor,
    request: RequestInputs,
) -> ApplyResult:
    timing_ctx = TimingContext(enabled=True)
    start = time.perf_counter()
    output = bench.processor.apply(
        ProcessorInputs(
            request.prompt_token_ids,
            request.mm_items,
            hf_processor_mm_kwargs={"return_mm_token_type_ids": False},
        ),
        timing_ctx,
    )
    latency_s = time.perf_counter() - start
    return ApplyResult(
        latency_s=latency_s,
        stage_secs=timing_ctx.get_stats_dict(),
        prompt_tokens=len(output["prompt_token_ids"]),
        image_placeholders=len(output["mm_placeholders"].get("image", [])),
    )


def run_full_hf_processor(
    bench: BenchmarkProcessor,
    request: RequestInputs,
) -> float:
    start = time.perf_counter()
    bench.hf_processor(
        text=bench.make_prompt_text(
            len(request.images),
            len(request.prompt_token_ids),
        ),
        images=request.images,
        return_tensors="pt",
        return_mm_token_type_ids=False,
    )
    return time.perf_counter() - start


def run_direct_image_processor(
    bench: BenchmarkProcessor,
    request: RequestInputs,
) -> float:
    start = time.perf_counter()
    bench.hf_processor.image_processor(
        images=request.images,
        return_tensors="pt",
    )
    return time.perf_counter() - start


def verify_direct_image_parity(
    bench: BenchmarkProcessor,
    request: RequestInputs,
) -> bool:
    output = bench.processor.apply(
        ProcessorInputs(
            request.prompt_token_ids,
            request.mm_items,
            hf_processor_mm_kwargs={"return_mm_token_type_ids": False},
        ),
        TimingContext(enabled=False),
    )
    direct = bench.hf_processor.image_processor(
        images=request.images,
        return_tensors="pt",
    )
    image_items = output["mm_kwargs"]["image"]
    vllm_pixel_values = torch.cat(
        [item["pixel_values"].data for item in image_items],
        dim=0,
    )
    vllm_grid = torch.stack(
        [item["image_grid_thw"].data for item in image_items],
        dim=0,
    )
    if vllm_pixel_values.shape != direct["pixel_values"].shape:
        image_processor = bench.hf_processor.image_processor
        temporal_patch_size = image_processor.temporal_patch_size
        patch_size = image_processor.patch_size
        if (
            vllm_pixel_values.shape[0] == direct["pixel_values"].shape[0]
            and vllm_pixel_values.shape[1] * temporal_patch_size
            == direct["pixel_values"].shape[1]
        ):
            vllm_pixel_values = (
                vllm_pixel_values.view(
                    vllm_pixel_values.shape[0],
                    3,
                    patch_size,
                    patch_size,
                )
                .unsqueeze(2)
                .expand(-1, -1, temporal_patch_size, -1, -1)
                .reshape_as(direct["pixel_values"])
            )

    return bool(
        torch.equal(vllm_grid, direct["image_grid_thw"])
        and torch.equal(vllm_pixel_values, direct["pixel_values"])
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * q))
    return ordered[index]


def mean_ms(values_s: list[float]) -> float:
    if not values_s:
        return 0.0
    return statistics.fmean(values_s) * 1000.0


def median_metric(runs: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.median(run[key] for run in runs)
        for key in runs[0]
    }


def run_once(args: argparse.Namespace) -> dict[str, float]:
    bench = BenchmarkProcessor()
    requests = make_requests(
        bench,
        request_count=args.requests,
        images_per_request=args.images_per_request,
        width=args.width,
        height=args.height,
        text_tokens=args.text_tokens,
    )

    for request in requests[: args.warmups]:
        run_vllm_apply(bench, request)

    start = time.perf_counter()
    if args.workers == 1:
        apply_results = [run_vllm_apply(bench, request) for request in requests]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            apply_results = list(
                executor.map(lambda request: run_vllm_apply(bench, request), requests)
            )
    wall_s = time.perf_counter() - start

    full_hf_times = [run_full_hf_processor(bench, request) for request in requests]
    direct_image_times = [
        run_direct_image_processor(bench, request) for request in requests
    ]
    apply_times = [result.latency_s for result in apply_results]
    apply_hf_processor_times = [
        result.stage_secs.get("apply_hf_processor_secs", 0.0)
        for result in apply_results
    ]
    apply_prompt_updates_times = [
        result.stage_secs.get("apply_prompt_updates_secs", 0.0)
        for result in apply_results
    ]

    parity_ok = verify_direct_image_parity(bench, requests[0])
    prompt_tokens = {result.prompt_tokens for result in apply_results}
    placeholders = {result.image_placeholders for result in apply_results}
    expected_placeholders = args.images_per_request
    if placeholders != {expected_placeholders}:
        raise AssertionError(
            f"Expected {expected_placeholders} image placeholders, got {placeholders}"
        )

    return {
        "qwen3_vllm_apply_mean_ms": mean_ms(apply_times),
        "qwen3_vllm_apply_p50_ms": percentile(apply_times, 0.50) * 1000.0,
        "qwen3_vllm_apply_p95_ms": percentile(apply_times, 0.95) * 1000.0,
        "qwen3_vllm_apply_wall_ms": wall_s * 1000.0,
        "qwen3_vllm_apply_throughput_rps": len(requests) / wall_s,
        "qwen3_apply_hf_processor_mean_ms": mean_ms(apply_hf_processor_times),
        "qwen3_apply_prompt_updates_mean_ms": mean_ms(
            apply_prompt_updates_times
        ),
        "qwen3_full_hf_processor_mean_ms": mean_ms(full_hf_times),
        "qwen3_direct_image_processor_mean_ms": mean_ms(direct_image_times),
        "qwen3_direct_image_parity": 1.0 if parity_ok else 0.0,
        "qwen3_image_pixel_count": float(args.width * args.height),
        "qwen3_processor_longest_edge": 16777216.0,
        "qwen3_processor_patch_size": 16.0,
        "qwen3_processor_merge_size": 2.0,
        "qwen3_text_token_count": float(args.text_tokens),
        "qwen3_prompt_token_count": float(next(iter(prompt_tokens))),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    primary = "qwen3_vllm_apply_mean_ms"
    print(f"METRIC {primary}={metrics[primary]:.6f}")
    for key in sorted(metrics):
        if key != primary:
            print(f"METRIC {key}={metrics[key]:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-VL text+image vLLM processor path."
    )
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--images-per-request", type=int, default=4)
    parser.add_argument("--width", type=int, default=DEFAULT_PDF_PAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_PDF_PAGE_HEIGHT)
    parser.add_argument("--text-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if args.requests <= 0:
        raise ValueError("--requests must be positive")
    if args.images_per_request <= 0:
        raise ValueError("--images-per-request must be positive")
    if args.text_tokens < 0:
        raise ValueError("--text-tokens must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    runs = [run_once(args) for _ in range(args.repetitions)]
    metrics = median_metric(runs)
    if metrics["qwen3_direct_image_parity"] != 1.0:
        raise AssertionError("Direct image processor parity failed")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
