# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark multimodal preprocessing scheduling overhead.

This benchmark isolates the API-server/frontend scheduling path around
``BaseRenderer._process_multimodal_async`` without loading a model or touching
live network resources.  It uses a deterministic fake multimodal processor with
hot cached images and synthetic cold-image processing cost.

The primary workload models high-concurrency PDF/VL extraction traffic where
many requests contain the same not-yet-cached image plus already-cached images.
The desired scheduler behavior is that requests sharing the same cold image are
released as soon as that image is processed, instead of being serialized behind
unrelated multimodal preprocessing work.
"""

import argparse
import asyncio
import statistics
import sys
import threading
import time
import types
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple


@dataclass(frozen=True)
class RequestSpec:
    idx: int
    kind: Literal["cached", "partial_miss"]
    images: tuple[int, ...]


@dataclass(frozen=True)
class RequestResult:
    idx: int
    kind: str
    latency_s: float
    completed_at_s: float


class VllmSymbols(NamedTuple):
    base_renderer: type[Any]
    timing_context: type[Any]
    make_async: Any
    atomic_counter: type[Any]


def load_vllm_symbols() -> VllmSymbols:
    # The source checkout may not have compiled extension modules.  They are not
    # needed for this renderer-only benchmark, so stub them before importing
    # Python modules from the checkout.
    for name in ("vllm._C", "vllm._C_stable_libtorch"):
        sys.modules.setdefault(name, types.ModuleType(name))

    from vllm.multimodal.processing import TimingContext
    from vllm.renderers.base import BaseRenderer
    from vllm.utils.async_utils import make_async
    from vllm.utils.counter import AtomicCounter

    return VllmSymbols(
        base_renderer=BaseRenderer,
        timing_context=TimingContext,
        make_async=make_async,
        atomic_counter=AtomicCounter,
    )


class FakeInfo:
    model_id = "fake-mm-model"

    @staticmethod
    def parse_mm_data(mm_data: dict[str, Any]) -> dict[str, Any]:
        return mm_data


class FakeProcessor:
    def __init__(
        self,
        *,
        initial_cache: set[int],
        miss_cost_s: float,
    ) -> None:
        self.info = FakeInfo()
        self.cache = set(initial_cache)
        self.lock = threading.Lock()
        self.miss_cost_s = miss_cost_s
        self.apply_calls = 0
        self.try_apply_calls = 0
        self.prefill_calls = 0
        self.apply_cache_hits = 0
        self.apply_cache_misses = 0
        self.new_images_processed = 0

    def try_apply_cached(
        self,
        inputs: Any,
        timing_ctx: Any,
    ) -> dict[str, object] | None:
        self.try_apply_calls += 1
        images = inputs.mm_data_items["images"]
        with self.lock:
            if not all(image in self.cache for image in images):
                return None
        return self._result(inputs)

    def get_cache_missing_hashes(
        self,
        inputs: Any,
        timing_ctx: Any,
    ) -> list[str] | None:
        images = inputs.mm_data_items["images"]
        with self.lock:
            return [
                str(image)
                for image in images
                if image not in self.cache
            ]

    def try_apply_cached_or_get_missing_hashes(
        self,
        inputs: Any,
        timing_ctx: Any,
    ) -> types.SimpleNamespace:
        mm_input = self.try_apply_cached(inputs, timing_ctx)
        return types.SimpleNamespace(
            mm_input=mm_input,
            missing_hashes=(
                [] if mm_input is not None else
                self.get_cache_missing_hashes(inputs, timing_ctx)
            ),
        )

    def prefill_cache_items(
        self,
        inputs: Any,
        item_hashes: Sequence[str],
        timing_ctx: Any,
    ) -> None:
        selected_hashes = {int(item_hash) for item_hash in item_hashes}
        images = inputs.mm_data_items["images"]
        with self.lock:
            missing = [
                image
                for image in images
                if image in selected_hashes and image not in self.cache
            ]

        if not missing:
            return

        self.prefill_calls += 1
        time.sleep(self.miss_cost_s * len(missing))
        with self.lock:
            self.cache.update(missing)
            self.new_images_processed += len(missing)


    def apply(
        self,
        inputs: Any,
        timing_ctx: Any,
    ) -> dict[str, object]:
        self.apply_calls += 1
        images = inputs.mm_data_items["images"]
        with self.lock:
            missing = [image for image in images if image not in self.cache]

        if missing:
            self.apply_cache_misses += 1
            time.sleep(self.miss_cost_s * len(missing))
            with self.lock:
                self.cache.update(missing)
                self.new_images_processed += len(missing)
        else:
            self.apply_cache_hits += 1

        return self._result(inputs)

    @staticmethod
    def _result(inputs: Any) -> dict[str, object]:
        return {
            "prompt_token_ids": inputs.prompt,
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }


class FakeTimingRegistry:
    def __init__(self, timing_context: type[Any]) -> None:
        self._timing_context = timing_context

    def get(self, request_id: str) -> Any:
        return self._timing_context(enabled=False)


def make_renderer(
    symbols: VllmSymbols,
    processor: FakeProcessor,
    *,
    max_workers: int,
) -> tuple[Any, ThreadPoolExecutor]:
    class TestRenderer(symbols.base_renderer):
        def render_messages(self, messages: Any, params: Any) -> Any:
            raise NotImplementedError

    renderer = object.__new__(TestRenderer)
    renderer.api_process_rank = 0
    renderer._mm_req_counter = symbols.atomic_counter()
    renderer._readonly_mm_processor = None
    renderer.mm_processor = processor
    renderer._mm_timing_registry = FakeTimingRegistry(symbols.timing_context)
    renderer._process_mm_uuids = lambda *args, **kwargs: None
    renderer.update_mm_cache_stats = lambda: None
    renderer.config = type(
        "Config",
        (),
        {
            "cache_config": type(
                "CacheConfig",
                (),
                {"enable_prefix_caching": True},
            )()
        },
    )()
    renderer.model_config = type(
        "ModelConfig",
        (),
        {"multimodal_config": None},
    )()
    renderer.get_mm_processor = lambda: processor
    renderer._mm_cache_inflight_futures = {}

    executor = ThreadPoolExecutor(max_workers=max_workers)
    renderer._process_multimodal_in_executor = symbols.make_async(
        renderer._process_multimodal,
        executor=executor,
    )
    renderer._prefill_multimodal_cache_in_executor = symbols.make_async(
        renderer._prefill_multimodal_cache,
        executor=executor,
    )
    return renderer, executor


def make_duplicate_partial_miss_workload(
    *,
    unique_new_images: int,
    repeats_per_new_image: int,
    hot_images: int,
) -> tuple[list[RequestSpec], set[int]]:
    initial_cache = set(range(hot_images))
    specs: list[RequestSpec] = []
    idx = 0
    for repeat_idx in range(repeats_per_new_image):
        for new_idx in range(unique_new_images):
            base = (new_idx * 3 + repeat_idx) % (hot_images - 3)
            specs.append(
                RequestSpec(
                    idx=idx,
                    kind="partial_miss",
                    images=(
                        base,
                        base + 1,
                        base + 2,
                        hot_images + new_idx,
                    ),
                ))
            idx += 1
    return specs, initial_cache



def make_mixed_inflight_new_workload(
    *,
    mixed_requests: int,
    hot_images: int,
) -> tuple[list[RequestSpec], set[int]]:
    initial_cache = set(range(hot_images))
    shared_cold_image = hot_images
    specs = [
        RequestSpec(
            idx=0,
            kind="partial_miss",
            images=(0, 1, 2, shared_cold_image),
        )
    ]
    for idx in range(1, mixed_requests + 1):
        base = idx % (hot_images - 2)
        specs.append(
            RequestSpec(
                idx=idx,
                kind="partial_miss",
                images=(
                    base,
                    base + 1,
                    shared_cold_image,
                    shared_cold_image + idx,
                ),
            ))
    return specs, initial_cache


def make_cached_hol_workload(
    *,
    cached_requests: int,
    hot_images: int,
) -> tuple[list[RequestSpec], set[int]]:
    initial_cache = set(range(hot_images))
    specs = [
        RequestSpec(
            idx=0,
            kind="partial_miss",
            images=(0, 1, 2, hot_images),
        )
    ]
    for idx in range(1, cached_requests + 1):
        base = idx % (hot_images - 3)
        specs.append(
            RequestSpec(
                idx=idx,
                kind="cached",
                images=(base, base + 1, base + 2, base + 3),
            ))
    return specs, initial_cache


async def call_renderer(renderer: Any, spec: RequestSpec) -> Any:
    return await renderer._process_multimodal_async(
        [spec.idx],
        {"images": spec.images},
        mm_uuids=None,
        mm_processor_kwargs=None,
        tokenization_kwargs=None,
    )


async def run_workload(
    symbols: VllmSymbols,
    specs: list[RequestSpec],
    initial_cache: set[int],
    *,
    concurrency: int,
    renderer_workers: int,
    miss_cost_s: float,
) -> tuple[list[RequestResult], FakeProcessor, float]:
    processor = FakeProcessor(initial_cache=initial_cache, miss_cost_s=miss_cost_s)
    renderer, executor = make_renderer(
        symbols,
        processor,
        max_workers=renderer_workers,
    )
    queue: asyncio.Queue[RequestSpec | None] = asyncio.Queue()
    for spec in specs:
        queue.put_nowait(spec)
    for _ in range(concurrency):
        queue.put_nowait(None)

    results: list[RequestResult] = []
    start_s = time.perf_counter()

    async def worker() -> None:
        while True:
            spec = await queue.get()
            if spec is None:
                return
            request_start_s = time.perf_counter()
            await call_renderer(renderer, spec)
            done_s = time.perf_counter()
            results.append(
                RequestResult(
                    idx=spec.idx,
                    kind=spec.kind,
                    latency_s=done_s - request_start_s,
                    completed_at_s=done_s - start_s,
                ))

    try:
        await asyncio.gather(*(worker() for _ in range(concurrency)))
    finally:
        executor.shutdown(wait=True)

    wall_s = time.perf_counter() - start_s
    results.sort(key=lambda result: result.idx)
    return results, processor, wall_s


def percentile_ms(values_s: list[float], percentile: float) -> float:
    if not values_s:
        return 0.0
    ordered = sorted(values_s)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index] * 1000.0


def mean_ms(values_s: list[float]) -> float:
    if not values_s:
        return 0.0
    return statistics.fmean(values_s) * 1000.0


def summarize(
    results: list[RequestResult],
    processor: FakeProcessor,
    wall_s: float,
    prefix: str,
) -> dict[str, float]:
    all_latencies = [result.latency_s for result in results]
    cached_latencies = [
        result.latency_s for result in results if result.kind == "cached"
    ]
    partial_miss_latencies = [
        result.latency_s for result in results if result.kind == "partial_miss"
    ]

    summary = {
        f"{prefix}_mean_ms": mean_ms(all_latencies),
        f"{prefix}_p50_ms": percentile_ms(all_latencies, 0.50),
        f"{prefix}_p95_ms": percentile_ms(all_latencies, 0.95),
        f"{prefix}_p99_ms": percentile_ms(all_latencies, 0.99),
        f"{prefix}_cached_p95_ms": percentile_ms(cached_latencies, 0.95),
        f"{prefix}_partial_miss_mean_ms": mean_ms(partial_miss_latencies),
        f"{prefix}_partial_miss_p95_ms": percentile_ms(
            partial_miss_latencies,
            0.95,
        ),
        f"{prefix}_wall_ms": wall_s * 1000.0,
        f"{prefix}_throughput_rps": len(results) / wall_s,
        f"{prefix}_ready_50ms": float(
            sum(1 for result in results if result.completed_at_s <= 0.05)
        ),
        f"{prefix}_ready_100ms": float(
            sum(1 for result in results if result.completed_at_s <= 0.10)
        ),
        f"{prefix}_apply_calls": float(processor.apply_calls),
        f"{prefix}_prefill_calls": float(processor.prefill_calls),
        f"{prefix}_processor_work_calls": float(
            processor.apply_calls + processor.prefill_calls
        ),
        f"{prefix}_try_apply_calls": float(processor.try_apply_calls),
        f"{prefix}_apply_cache_hits": float(processor.apply_cache_hits),
        f"{prefix}_apply_cache_misses": float(processor.apply_cache_misses),
        f"{prefix}_new_images_processed": float(processor.new_images_processed),
    }
    return summary


async def run_once(args: argparse.Namespace) -> dict[str, float]:
    symbols = load_vllm_symbols()

    duplicate_specs, duplicate_cache = make_duplicate_partial_miss_workload(
        unique_new_images=args.unique_new_images,
        repeats_per_new_image=args.repeats_per_new_image,
        hot_images=args.hot_images,
    )
    duplicate_results, duplicate_processor, duplicate_wall_s = await run_workload(
        symbols,
        duplicate_specs,
        duplicate_cache,
        concurrency=args.concurrency,
        renderer_workers=args.renderer_workers,
        miss_cost_s=args.miss_cost,
    )

    cached_hol_specs, cached_hol_cache = make_cached_hol_workload(
        cached_requests=args.cached_hol_requests,
        hot_images=args.hot_images,
    )
    cached_hol_results, cached_hol_processor, cached_hol_wall_s = await run_workload(
        symbols,
        cached_hol_specs,
        cached_hol_cache,
        concurrency=args.concurrency,
        renderer_workers=args.renderer_workers,
        miss_cost_s=args.miss_cost,
    )

    mixed_specs, mixed_cache = make_mixed_inflight_new_workload(
        mixed_requests=args.mixed_requests,
        hot_images=args.hot_images,
    )
    mixed_results, mixed_processor, mixed_wall_s = await run_workload(
        symbols,
        mixed_specs,
        mixed_cache,
        concurrency=args.concurrency,
        renderer_workers=args.renderer_workers,
        miss_cost_s=args.miss_cost,
    )

    metrics = summarize(
        duplicate_results,
        duplicate_processor,
        duplicate_wall_s,
        "duplicate",
    )
    metrics.update(
        summarize(
            cached_hol_results,
            cached_hol_processor,
            cached_hol_wall_s,
            "cached_hol",
        ))
    metrics.update(
        summarize(
            mixed_results,
            mixed_processor,
            mixed_wall_s,
            "mixed",
        ))


    # Primary score: mean scheduler-ready latency for the mixed in-flight/new
    # miss workload.  This captures whether requests can preprocess their new
    # missing item while waiting for a different in-flight cache miss.
    metrics["mm_preprocess_ready_mean_ms"] = metrics["mixed_mean_ms"]
    return metrics


def aggregate_runs(runs: list[dict[str, float]]) -> dict[str, float]:
    keys = runs[0].keys()
    return {
        key: statistics.median(run[key] for run in runs)
        for key in keys
    }


def print_metrics(metrics: dict[str, float]) -> None:
    primary = "mm_preprocess_ready_mean_ms"
    print(f"METRIC {primary}={metrics[primary]:.6f}")
    for key in sorted(metrics):
        if key == primary:
            continue
        print(f"METRIC {key}={metrics[key]:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multimodal preprocessing scheduler readiness."
    )
    parser.add_argument("--concurrency", type=int, default=72)
    parser.add_argument("--renderer-workers", type=int, default=1)
    parser.add_argument("--miss-cost", type=float, default=0.02)
    parser.add_argument("--hot-images", type=int, default=256)
    parser.add_argument("--unique-new-images", type=int, default=12)
    parser.add_argument("--repeats-per-new-image", type=int, default=6)
    parser.add_argument("--cached-hol-requests", type=int, default=71)
    parser.add_argument("--mixed-requests", type=int, default=71)
    parser.add_argument("--repetitions", type=int, default=5)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    runs = [await run_once(args) for _ in range(args.repetitions)]
    print_metrics(aggregate_runs(runs))


if __name__ == "__main__":
    asyncio.run(main())
