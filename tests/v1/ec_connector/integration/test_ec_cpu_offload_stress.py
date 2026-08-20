# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stress/liveness test for the EC CPU offload connector (`ECCPUConnector`).

Targets the event-driven memcpy-completion signal (CUDA-event polling -- see
docs/superpowers/specs/2026-07-26-ec-memcpy-completion-HANDOFF.md, Stage 7)
under heavy save/evict/reload churn: many more distinct images than the CPU
region can hold, across several batches of many concurrent requests, run
under the most aggressive concurrency setting available.

Three capacities are set against each other so that saves, evictions, and
reloads all occur in the same run -- see `_ENCODER_CACHE_TOKENS` and
`_EC_REGION_IMAGES`. Some re-requested images are served from the CPU region
and some are recomputed after being evicted from it; both are correct, so the
test asserts that reloads happen at all and that nothing is left permanently
in flight, rather than pinning down which images took which path.

This is a robustness check, not a correctness-by-exact-match test: outputs
are only checked for basic sanity (non-empty), not equality to any baseline.
Exact-match correctness of reloaded embeddings is covered by
test_ec_cpu_offload_e2e.py.
"""

import os
import time

import pytest

from tests.utils import create_new_process_for_each_test
from vllm import LLM
from vllm.platforms import current_platform

from ._ec_cpu_offload_helpers import (
    build_llm_kwargs,
    default_sampling_params,
    detect_concurrency_matrix,
    drain_pending_push_work,
    estimate_bytes_per_image,
    estimate_embeds_per_image,
    get_encoder_cache_size,
    get_scheduler_ec_connector,
    make_image_message,
    make_text_message,
    observe_ec_transfers,
    shutdown_llm,
    stress_image_urls,
    stress_max_pixels,
)

pytestmark = [
    pytest.mark.skipif(
        current_platform.device_count() < 1,
        reason="EC CPU offload stress test requires a GPU",
    ),
    # See test_ec_cpu_offload_e2e.py for why: the global cleanup_fixture
    # would run cleanup_dist_env_and_memory() in the main pytest process,
    # breaking this test's own worker fork if it runs after another case.
    pytest.mark.skip_global_cleanup,
]

_NUM_DISTINCT_IMAGES = 24
_NUM_BATCHES = 5
_REQUESTS_PER_BATCH = 20
_RUN_TIMEOUT_S = 900.0

# GPU encoder cache capacity in encoder embeddings, set via
# max_num_batched_tokens (SchedulerConfig.encoder_cache_size derives from it).
# Paired with max_pixels=stress_max_pixels() and the zeroed video limit in
# build_llm_kwargs: compute_mm_encoder_budget floors the cache at the largest
# tokens-per-item over active modalities, and either the uncapped image figure
# or the video figure would otherwise stop the cache ever evicting.
# Deliberately tiny -- roughly two images -- so any image re-requested a batch
# later is certain to have been evicted from the GPU cache. Without that miss
# the scheduler answers from the GPU cache and never consults the connector.
# Stays at or above the largest single item (448px -> 256 embeds) so every
# request remains schedulable.
_ENCODER_CACHE_TOKENS = 320

# CPU region capacity in images, sized between the two other capacities:
# well above the GPU encoder cache, so an image the GPU cache dropped can
# still be served from CPU; below the full working set (24 images, ~3550
# blocks at one block per visual token), so the region itself keeps evicting.
_EC_REGION_IMAGES = 20


def _build_batches(urls: list[str]) -> list[list]:
    """5 batches x ~20 concurrent requests mixing fresh and re-requested images.

    Every fifth request after the first batch re-requests an image from a few
    positions before the end of the previous batch. That distance is chosen
    against the two cache capacities: far enough back that the image has left
    the small GPU encoder cache, recent enough that the CPU region usually
    still holds it. The remaining requests walk fresh images to drive
    save/evict churn, with an occasional text-only request mixed in.
    """
    batches = []
    for batch_idx in range(_NUM_BATCHES):
        batch = []
        for req_idx in range(_REQUESTS_PER_BATCH):
            if req_idx % 5 == 0 and batch_idx > 0:
                prev_batch_end = batch_idx * _REQUESTS_PER_BATCH
                url = urls[(prev_batch_end - 5 - req_idx // 5) % len(urls)]
                batch.append(make_image_message(url, text="Describe this image."))
            elif req_idx % 7 == 0:
                batch.append(make_text_message(f"Say the number {req_idx}."))
            else:
                url = urls[(batch_idx * _REQUESTS_PER_BATCH + req_idx) % len(urls)]
                batch.append(make_image_message(url, text="What's in this image?"))
        batches.append(batch)
    return batches


@create_new_process_for_each_test(method="spawn")
def test_ec_cpu_offload_stress() -> None:
    # Force the in-process EngineCore client so the scheduler-side connector
    # delegate is directly reachable for the has_pending_push_work() probe.
    # Set directly (not via the `monkeypatch` fixture): this test body runs
    # in a disposable spawned subprocess, so there's nothing to restore, and
    # a live MonkeyPatch fixture object wouldn't survive being pickled across
    # the spawn boundary anyway.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Prefer the widest parallelism with async scheduling on -- the most
    # aggressive step overlap available on this machine. The exact resulting
    # max_concurrent_batches also depends on VllmConfig.use_v2_model_runner
    # (see _ec_cpu_offload_helpers.detect_concurrency_matrix), so this picks
    # by settings rather than by a predicted concurrency value.
    case = max(
        detect_concurrency_matrix(),
        key=lambda c: (c.num_gpus, c.tp_size, c.pp_size, c.async_scheduling),
    )
    ec_cpu_bytes = _EC_REGION_IMAGES * estimate_bytes_per_image()

    urls = stress_image_urls(_NUM_DISTINCT_IMAGES)
    batches = _build_batches(urls)

    llm = LLM(
        **build_llm_kwargs(
            tp_size=case.tp_size,
            pp_size=case.pp_size,
            async_scheduling=case.async_scheduling,
            use_connector=True,
            ec_cpu_bytes=ec_cpu_bytes,
            max_num_batched_tokens=_ENCODER_CACHE_TOKENS,
            max_pixels=stress_max_pixels(),
        )
    )
    try:
        connector = get_scheduler_ec_connector(llm)
        assert connector is not None, (
            "expected the scheduler-side ECCPUConnector to be reachable "
            "in-process; check VLLM_ENABLE_V1_MULTIPROCESSING wiring"
        )

        # Verify the reload precondition against the capacity the engine
        # actually resolved, before spending the whole churn run to find out.
        encoder_cache_size = get_encoder_cache_size(llm)
        region_blocks = _EC_REGION_IMAGES * estimate_embeds_per_image()
        print(
            f"[ec_cpu_offload][stress] encoder_cache_size={encoder_cache_size} "
            f"embeds, ec_region={region_blocks} blocks"
        )
        assert encoder_cache_size is not None
        assert encoder_cache_size < region_blocks, (
            f"CPU region holds {region_blocks} blocks but the GPU encoder cache "
            f"resolved to {encoder_cache_size} embeds; the region evicts first, "
            f"so an encoder-cache miss can never be served from CPU. "
            f"compute_mm_encoder_budget floors the cache at the model's own max "
            f"tokens per image, so max_num_batched_tokens="
            f"{_ENCODER_CACHE_TOKENS} only applies once max_pixels caps it."
        )

        counts = observe_ec_transfers(connector)

        deadline = time.monotonic() + _RUN_TIMEOUT_S
        for batch_idx, batch in enumerate(batches):
            assert time.monotonic() < deadline, (
                f"stress run exceeded {_RUN_TIMEOUT_S}s before batch {batch_idx}"
            )
            results = llm.chat(batch, sampling_params=default_sampling_params())
            assert len(results) == len(batch)
            for req_idx, result in enumerate(results):
                text = result.outputs[0].text
                assert text, (
                    f"batch {batch_idx} request {req_idx} produced an empty "
                    "output under eviction churn"
                )
            # Settle this batch's transfers so the next batch's re-requests
            # meet ready CPU entries instead of racing them.
            drain_pending_push_work(llm, connector, timeout_s=60.0)

        print(f"[ec_cpu_offload][stress] transfers={counts}")

        # Confirm no save/load was left permanently pinned/not-ready.
        assert not connector.has_pending_push_work()

        assert counts.saves_completed > 0, (
            f"no save was ever confirmed, so the churn never reached the CPU "
            f"region ({counts})"
        )
        assert counts.loads_completed > 0, (
            f"no CPU->GPU reload completed, so every re-requested image was "
            f"recomputed and the reload path went unexercised ({counts}). The "
            f"CPU region ({_EC_REGION_IMAGES} images) must stay larger than "
            f"the GPU encoder cache ({_ENCODER_CACHE_TOKENS} embeds) for a "
            f"reload to be possible."
        )
    finally:
        shutdown_llm(llm)
