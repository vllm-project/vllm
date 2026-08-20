# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E correctness + perf-logging test for the EC CPU offload connector
(`ECCPUConnector`).

For each point in the detected `max_concurrent_batches` matrix (async
scheduling on/off, plus pipeline-parallel cases if multiple GPUs are
available), runs the same multimodal + text batches once without the
connector (baseline) and once with it, and asserts the outputs are
byte-for-byte identical under greedy decoding. Wall-clock time for each run
is printed for a with/without-connector performance comparison -- no
threshold is asserted, since perf comparisons are inherently noisy; only
correctness is a hard pass/fail signal here.

Byte-for-byte equality requires `VLLM_BATCH_INVARIANT`: serving an encoder
output from the CPU region takes the `external_load_encoder_input` path
instead of running the encoder, which changes which requests batch together
and hence reduction order, so greedy decoding can pick a different token
without either run being wrong. Batch-invariant mode removes that freedom
and needs compute capability >= 9.0, so on older GPUs the equality check is
reported as skipped while the transfer assertions still run.

The run is sized so the CPU->GPU reload path actually executes, and the
assertions require it: without a reload the equality check would only prove
that offloading is harmless, not that reloaded embeddings are correct.
Reaching that path needs three capacities set against each other -- see
`_MAX_PIXELS`, `_ENCODER_CACHE_TOKENS` and `_EC_REGION_IMAGES`.
"""

import json
import os

import pytest

from tests.utils import create_new_process_for_each_test
from vllm import LLM
from vllm.platforms import current_platform

from ._ec_cpu_offload_helpers import (
    E2E_IMAGE_NAMES,
    IMAGE_SIZE,
    Timer,
    build_llm_kwargs,
    default_sampling_params,
    detect_concurrency_matrix,
    drain_pending_push_work,
    estimate_bytes_per_image,
    estimate_embeds_per_image,
    get_encoder_cache_size,
    get_scheduler_ec_connector,
    image_url,
    make_image_message,
    make_text_message,
    observe_ec_transfers,
    shutdown_llm,
)

pytestmark = [
    pytest.mark.skipif(
        current_platform.device_count() < 1,
        reason="EC CPU offload e2e test requires a GPU",
    ),
    # Each test body already runs in its own forked child (see
    # @create_new_process_for_each_test below), so GPU memory is reclaimed
    # by that child exiting. The global cleanup_fixture in tests/conftest.py
    # would otherwise run cleanup_dist_env_and_memory() in the *main* pytest
    # process between cases, initializing a CUDA context there and breaking
    # the next case's own worker fork (any multi-GPU case forks fresh
    # workers per LLM()).
    pytest.mark.skip_global_cleanup,
]


# Caps the model's reported max tokens per image at what this test actually
# sends. compute_mm_encoder_budget floors the encoder cache at the largest
# tokens-per-item over active modalities, so without this cap the model default
# swamps _ENCODER_CACHE_TOKENS and the cache never evicts. Equal to IMAGE_SIZE's
# area, so no image is resized. Works only alongside the zeroed video limit in
# build_llm_kwargs, which removes the other contender for that floor.
_MAX_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1]

# GPU encoder cache capacity in encoder embeddings, set via
# max_num_batched_tokens (SchedulerConfig.encoder_cache_size derives from it).
# Held below the 4-image working set -- 4 x 144 = 576 embeds at IMAGE_SIZE,
# since 336 // 14 // 2 = 12 and 12 x 12 = 144 -- so batch 1 alone overflows it
# and batch 2's reuse is a guaranteed encoder-cache miss. Kept above the 288
# embeds of the two-image request so that request always fits in one step.
_ENCODER_CACHE_TOKENS = 512

# CPU region capacity in images. Must exceed the GPU encoder cache above:
# a reload is only possible for an entry the GPU cache dropped while the CPU
# region still holds it, so a region smaller than the GPU cache can never
# serve a miss. Six images covers the whole four-image working set.
_EC_REGION_IMAGES = 6

# Passed to both LLM() instances as kernel_config. None (the default) leaves
# vLLM on "auto"; set VLLM_TEST_EC_KERNEL_CONFIG to a JSON object such as
# '{"linear_backend": "deep_gemm"}' when MODEL is swapped for an FP8 model on
# an image without the NVRTC dev header, since the default flashinfer_*
# linear backends JIT-compile their block-scaled GEMM on first forward and
# abort on a missing nvrtc.h there.
_KERNEL_CONFIG = (
    json.loads(os.environ["VLLM_TEST_EC_KERNEL_CONFIG"])
    if "VLLM_TEST_EC_KERNEL_CONFIG" in os.environ
    else None
)

# Passed to both LLM() instances as load_format. None (the default) leaves
# vLLM on "auto"; set VLLM_TEST_EC_LOAD_FORMAT="fastsafetensors" at tp_size >
# 1 so each TP rank reads only its own subset of checkpoint files and
# redistributes over NCCL, instead of every rank re-reading the whole
# checkpoint. Needs `pip install 'vllm[fastsafetensors]'`.
_LOAD_FORMAT = os.environ.get("VLLM_TEST_EC_LOAD_FORMAT")


def _batch_invariant_supported() -> bool:
    """Whether VLLM_BATCH_INVARIANT can be relied on here.

    init_batch_invariance() neither raises nor warns on unsupported hardware,
    so the documented compute-capability floor has to be checked here.
    """
    return current_platform.has_device_capability(90)


def _build_batches() -> tuple[list, list]:
    urls = [image_url(name) for name in E2E_IMAGE_NAMES]

    batch1 = [
        make_image_message(urls[0], text="What's in this image?"),
        make_text_message("What is the capital of France?"),
        make_image_message(urls[1], text="Describe this image briefly."),
        make_image_message(urls[2], urls[3], text="Compare these two images."),
        make_text_message("Explain photosynthesis in one sentence."),
    ]
    # Batch 2 re-uses the same images (save->reload cache-hit path) plus one
    # further multi-image combination.
    batch2 = [
        make_image_message(urls[0], text="What color is the main object?"),
        make_image_message(urls[1], text="What season does this suggest?"),
        make_image_message(urls[2], urls[3], text="What do these have in common?"),
        make_text_message("Name three prime numbers."),
    ]
    return batch1, batch2


def _run_batches(
    llm: LLM, batches: tuple[list, list], connector: object | None = None
) -> tuple[list[list[str]], list[float]]:
    """Run each batch, optionally settling connector transfers in between.

    When `connector` is given, in-flight saves are drained after each batch so
    the next batch's reuse finds ready CPU entries rather than racing them.
    Draining happens outside the Timer: it is test synchronization, not work
    the engine would do here on its own.
    """
    outputs = []
    timings = []
    for batch in batches:
        with Timer() as t:
            results = llm.chat(batch, sampling_params=default_sampling_params())
        outputs.append([r.outputs[0].text for r in results])
        timings.append(t.elapsed)
        if connector is not None:
            drain_pending_push_work(llm, connector, timeout_s=60.0)
    return outputs, timings


@create_new_process_for_each_test(method="spawn")
@pytest.mark.parametrize(
    "case",
    detect_concurrency_matrix(),
    ids=lambda c: f"tp{c.tp_size}_pp{c.pp_size}_async{c.async_scheduling}",
)
def test_ec_cpu_offload_correctness_and_perf(case) -> None:
    # Must be set before BOTH engines: it is what makes their outputs
    # comparable at all, since the connector run batches differently once it
    # starts serving encoder outputs from CPU instead of computing them.
    batch_invariant = _batch_invariant_supported()
    if batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"

    batches = _build_batches()
    ec_cpu_bytes = _EC_REGION_IMAGES * estimate_bytes_per_image()

    llm = LLM(
        **build_llm_kwargs(
            tp_size=case.tp_size,
            pp_size=case.pp_size,
            async_scheduling=case.async_scheduling,
            use_connector=False,
            max_num_batched_tokens=_ENCODER_CACHE_TOKENS,
            max_pixels=_MAX_PIXELS,
            kernel_config=_KERNEL_CONFIG,
            load_format=_LOAD_FORMAT,
        )
    )
    try:
        baseline_max_concurrent_batches = (
            llm.llm_engine.vllm_config.max_concurrent_batches
        )
        print(
            f"[ec_cpu_offload][{case}] "
            f"max_concurrent_batches={baseline_max_concurrent_batches}"
        )
        baseline_outputs, baseline_timings = _run_batches(llm, batches)
    finally:
        shutdown_llm(llm)

    # Only the connector run needs the in-process EngineCore client, which is
    # what makes the scheduler-side delegate reachable for the transfer-count
    # assertions. Deliberately not set for the baseline above: in-process means
    # the pp=1 worker holds its GPU allocation in *this* process, which
    # shutdown_llm cannot reclaim (it relies on worker subprocesses exiting),
    # and the second engine would then fail to acquire its memory. Set directly
    # rather than via `monkeypatch`: this body runs in a disposable spawned
    # subprocess, so there is nothing to restore, and a live MonkeyPatch
    # fixture would not survive being pickled across the spawn boundary.
    # The resulting CUDA context in this process makes the pp>1 cases spawn
    # their workers instead of forking -- get_mp_context() calls
    # _maybe_force_spawn() per executor init, which switches on
    # cuda_is_initialized().
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    llm = LLM(
        **build_llm_kwargs(
            tp_size=case.tp_size,
            pp_size=case.pp_size,
            async_scheduling=case.async_scheduling,
            use_connector=True,
            ec_cpu_bytes=ec_cpu_bytes,
            max_num_batched_tokens=_ENCODER_CACHE_TOKENS,
            max_pixels=_MAX_PIXELS,
            kernel_config=_KERNEL_CONFIG,
            load_format=_LOAD_FORMAT,
        )
    )
    try:
        # NOT asserted equal to the baseline value: the two runs can resolve
        # async_scheduling+pp>1 to different concurrency
        # (VllmConfig.max_concurrent_batches) depending on which model runner
        # each config selects. That's current, intentional vLLM behavior, not
        # something this connector controls or a bug to assert against.
        connector_max_concurrent_batches = (
            llm.llm_engine.vllm_config.max_concurrent_batches
        )
        print(
            f"[ec_cpu_offload][{case}] baseline max_concurrent_batches="
            f"{baseline_max_concurrent_batches}, connector max_concurrent_batches="
            f"{connector_max_concurrent_batches}"
        )
        connector = get_scheduler_ec_connector(llm)
        assert connector is not None, (
            "expected the scheduler-side ECCPUConnector to be reachable "
            "in-process; check VLLM_ENABLE_V1_MULTIPROCESSING wiring"
        )

        # Check the reload precondition up front, against the capacity the
        # engine actually resolved rather than the one requested. Asserting it
        # here reports the offending number directly; discovering it via a zero
        # load count after both batches would not say which cap was wrong.
        encoder_cache_size = get_encoder_cache_size(llm)
        working_set = len(E2E_IMAGE_NAMES) * estimate_embeds_per_image()
        print(
            f"[ec_cpu_offload][{case}] encoder_cache_size={encoder_cache_size} "
            f"embeds, working_set={working_set} embeds, "
            f"ec_region={_EC_REGION_IMAGES * estimate_embeds_per_image()} blocks"
        )
        assert encoder_cache_size is not None
        assert encoder_cache_size < working_set, (
            f"GPU encoder cache resolved to {encoder_cache_size} embeds but the "
            f"working set is only {working_set}, so no image is ever evicted "
            f"and no reload can be dispatched. compute_mm_encoder_budget floors "
            f"the cache at the model's own max tokens per image, so "
            f"max_num_batched_tokens={_ENCODER_CACHE_TOKENS} only takes effect "
            f"when max_pixels={_MAX_PIXELS} caps that floor below it."
        )
        region_blocks = _EC_REGION_IMAGES * estimate_embeds_per_image()
        assert encoder_cache_size < region_blocks, (
            f"CPU region holds {region_blocks} blocks but the GPU encoder cache "
            f"holds {encoder_cache_size} embeds; the region evicts first, so an "
            f"encoder-cache miss can never be served from CPU"
        )

        counts = observe_ec_transfers(connector)

        connector_outputs, connector_timings = _run_batches(
            llm, batches, connector=connector
        )
        print(f"[ec_cpu_offload][{case}] transfers={counts}")
    finally:
        shutdown_llm(llm)

    # Offloading is only meaningfully exercised if data made the full round
    # trip. Without these the equality check below would still pass on a
    # connector that saved nothing and reloaded nothing.
    assert counts.saves_completed >= len(E2E_IMAGE_NAMES), (
        f"expected every distinct image to be saved to CPU and confirmed, got "
        f"{counts.saves_completed} completed saves for "
        f"{len(E2E_IMAGE_NAMES)} images ({counts})"
    )
    assert counts.loads_dispatched > 0, (
        f"no CPU->GPU reload was ever dispatched, so batch 2 recomputed every "
        f"image instead of reusing the offloaded copies ({counts}). The GPU "
        f"encoder cache ({_ENCODER_CACHE_TOKENS} embeds) must be smaller than "
        f"the working set and the CPU region ({_EC_REGION_IMAGES} images) "
        f"larger than it."
    )
    assert counts.loads_completed > 0, (
        f"reloads were dispatched but none completed, so the worker never "
        f"confirmed a CPU->GPU copy ({counts})"
    )

    if not batch_invariant:
        print(
            f"[ec_cpu_offload][{case}] SKIPPING the output-equality check: "
            "batch-invariant mode needs compute capability >= 9.0, and without "
            "it a reload-induced change in batch composition can legitimately "
            "flip a greedily decoded token. Transfer assertions above still ran."
        )
    for batch_idx, (baseline_batch, connector_batch) in enumerate(
        zip(baseline_outputs, connector_outputs)
    ):
        if not batch_invariant or connector_batch == baseline_batch:
            continue
        # Report the first differing pair verbatim. How they differ is the
        # diagnostic: a shared prefix that then drifts points at numerics,
        # whereas divergence from the first token points at the reloaded
        # embeddings being wrong rather than merely reordered.
        diffs = [
            (i, b, c)
            for i, (b, c) in enumerate(zip(baseline_batch, connector_batch))
            if b != c
        ]
        idx, base_text, conn_text = diffs[0]
        shared = 0
        for shared, (bc, cc) in enumerate(zip(base_text, conn_text)):
            if bc != cc:
                break
        raise AssertionError(
            f"batch {batch_idx} outputs diverged between baseline and "
            f"connector runs for {case}: {len(diffs)}/{len(baseline_batch)} "
            f"requests differ; transfers={counts}\n"
            f"  first differing request index {idx}, identical for the first "
            f"{shared} chars\n"
            f"  baseline : {base_text!r}\n"
            f"  connector: {conn_text!r}"
        )

    for batch_idx, (base_t, conn_t) in enumerate(
        zip(baseline_timings, connector_timings)
    ):
        print(
            f"[ec_cpu_offload][{case}] batch {batch_idx}: "
            f"baseline={base_t:.3f}s connector={conn_t:.3f}s"
        )
