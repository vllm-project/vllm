# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared plumbing for the EC CPU offload connector (ECCPUConnector)
e2e and stress tests.

All images are resized to a single fixed resolution (`IMAGE_SIZE`) so that
`estimate_bytes_per_image` (computed once, ahead of engine startup) is a
reliable proxy for the actual encoder-cache footprint every test image will
occupy.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from vllm.assets.image import ImageAsset
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.multimodal.utils import encode_image_url
from vllm.sampling_params import SamplingParams

# Overridable so this suite can be pointed at e.g. an FP8 model that needs
# VLLM_TEST_EC_KERNEL_CONFIG set alongside it (see build_llm_kwargs).
MODEL = os.environ.get("VLLM_TEST_EC_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
IMAGE_SIZE = (336, 336)
SEED = 42
MAX_TOKENS = 64

# 4 visually distinct assets for the correctness/perf e2e test.
E2E_IMAGE_NAMES = ["stop_sign", "cherry_blossom", "hato", "27-500x500"]

# (name, ext) pairs with a confirmed-working remote file -- not every
# ImageAssetName has a plain "<name>.jpg" on S3 (ImageAsset.pil_image
# defaults to "jpg"), and one entry (the Venn diagram) actually lives at a
# double extension ("...svg.png") that doesn't map through ImageAsset
# cleanly, so it's deliberately excluded rather than guessed at.
_STRESS_CATALOG = [
    ("stop_sign", "jpg"),
    ("cherry_blossom", "jpg"),
    ("hato", "jpg"),
    ("27-500x500", "jpg"),
    ("2560px-Gfp-wisconsin-madison-the-nature-boardwalk", "jpg"),
    ("Grayscale_8bits_palette_sample_image", "png"),
    ("RGBA_comp", "png"),
]
# Cycled across several resolutions to synthesize enough distinct
# encoder-cache entries for the stress test from this modest catalog.
_STRESS_SIZES = [(336, 336), (448, 448), (224, 224), (280, 280)]


def _load_image(name: str, ext: str, size: tuple[int, int]) -> Image.Image:
    return ImageAsset(name).pil_image_ext(ext=ext).convert("RGB").resize(size)


def image_url(name: str, size: tuple[int, int] = IMAGE_SIZE, ext: str = "jpg") -> str:
    """Base64 data URL for a distinct (name, ext, size) triple.

    Resizing changes the encoded bytes, so varying size is a cheap way to
    synthesize extra distinct mm_hashes from a small image catalog.
    """
    return encode_image_url(_load_image(name, ext, size))


def stress_max_pixels() -> int:
    """Largest image area used by the stress catalog, for `max_pixels`."""
    return max(w * h for w, h in _STRESS_SIZES)


def stress_image_urls(count: int) -> list[str]:
    """Return `count` distinct image URLs, cycling asset x resolution."""
    variants = [
        (name, ext, size) for size in _STRESS_SIZES for name, ext in _STRESS_CATALOG
    ]
    assert len(variants) >= count, (
        f"catalog only yields {len(variants)} distinct variants, need {count}"
    )
    return [image_url(name, size, ext) for name, ext, size in variants[:count]]


@dataclass(frozen=True)
class ConcurrencyCase:
    pp_size: int
    async_scheduling: bool
    tp_size: int = 1

    @property
    def num_gpus(self) -> int:
        return self.tp_size * self.pp_size


def detect_concurrency_matrix() -> list[ConcurrencyCase]:
    """Concurrency cases to exercise, based on GPUs actually available.

    Always includes tp=1/pp=1 with async scheduling off/on. Adds the tp=2,
    pp=2, and tp=4 variants only when enough GPUs exist for them (a case
    needs tp_size * pp_size devices), so this never requires multi-GPU to
    run.

    TP and PP are both covered because the connector treats them differently:
    only tp_rank 0 saves (all TP ranks hold identical encoder output, so one
    writer suffices) while every rank loads, whereas PP confines the encoder
    to the first stage entirely.

    Deliberately does NOT predict the resulting `max_concurrent_batches`
    here: that value also depends on `VllmConfig.use_v2_model_runner`
    (env-var/model-dependent, see vllm/config/vllm.py), which this helper
    has no way to evaluate ahead of constructing the real `LLM`. Each test
    case reads the actual value off the constructed engine instead of
    asserting against a value re-derived (and liable to drift out of sync)
    from `VllmConfig.max_concurrent_batches`'s own formula.
    """
    from vllm.platforms import current_platform

    num_gpus = current_platform.device_count()
    candidates = [
        ConcurrencyCase(tp_size=tp, pp_size=pp, async_scheduling=async_sched)
        for tp, pp in ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1))
        for async_sched in (False, True)
    ]
    cases = [c for c in candidates if c.num_gpus <= num_gpus]

    print(
        f"[ec_cpu_offload] detected {num_gpus} GPU(s); concurrency matrix: "
        + ", ".join(
            f"(tp={c.tp_size}, pp={c.pp_size}, async_scheduling={c.async_scheduling})"
            for c in cases
        )
    )
    return cases


def estimate_embeds_per_image(
    model_name: str = MODEL, image_size: tuple[int, int] = IMAGE_SIZE
) -> int:
    """Visual tokens one image of `image_size` occupies in the encoder cache.

    Derived from the vision tower's patch and merge size, matching how the
    scheduler counts encoder embeddings. Also the number of EC blocks the
    image occupies, since one block holds one visual token.
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    vision_config = getattr(hf_config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", 14) if vision_config else 14
    merge_size = getattr(vision_config, "spatial_merge_size", 2) if vision_config else 2
    return max(
        1,
        (image_size[0] // patch_size // merge_size)
        * (image_size[1] // patch_size // merge_size),
    )


def get_encoder_cache_size(llm: Any) -> int | None:
    """Resolved GPU encoder cache capacity, in encoder embeddings.

    This is the value `compute_mm_encoder_budget` actually settled on, which
    is floored at the model's own maximum tokens per image and so can be far
    above `max_num_batched_tokens`. Requires an in-process engine core; returns
    None otherwise.
    """
    engine_core = llm.llm_engine.engine_core
    inner = getattr(engine_core, "engine_core", None)
    if inner is None:
        return None
    return inner.scheduler.encoder_cache_manager.cache_size


def estimate_bytes_per_image(
    model_name: str = MODEL, image_size: tuple[int, int] = IMAGE_SIZE
) -> int:
    """Approximate CPU-offload bytes for one image's encoder-cache entry.

    Mirrors `create_ec_shared_region`'s hidden-dim formula (out_hidden_size
    times 1 + deepstack layers, falling back to plain hidden size) times an
    estimated visual-token count for `image_size`, derived from the vision
    tower's patch/merge size. This is an estimate, not an exact figure --
    sizing only needs to be roughly right to force eviction with a handful
    of distinct images.
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    vision_config = getattr(hf_config, "vision_config", None)

    out_hidden_size = (
        getattr(vision_config, "out_hidden_size", None) if vision_config else None
    )
    deepstack_indexes = (
        getattr(vision_config, "deepstack_visual_indexes", None)
        if vision_config
        else None
    )
    if out_hidden_size is not None and deepstack_indexes:
        hidden_dim = out_hidden_size * (1 + len(deepstack_indexes))
    elif out_hidden_size is not None:
        hidden_dim = out_hidden_size
    else:
        hidden_dim = getattr(hf_config, "hidden_size", 4096)

    tokens_per_image = estimate_embeds_per_image(model_name, image_size)

    dtype_name = getattr(hf_config, "torch_dtype", None)
    dtype = (
        getattr(torch, str(dtype_name), torch.bfloat16)
        if dtype_name
        else (torch.bfloat16)
    )
    element_size = torch.empty(0, dtype=dtype).element_size()

    return hidden_dim * element_size * tokens_per_image


def build_llm_kwargs(
    *,
    pp_size: int,
    async_scheduling: bool,
    tp_size: int = 1,
    use_connector: bool,
    ec_cpu_bytes: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_pixels: int | None = None,
    kernel_config: dict[str, Any] | None = None,
    load_format: str | None = None,
) -> dict[str, Any]:
    """Assemble `LLM()` kwargs shared by the baseline and connector runs.

    Both capacity arguments must be passed identically to the two runs of a
    with/without-connector comparison, so each sees the same batching.

    Args:
        max_num_batched_tokens: Also sets the GPU encoder cache capacity
            (`SchedulerConfig.encoder_cache_size` derives from it).
        max_pixels: Caps the model's reported maximum tokens per image.
            `compute_mm_encoder_budget` floors the encoder cache at the
            largest tokens-per-item across active modalities, so without a cap
            the model's default (tens of thousands of tokens for Qwen2-VL)
            overrides `max_num_batched_tokens` entirely and the cache never
            evicts. Set it to the largest image area the test actually sends;
            areas at or below the cap are not resized, so this does not change
            the images. Capping images is necessary but not sufficient -- see
            the zeroed video limit below, which removes the other contender
            for that floor.
        kernel_config: Passed through as `LLM(kernel_config=...)`; None leaves
            vLLM on "auto". Needed for FP8 models on images without the NVRTC
            dev header: the default flashinfer_* linear backends JIT-compile
            their block-scaled GEMM on the first forward and abort on a
            missing nvrtc.h, while `{"linear_backend": "deep_gemm"}` (or
            "cutlass"/"triton") uses a prebuilt kernel instead.
        load_format: Passed through as `LLM(load_format=...)`; None leaves
            vLLM on "auto". Use "fastsafetensors" at tp_size > 1 to have each
            TP rank read only its own subset of checkpoint files and
            redistribute over NCCL, instead of every rank re-reading the
            whole checkpoint. Needs the extra installed
            (`pip install 'vllm[fastsafetensors]'`).
    """
    kwargs: dict[str, Any] = dict(
        model=MODEL,
        max_model_len=4096,
        max_num_seqs=32,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        async_scheduling=async_scheduling,
        # video is zeroed, not merely unused: compute_mm_encoder_budget floors
        # the encoder cache at max() over *every* active modality's tokens per
        # item, so this model's video figure would set the floor even though
        # these tests only send images. A zero limit drops video from
        # MultiModalBudget's active modalities entirely.
        limit_mm_per_prompt={"image": 4, "video": 0},
        seed=SEED,
    )
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if max_pixels is not None:
        kwargs["mm_processor_kwargs"] = {"max_pixels": max_pixels}
    if kernel_config is not None:
        kwargs["kernel_config"] = kernel_config
    if load_format is not None:
        kwargs["load_format"] = load_format
    if use_connector:
        assert ec_cpu_bytes is not None
        kwargs["ec_transfer_config"] = {
            "ec_connector": "ECCPUConnector",
            "ec_role": "ec_both",
            "ec_connector_extra_config": {"ec_cpu_bytes": ec_cpu_bytes},
        }
    return kwargs


def make_image_message(*urls: str, text: str) -> list[ChatCompletionMessageParam]:
    """User message with `text` ahead of the images.

    Text first so that re-sending an image under a different prompt still
    requires its encoder output. With the image first, a repeat request shares
    its whole prefix -- chat template plus image placeholder tokens -- with the
    earlier one; cached KV over that span puts the image outside
    `get_mm_features_in_window`, and the scheduler then never asks for an
    encoder output at all, offloaded or otherwise. Differing leading text
    diverges the prefix before the placeholders, so the encoder output is
    genuinely needed and can be served from the CPU region.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in urls)
    return [{"role": "user", "content": content}]


def make_text_message(text: str) -> list[ChatCompletionMessageParam]:
    return [{"role": "user", "content": text}]


def default_sampling_params() -> SamplingParams:
    return SamplingParams(temperature=0.0, seed=SEED, max_tokens=MAX_TOKENS)


def shutdown_llm(llm: Any) -> None:
    """Tear down an `LLM`'s engine core (and its worker subprocesses).

    Deliberately does NOT call `cleanup_dist_env_and_memory()`: that helper's
    `torch.accelerator.empty_cache()` call initializes a CUDA context in
    whichever process calls it, which then breaks the *next* `LLM()`
    construction's worker fork in this same process/test (pp>1 needs to fork
    fresh workers, and forking a process with an already-initialized CUDA
    context raises "Cannot re-initialize CUDA in forked subprocess"). GPU
    memory is reclaimed via the killed worker subprocesses on `shutdown()`,
    not via this driver process.
    """
    with contextlib.suppress(Exception):
        llm.llm_engine.engine_core.shutdown()
    del llm


def get_scheduler_ec_connector(llm: Any) -> Any | None:
    """Reach the scheduler-side ECCPUConnector delegate for direct probing.

    Only works when the engine core runs in-process (client is
    `InprocClient`, i.e. `VLLM_ENABLE_V1_MULTIPROCESSING=0`); returns None
    otherwise so callers can skip the probe rather than crash.
    """
    engine_core = llm.llm_engine.engine_core
    inner = getattr(engine_core, "engine_core", None)
    if inner is None:
        return None
    return inner.scheduler.get_ec_connector()


@dataclass
class ECTransferCounts:
    """Running tally of what the connector dispatched and what completed."""

    saves_dispatched: int = 0
    loads_dispatched: int = 0
    saves_completed: int = 0
    # Distinct load transfers reported complete. Each participating rank
    # reports the same transfer id, so the ids are de-duplicated rather than
    # counted per report.
    completed_load_ids: set[int] = field(default_factory=set)

    @property
    def loads_completed(self) -> int:
        return len(self.completed_load_ids)


def observe_ec_transfers(connector: Any) -> ECTransferCounts:
    """Tally save/load traffic by wrapping the scheduler delegate's hooks.

    `build_connector_meta` is what dispatches work to the worker and
    `update_connector_output` is what receives the worker's completion
    report, so wrapping the pair on the live delegate instance observes both
    ends of every transfer without instrumenting the connector itself.

    Args:
        connector: The scheduler-side `ECCPUConnector` from
            `get_scheduler_ec_connector`.

    Returns:
        A counter object updated in place as the engine steps.
    """
    delegate = connector.connector_scheduler
    assert delegate is not None, "expected a scheduler-role connector"
    counts = ECTransferCounts()

    build_meta = delegate.build_connector_meta
    update_output = delegate.update_connector_output

    def counting_build_connector_meta(scheduler_output: Any) -> Any:
        meta = build_meta(scheduler_output)
        counts.saves_dispatched += len(meta.saves)
        counts.loads_dispatched += len(meta.loads)
        return meta

    def counting_update_connector_output(connector_output: Any) -> None:
        worker_meta = connector_output.ec_connector_worker_meta
        if worker_meta is not None:
            counts.saves_completed += len(worker_meta.completed_saves)
            counts.completed_load_ids.update(worker_meta.completed_loads)
        update_output(connector_output)

    delegate.build_connector_meta = counting_build_connector_meta
    delegate.update_connector_output = counting_update_connector_output
    return counts


def drain_pending_push_work(
    llm: Any, connector: Any, *, timeout_s: float = 30.0, poll_interval: float = 0.05
) -> None:
    """Step the engine until the connector reports no in-flight transfers.

    `llm.chat()`/`llm.generate()` stop stepping the engine once every
    *request* has finished, which can race ahead of the connector's
    completion-report draining. Call this after a batch to force the extra
    steps needed to observe `has_pending_push_work() -> False` reliably.
    """
    deadline = time.monotonic() + timeout_s
    while connector.has_pending_push_work():
        if time.monotonic() > deadline:
            raise TimeoutError("EC connector push work never drained within timeout")
        llm.llm_engine.step()
        time.sleep(poll_interval)


class Timer:
    """Minimal wall-clock timer for perf logging (`with Timer() as t: ...`)."""

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
