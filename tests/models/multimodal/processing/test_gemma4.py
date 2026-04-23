# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorSenderCache

from ....conftest import ImageTestAssets
from ...utils import build_model_context

# TODO: to be updated to "google/gemma-4-e2b-it" once the models are available
GEMMA4_MODEL_ID = "google/gemma-4-E2B-it"


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_limit_mm_per_prompt(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that limit_mm_per_prompt accurately restricts multiple images."""
    # We only allow 1 image
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Provide 2 images in the prompt
    prompt = "<image><image>"
    # image_assets usually has multiple images
    images = [asset.pil_image for asset in image_assets][:2]
    if len(images) < 2:
        images = [images[0], images[0]]

    mm_data = {"image": images}

    # Expect ValueError when exceeding limit
    with pytest.raises(ValueError, match="At most 1 image"):
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )


# ---------------------------------------------------------------------------
# max_soft_tokens="auto" — early canonicalization + cache coherence
# ---------------------------------------------------------------------------

# Sizes chosen so they fall into distinct _SUPPORTED_SOFT_TOKENS buckets
# at the default Gemma 4 geometry (patch_size=16, pooling_kernel_size=3 =>
# 2304 px per token). See test_gemma4_auto_budget.py for bucket math.
_SIZE_SMALL_70 = (100, 75)  # area 7,500 → 70
_SIZE_LARGE_1120 = (1920, 1080)  # area 2,073,600 → 1120


def _make_image(size: tuple[int, int], seed: int) -> Image.Image:
    w, h = size
    # Deterministic per-seed content so hashes are stable across runs
    # within a single test invocation but diverge per-seed so we exercise
    # multi-image paths without relying on PIL identity dedup.
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _build_processor(
    model_id: str,
    *,
    mm_processor_kwargs: dict | None = None,
    cache: object | None = "default",
):
    """Build a processor + its context. ``cache="default"`` uses whatever
    the registry picks; pass ``cache=None`` to disable caching; pass an
    explicit cache instance to use it."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs or {},
        limit_mm_per_prompt={"image": 4},
        mm_processor_cache_gb=1,
    )
    if cache == "default":
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    else:
        processor = MULTIMODAL_REGISTRY.create_processor(
            ctx.model_config, cache=cache
        )
    return ctx, processor


def _image_prompt(processor, n_images: int) -> str:
    """Build a prompt containing ``n_images`` copies of the processor's
    actual image token — matches what Gemma 4's chat template inserts."""
    token = processor.info.get_hf_processor().image_token
    return token * n_images


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_auto_hash_equals_explicit_for_small_image(model_id: str):
    """Server with ``max_soft_tokens="auto"`` must yield the same
    mm_hashes as ``max_soft_tokens=70`` when every image in the request
    resolves to 70. If the sentinel is not canonicalized before hashing,
    the "auto" request's hash would differ from the explicit-70 hash
    for the same image — the exact cache-poisoning failure mode."""
    _, proc_auto = _build_processor(model_id, cache=None)
    _, proc_explicit = _build_processor(model_id, cache=None)

    img = _make_image(_SIZE_SMALL_70, seed=0)
    mm_data = {"image": [img]}

    # vLLM computes hashes inside processor(...); compare the resulting
    # mm_hashes across the two configurations.
    out_auto = proc_auto(
        _image_prompt(proc_auto, 1),
        mm_items=proc_auto.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"max_soft_tokens": "auto"},
    )
    out_explicit = proc_explicit(
        _image_prompt(proc_explicit, 1),
        mm_items=proc_explicit.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"max_soft_tokens": 70},
    )

    # Same effective config → same per-item hashes.
    assert out_auto["mm_hashes"] == out_explicit["mm_hashes"], (
        "auto resolved to 70 must produce the same mm_hashes as "
        "explicit max_soft_tokens=70 — otherwise the sentinel is "
        "leaking into the cache key."
    )


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_auto_hash_differs_when_context_changes(model_id: str):
    """The same PIL image submitted (a) alone and (b) batched with a
    larger image must yield different mm_hashes under
    ``max_soft_tokens="auto"``, because the effective budget differs
    (70 vs 1120).  If hashes match, the mm-processor cache will serve
    stale 70-token tensors for a request that actually needs 1120."""
    _, proc = _build_processor(model_id, cache=None)

    small = _make_image(_SIZE_SMALL_70, seed=1)
    large = _make_image(_SIZE_LARGE_1120, seed=2)

    out_alone = proc(
        _image_prompt(proc, 1),
        mm_items=proc.info.parse_mm_data({"image": [small]}),
        hf_processor_mm_kwargs={"max_soft_tokens": "auto"},
    )
    out_batched = proc(
        _image_prompt(proc, 2),
        mm_items=proc.info.parse_mm_data({"image": [small, large]}),
        hf_processor_mm_kwargs={"max_soft_tokens": "auto"},
    )

    # The small image's hash when alone (budget=70) must not collide
    # with its hash when batched with a large image (budget=1120).
    small_hash_alone = out_alone["mm_hashes"]["image"][0]
    small_hash_batched = out_batched["mm_hashes"]["image"][0]
    assert small_hash_alone != small_hash_batched, (
        "Same image in different batch compositions must produce "
        "different mm_hashes under auto; otherwise a cache hit from "
        "the alone-request will serve stale 70-token tensors for a "
        "batched request that needs 1120."
    )


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
@pytest.mark.parametrize("order", ["alone_then_batched", "batched_then_alone"])
def test_auto_cache_coherence_across_requests(model_id: str, order: str):
    """With mm_processor_cache on, sequencing "alone then batched" and
    "batched then alone" must both produce prompt_token_ids matching a
    cache-less baseline.  A stale cache entry from the prior request
    would otherwise corrupt the second one."""
    _, baseline = _build_processor(model_id, cache=None)

    sender_cache = MultiModalProcessorSenderCache(baseline.info.ctx.model_config)
    _, cached = _build_processor(model_id, cache=sender_cache)

    small = _make_image(_SIZE_SMALL_70, seed=3)
    large = _make_image(_SIZE_LARGE_1120, seed=4)

    def run(proc, images, prompt):
        return proc(
            prompt,
            mm_items=proc.info.parse_mm_data({"image": images}),
            hf_processor_mm_kwargs={"max_soft_tokens": "auto"},
        )["prompt_token_ids"]

    p1 = _image_prompt(baseline, 1)
    p2 = _image_prompt(baseline, 2)
    baseline_alone = run(baseline, [small], p1)
    baseline_batched = run(baseline, [small, large], p2)

    if order == "alone_then_batched":
        first = run(cached, [small], p1)
        second = run(cached, [small, large], p2)
        assert first == baseline_alone
        assert second == baseline_batched, (
            "Cached run of [small, large] differs from baseline — the "
            "small-image cache entry from the prior alone-request was "
            "probably reused with a stale 70-token tensor."
        )
    else:
        first = run(cached, [small, large], p2)
        second = run(cached, [small], p1)
        assert first == baseline_batched
        assert second == baseline_alone, (
            "Cached alone-request differs from baseline — the cache "
            "entry for small from the prior batched request was "
            "probably reused with a stale 1120-token tensor."
        )


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_auto_partial_cache_hit(model_id: str):
    """Warm the cache with a single small image, then send it batched
    with a fresh large image.  The partial-hit path in
    _cached_apply_hf_processor must not serve the small image's stale
    70-token tensor when the batched request needs 1120."""
    _, baseline = _build_processor(model_id, cache=None)
    sender_cache = MultiModalProcessorSenderCache(baseline.info.ctx.model_config)
    _, cached = _build_processor(model_id, cache=sender_cache)

    small = _make_image(_SIZE_SMALL_70, seed=5)
    large = _make_image(_SIZE_LARGE_1120, seed=6)

    def run(proc, images, prompt):
        return proc(
            prompt,
            mm_items=proc.info.parse_mm_data({"image": images}),
            hf_processor_mm_kwargs={"max_soft_tokens": "auto"},
        )["prompt_token_ids"]

    p1 = _image_prompt(baseline, 1)
    p2 = _image_prompt(baseline, 2)

    # Warm: small alone → 70 cached under hash(small, 70)-derived key.
    _ = run(cached, [small], p1)

    # Partial: small (potential hit) + large (miss) → 1120 needed.
    baseline_ids = run(baseline, [small, large], p2)
    cached_ids = run(cached, [small, large], p2)
    assert cached_ids == baseline_ids


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_auto_server_default_with_empty_request_kwargs(model_id: str):
    """``--mm-processor-kwargs='{"max_soft_tokens":"auto"}'`` (server
    default) with empty per-request kwargs must still canonicalize.
    This is the case where "auto" lives only in the merged kwargs, not
    in the raw request kwargs — the canonicalizer must still find it."""
    _, proc = _build_processor(
        model_id,
        mm_processor_kwargs={"max_soft_tokens": "auto"},
        cache=None,
    )

    img = _make_image(_SIZE_SMALL_70, seed=7)
    mm_data = {"image": [img]}

    # Must not raise — "auto" must be resolved before the HF processor
    # validator sees it.  A failure here means the server-default path
    # is not reaching the canonicalizer.
    out = proc(
        _image_prompt(proc, 1),
        mm_items=proc.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )
    assert out["prompt_token_ids"]


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_auto_uniform_budget_within_request(model_id: str):
    """All images in a single request must share one resolved budget
    (Gemma 4 invariant).  Verified indirectly via prompt_token_ids
    equivalence: the output under ``auto`` in a [small, large] batch
    must match the output under explicit ``max_soft_tokens=1120`` for
    the same batch."""
    _, proc_auto = _build_processor(model_id, cache=None)
    _, proc_explicit = _build_processor(model_id, cache=None)

    small = _make_image(_SIZE_SMALL_70, seed=8)
    large = _make_image(_SIZE_LARGE_1120, seed=9)

    common = {"image": [small, large]}
    out_auto = proc_auto(
        _image_prompt(proc_auto, 2),
        mm_items=proc_auto.info.parse_mm_data(common),
        hf_processor_mm_kwargs={"max_soft_tokens": "auto"},
    )["prompt_token_ids"]
    out_explicit = proc_explicit(
        _image_prompt(proc_explicit, 2),
        mm_items=proc_explicit.info.parse_mm_data(common),
        hf_processor_mm_kwargs={"max_soft_tokens": 1120},
    )["prompt_token_ids"]

    assert out_auto == out_explicit, (
        "auto must resolve to 1120 for a [small, large] batch (the "
        "max per-image budget) and produce the same token ids as an "
        "explicit 1120 request."
    )
