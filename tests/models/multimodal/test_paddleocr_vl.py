# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PaddleOCR-VL encodes all scheduled images in one vision tower call; each
image must still behave as if it were encoded on its own."""

import math

import pytest
import torch
import torch.nn as nn
from transformers import BatchFeature, PreTrainedConfig, SiglipVisionConfig

from vllm.model_executor.models.paddleocr_vl import (
    PaddleOCRVLForConditionalGeneration,
    PaddleOCRVLMultiModalProcessor,
    Projector,
    SiglipVisionModel,
)
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.utils import group_and_batch_mm_items
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

VISION_CONFIG = SiglipVisionConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    image_size=56,
    patch_size=14,
    hidden_act="gelu_pytorch_tanh",
)
TEXT_CONFIG = PreTrainedConfig(hidden_size=48)

# Images 0 and 2 share a shape, so swapping them is only visible in values.
MIXED_GRIDS = [(1, 4, 6), (1, 2, 2), (1, 4, 6), (1, 6, 2)]
GRID_CASES = [[(1, 4, 6)], MIXED_GRIDS]

pytestmark = pytest.mark.core_model

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="requires CUDA"
)


def _num_patches(grids):
    return [math.prod(grid) for grid in grids]


def _make_pixels(grids, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(sum(_num_patches(grids)), 3, 14, 14, generator=gen)


def _make_model(dtype: torch.dtype) -> PaddleOCRVLForConditionalGeneration:
    # The encoder picks its attention backend from the default dtype.
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = object.__new__(PaddleOCRVLForConditionalGeneration)
        nn.Module.__init__(model)
        model.visual = SiglipVisionModel(VISION_CONFIG)
        model.mlp_AR = Projector(TEXT_CONFIG, VISION_CONFIG)
    finally:
        torch.set_default_dtype(default_dtype)

    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            init = torch.randn(param.shape, generator=gen) * 0.05
            if "norm" in name and name.endswith("weight"):
                init += 1.0
            param.copy_(init)
    return model.to(device="cuda", dtype=dtype).eval()


def _attn_backend(model) -> AttentionBackendEnum:
    return model.visual.vision_model.encoder.attn_backend


@torch.inference_mode()
def _embed(model, pixels, grids):
    return model.embed_multimodal(
        pixel_values=pixels.to(device="cuda", dtype=model.visual.dtype),
        image_grid_thw=torch.tensor(grids),
    )


def _embed_one_by_one(model, pixels, grids):
    return [
        _embed(model, image_pixels, [grid])[0]
        for image_pixels, grid in zip(pixels.split(_num_patches(grids)), grids)
    ]


def _rel_errors(outputs, references):
    assert len(outputs) == len(references)
    return [
        ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        for out, ref in zip(outputs, references, strict=True)
    ]


@pytest.mark.cpu_test
def test_processor_keeps_patches_flat_and_sliced_per_image():
    grids = torch.tensor(MIXED_GRIDS)
    pixels = _make_pixels(MIXED_GRIDS)
    hf_inputs = BatchFeature({"pixel_values": pixels, "image_grid_thw": grids})

    fields = PaddleOCRVLMultiModalProcessor._get_mm_fields_config(None, hf_inputs, {})
    items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, fields)["image"]

    for item, image_pixels in zip(
        items, pixels.split(_num_patches(MIXED_GRIDS)), strict=True
    ):
        torch.testing.assert_close(item["pixel_values"].data, image_pixels)

    # Items are re-packed into the flat layout the model's schema expects.
    [(num_items, batch)] = group_and_batch_mm_items(items)
    assert num_items == len(MIXED_GRIDS)
    model = object.__new__(PaddleOCRVLForConditionalGeneration)
    image_input = model._parse_and_validate_image_input(**batch)
    torch.testing.assert_close(image_input.pixel_values, pixels)
    torch.testing.assert_close(image_input.image_grid_thw, grids)


@pytest.mark.cpu_test
def test_projector_merges_2x2_within_each_image():
    projector = Projector(TEXT_CONFIG, VISION_CONFIG)
    projector.pre_norm = nn.Identity()
    projector.linear_1 = nn.Identity()
    projector.act = nn.Identity()
    projector.linear_2 = nn.Identity()

    # Each patch's feature is its global patch index, so every merged token
    # names exactly which patches it was built from.
    num_patches = _num_patches(MIXED_GRIDS)
    patch_ids = torch.arange(sum(num_patches), dtype=torch.float32)
    features = patch_ids[:, None].expand(-1, VISION_CONFIG.hidden_size)
    outputs = projector(features.split(num_patches), torch.tensor(MIXED_GRIDS))

    for output, ids, (_, h, w) in zip(
        outputs, patch_ids.split(num_patches), MIXED_GRIDS, strict=True
    ):
        expected = (
            ids.view(h // 2, 2, w // 2, 2)
            .permute(0, 2, 1, 3)
            .reshape(-1, 4)
            .repeat_interleave(VISION_CONFIG.hidden_size, dim=1)
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize("grids", GRID_CASES)
def test_batched_encode_matches_per_image_fp32(grids):
    model = _make_model(torch.float32)
    pixels = _make_pixels(grids)

    batched = _embed(model, pixels, grids)
    reference = _embed_one_by_one(model, pixels, grids)

    assert [out.shape for out in batched] == [ref.shape for ref in reference]
    # fp32 GEMMs of different row counts differ at ~5e-7; mixing positions,
    # attention spans or image order gives errors of order 1.
    assert max(_rel_errors(batched, reference)) < 1e-5


@requires_cuda
@pytest.mark.usefixtures("dist_init")
def test_batched_encode_bf16_flash_attn_error_vs_fp32():
    model = _make_model(torch.bfloat16)
    if _attn_backend(model) != AttentionBackendEnum.FLASH_ATTN:
        pytest.skip("FlashAttention ViT backend not selected")
    reference_model = _make_model(torch.float32)
    pixels = _make_pixels(MIXED_GRIDS)

    reference = _embed_one_by_one(reference_model, pixels, MIXED_GRIDS)
    batched_errors = _rel_errors(_embed(model, pixels, MIXED_GRIDS), reference)
    per_image_errors = _rel_errors(
        _embed_one_by_one(model, pixels, MIXED_GRIDS), reference
    )

    # Batching changes GEMM shapes, so bf16 results need not be bitwise equal
    # to per-image runs. Both land ~5e-3 from fp32; a correctness bug would
    # put the batched error near 1.
    for batched_error, per_image_error in zip(batched_errors, per_image_errors):
        assert per_image_error < 2e-2
        assert batched_error < 2e-2


@requires_cuda
@pytest.mark.usefixtures("dist_init")
def test_batched_images_do_not_attend_to_each_other():
    model = _make_model(torch.bfloat16)
    pixels = _make_pixels(MIXED_GRIDS)
    perturbed = pixels.clone()
    start, end = _num_patches(MIXED_GRIDS[:1])[0], sum(_num_patches(MIXED_GRIDS[:2]))
    perturbed[start:end] = _make_pixels(MIXED_GRIDS[1:2], seed=1)

    outputs = _embed(model, pixels, MIXED_GRIDS)
    perturbed_outputs = _embed(model, perturbed, MIXED_GRIDS)

    # Same shapes run the same kernels, and every op except attention is
    # per-token, so other images must be bitwise unchanged.
    assert not torch.equal(outputs[1], perturbed_outputs[1])
    for i in (0, 2, 3):
        torch.testing.assert_close(outputs[i], perturbed_outputs[i], rtol=0, atol=0)


@requires_cuda
@pytest.mark.usefixtures("dist_init")
def test_batched_encode_does_not_sync_with_flash_attn():
    model = _make_model(torch.bfloat16)
    if _attn_backend(model) != AttentionBackendEnum.FLASH_ATTN:
        pytest.skip("FlashAttention ViT backend not selected")
    pixels = _make_pixels(MIXED_GRIDS).to(device="cuda", dtype=torch.bfloat16)
    grids = torch.tensor(MIXED_GRIDS)

    # torch's own check also covers syncs that `gpu_sync_allowed()` hides,
    # such as `max_seqlen.item()` on a device tensor.
    torch.accelerator.synchronize()
    sync_debug_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.inference_mode():
            model.embed_multimodal(pixel_values=pixels, image_grid_thw=grids)
    finally:
        torch.cuda.set_sync_debug_mode(sync_debug_mode)
