# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AutoConfig

import vllm.utils.gpu_sync_debug as gpu_sync_debug
from vllm.models.deepseek_v4.common.vision import (
    MAX_IMAGES_PER_PACKED_PASS,
    DeepseekV4Aligner,
    DeepseekV4ViT,
    encode_images,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA"),
]

GRIDS = [[28, 28], [27, 34], [10, 5]]


def _tiny_tower(dtype: torch.dtype) -> tuple[DeepseekV4ViT, DeepseekV4Aligner]:
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp")
    config.update(
        {
            "vision_n_layers": 2,
            "vision_dim": 64,
            "vision_n_heads": 4,
            "vision_inter_dim": 128,
            "hidden_size": 32,
        }
    )
    with torch.device("cuda"), set_default_torch_dtype(dtype):
        vision, aligner = DeepseekV4ViT(config), DeepseekV4Aligner(config)
    with torch.no_grad():
        for param in [*vision.parameters(), *aligner.parameters()]:
            param.normal_(std=0.1)
    return vision, aligner


def _patches(grids: list[list[int]], dtype: torch.dtype) -> torch.Tensor:
    num_patches = sum(h * w for h, w in grids)
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(
        num_patches, 3, 14, 14, device="cuda", dtype=dtype, generator=generator
    )


def _count_aligner_calls(aligner: DeepseekV4Aligner) -> list[int]:
    num_rows: list[int] = []
    aligner.w1.register_forward_hook(lambda _, args, __: num_rows.append(len(args[0])))
    return num_rows


def test_encode_images_matches_per_image_encoding(
    default_vllm_config, dist_init
) -> None:
    vision, aligner = _tiny_tower(torch.bfloat16)
    patches = _patches(GRIDS, torch.bfloat16)
    num_rows = _count_aligner_calls(aligner)

    together = encode_images(vision, aligner, patches, GRIDS)

    assert len(num_rows) == 1
    items = patches.split([h * w for h, w in GRIDS])
    for embeds, item, grid in zip(together, items, GRIDS, strict=True):
        alone = encode_images(vision, aligner, item, [grid])[0]
        torch.testing.assert_close(embeds, alone, rtol=0.05, atol=0.05)


def test_encode_images_caps_images_per_packed_pass(
    default_vllm_config, dist_init
) -> None:
    vision, aligner = _tiny_tower(torch.bfloat16)
    grids = (GRIDS * MAX_IMAGES_PER_PACKED_PASS)[: MAX_IMAGES_PER_PACKED_PASS + 1]
    patches = _patches(grids, torch.bfloat16)
    num_rows = _count_aligner_calls(aligner)

    together = encode_images(vision, aligner, patches, grids)

    assert len(num_rows) == 2
    items = patches.split([h * w for h, w in grids])
    for embeds, item, grid in zip(together, items, grids, strict=True):
        alone = encode_images(vision, aligner, item, [grid])[0]
        torch.testing.assert_close(embeds, alone, rtol=0.05, atol=0.05)


def test_encode_images_keeps_one_image_on_the_dense_path(
    default_vllm_config, dist_init
) -> None:
    vision, aligner = _tiny_tower(torch.bfloat16)
    patches = _patches(GRIDS[:1], torch.bfloat16)
    h, w = GRIDS[0]

    embeds = encode_images(vision, aligner, patches, GRIDS[:1])

    assert torch.equal(embeds[0], aligner(vision(patches, h, w), h, w))


def test_encode_images_has_no_gpu_sync(
    monkeypatch, default_vllm_config, dist_init
) -> None:
    vision, aligner = _tiny_tower(torch.bfloat16)
    patches = _patches(GRIDS, torch.bfloat16)
    encode_images(vision, aligner, patches, GRIDS)
    monkeypatch.setattr(gpu_sync_debug, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gpu_sync_debug, "_sync_check_enabled", False)
    gpu_sync_debug.enable_gpu_sync_check()

    embeds = gpu_sync_debug.with_gpu_sync_check(encode_images)(
        vision, aligner, patches, GRIDS
    )

    assert [e.shape[0] for e in embeds] == [100, 108, 8]
