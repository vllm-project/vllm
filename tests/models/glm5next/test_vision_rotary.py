# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash vision tower: per-image rotary positions and encoder metadata.

``Glm5NextProcessor`` bounds images by pixel budget, not aspect ratio, so a slim
image yields a patch grid whose long side exceeds the 8192 entries the vision
rope table used to have; gathering past its end is a device-side assert that
takes down the engine. Separately, every image's metadata is built on the host,
and a pageable copy to the GPU fails ``VLLM_GPU_SYNC_CHECK=error`` on the first
image.
"""

import pytest
import torch
from transformers import Glm5NextTextConfig, Glm5NextVisionConfig

import vllm.utils.gpu_sync_debug as gsd
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.models.glm5next.common import multimodal

HEAD_DIM, ROTARY_DIM, BASE = 64, 32, 10000


def _tiny_tower(monkeypatch, max_position_embeddings: int):
    """Real tower with no transformer blocks, on a single-rank TP world."""
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(multimodal, "is_vit_use_data_parallel", lambda: True)
    text_config = Glm5NextTextConfig(
        max_position_embeddings=max_position_embeddings, swiglu_limit=10.0
    )
    vision_config = Glm5NextVisionConfig(
        depth=0,
        hidden_size=HEAD_DIM,
        num_heads=1,
        intermediate_size=32,
        out_hidden_size=32,
        projection_intermediate_size=32,
        swiglu_limit=10.0,
    )
    return multimodal.Glm5NextVisionTransformer(text_config, vision_config)


def test_rot_pos_emb_covers_grid_longer_than_8192(monkeypatch, default_vllm_config):
    tower = _tiny_tower(monkeypatch, max_position_embeddings=16384)

    cos, sin, pos_ids = tower.rot_pos_emb([[1, 2, 8194]])

    assert int(pos_ids.max()) == 8193
    inv_freq = 1.0 / BASE ** (
        torch.arange(0, ROTARY_DIM, 2, dtype=torch.float64) / ROTARY_DIM
    )
    freqs = (pos_ids.unsqueeze(-1) * inv_freq).flatten(1)
    # fp32 rope arguments near position 8192 only carry ~5e-4 absolute precision.
    torch.testing.assert_close(cos.double(), freqs.cos(), rtol=0, atol=2e-3)
    torch.testing.assert_close(sin.double(), freqs.sin(), rtol=0, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_image_forward_has_no_implicit_gpu_sync(monkeypatch, default_vllm_config):
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    gsd._install_copy_checkers()
    tower = _tiny_tower(monkeypatch, max_position_embeddings=16384).to("cuda")
    # The model runner keeps image_grid_thw on the CPU.
    grid_thw = torch.tensor([[1, 4, 4]])
    pixel_values = torch.randn(16, 3 * 2 * 14 * 14, device="cuda")

    out = gsd.with_gpu_sync_check(tower)(pixel_values, grid_thw)

    assert out.shape == (4, 32)
