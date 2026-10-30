# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed Gemma 4 vision tower against transformers' padded one."""

from unittest.mock import patch

import pytest
import torch
from transformers import AutoModel
from transformers.models.gemma4 import Gemma4VisionConfig

from vllm.model_executor.layers.attention import mm_encoder_attention
from vllm.model_executor.models import gemma4_vision
from vllm.model_executor.models.gemma4_vision import Gemma4VisionModel
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# fp32 for a tight comparison: backends that support it.
BACKENDS = [AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.TRITON_ATTN]

# (columns, rows) of patches; both multiples of the pooling kernel. Equal
# grids, adjacent or not, share one SDPA call.
GRIDS = [(9, 6), (9, 6), (3, 3), (12, 9), (9, 6), (6, 15)]
MAX_PATCHES = 15 * 12


def _config(use_clipped_linears: bool) -> Gemma4VisionConfig:
    return Gemma4VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        position_embedding_size=32,
        use_clipped_linears=use_clipped_linears,
        standardize=True,
    )


def _hf_tower(config: Gemma4VisionConfig) -> torch.nn.Module:
    config._attn_implementation = "sdpa"
    tower = AutoModel.from_config(config).eval()
    with torch.no_grad():
        for name, tensor in tower.state_dict().items():
            if name.endswith(("input_min", "output_min")):
                tensor.fill_(-2.0)
            elif name.endswith(("input_max", "output_max")):
                tensor.fill_(2.0)
            elif (
                name.endswith("std_scale")
                or name.endswith("norm.weight")
                or "layernorm" in name
            ):
                tensor.uniform_(0.5, 1.5)
            else:
                tensor.normal_(std=0.1)
    return tower


def _image(columns: int, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Patches laid out like the processor's: row-major, padded at the end."""
    grid = torch.meshgrid(torch.arange(columns), torch.arange(rows), indexing="xy")
    positions = torch.full((MAX_PATCHES, 2), -1, dtype=torch.long)
    positions[: columns * rows] = torch.stack(grid, -1).reshape(-1, 2)
    pixel_values = torch.zeros(MAX_PATCHES, 3 * 16 * 16)
    pixel_values[: columns * rows] = torch.rand(columns * rows, 3 * 16 * 16)
    return pixel_values, positions


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_clipped_linears", [False, True])
@torch.inference_mode()
def test_packed_tower_matches_transformers(
    default_vllm_config, dist_init, backend, use_clipped_linears
):
    torch.manual_seed(0)
    config = _config(use_clipped_linears)
    hf_tower = _hf_tower(config).cuda()
    images = [_image(*grid) for grid in GRIDS]
    expected = [
        hf_tower(pv[None].cuda(), pos[None].cuda()).last_hidden_state
        for pv, pos in images
    ]

    with (
        patch.object(gemma4_vision, "get_vit_attn_backend", lambda **_: backend),
        patch.object(mm_encoder_attention, "get_vit_attn_backend", lambda **_: backend),
        torch.device("cuda"),
    ):
        tower = Gemma4VisionModel(config)
    tower.load_state_dict(hf_tower.state_dict())

    num_patches = [columns * rows for columns, rows in GRIDS]
    metadata = tower.prepare_encoder_metadata(
        [pos[:n].numpy() for (_, pos), n in zip(images, num_patches)]
    )
    pixel_values = torch.cat([pv[:n] for (pv, _), n in zip(images, num_patches)])
    actual = tower(pixel_values.cuda(), metadata).split([n // 9 for n in num_patches])

    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)
