# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3 vision tower keeps its RoPE cache and rotation in FP32."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype

# BF16 angles or BF16 arithmetic give ~2e-3 relative L2 against the FP32
# reference; FP32 angles and arithmetic stay below 1e-5.
_MAX_REL_L2 = 1e-4


@pytest.fixture
def dist_init():
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()


def _make_vision_config():
    # Real Qwen3.5 vision values, with depth and widths shrunk. head_dim stays
    # 72 (hidden_size // num_heads) as in the real tower; depth=2 keeps more
    # than one block so the per-block checks cover the whole loop.
    return SimpleNamespace(
        hidden_size=144,
        num_heads=2,
        depth=2,
        intermediate_size=128,
        out_hidden_size=64,
        hidden_act="gelu_pytorch_tanh",
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[],
    )


def _build_visual(config):
    # vLLM builds the model with the model dtype as the default dtype.
    with set_default_torch_dtype(torch.bfloat16):
        return Qwen3_VisionTransformer(
            config, norm_eps=1e-6, quant_config=None, prefix="visual"
        )


def _make_grid(config):
    # One image (t=1) with 16 merge blocks per side: height and width must be
    # multiples of spatial_merge_size, and 16 gives the 32 x 32 patch grid
    # used in the #66070 repro.
    blocks_per_side = 16
    merge = config.spatial_merge_size
    return [1, blocks_per_side * merge, blocks_per_side * merge]


def _fp32_cos_sin(visual, grid):
    """Angles computed independently in FP32, laid out like rot_pos_emb."""
    rope = visual.rotary_pos_emb
    exponents = torch.arange(0, rope.rotary_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (rope.base ** (exponents / rope.rotary_dim))
    positions = visual.rot_pos_ids(grid[1], grid[2], visual.spatial_merge_size)
    angles = (positions[..., None].float() * inv_freq).flatten(1)
    return angles.cos(), angles.sin()


def _rotate_fp32(qk, cos, sin):
    """Neox-style rotation in FP32, cast back to the input dtype."""
    x = qk.float()
    cos = torch.cat((cos, cos), dim=-1)[None, :, None, :]
    sin = torch.cat((sin, sin), dim=-1)[None, :, None, :]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos + rotated * sin).to(qk.dtype)


def _rel_l2(actual, expected):
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm()).item()


@pytest.mark.usefixtures("dist_init")
def test_qwen3_vision_rope_cache_is_fp32():
    config = _make_vision_config()
    visual = _build_visual(config)

    assert visual.rotary_pos_emb.cos_sin_cache.dtype == torch.float32
    cos, sin = visual.rot_pos_emb([_make_grid(config)])
    assert cos.dtype == sin.dtype == torch.float32


@pytest.mark.usefixtures("dist_init")
def test_qwen3_vision_blocks_rotate_in_fp32():
    config = _make_vision_config()
    visual = _build_visual(config)

    assert len(visual.blocks) == config.depth
    assert all(
        block.attn.apply_rotary_emb.enable_fp32_compute for block in visual.blocks
    )


@pytest.mark.usefixtures("dist_init")
def test_qwen3_vision_rotation_matches_fp32_reference():
    torch.manual_seed(0)
    config = _make_vision_config()
    visual = _build_visual(config)
    grid = _make_grid(config)
    cos, sin = visual.rot_pos_emb([grid])

    num_patches = grid[0] * grid[1] * grid[2]
    head_dim = config.hidden_size // config.num_heads
    # q and k stacked along dim 0, as Qwen2_5_VisionAttention.forward does.
    qk = torch.randn(2, num_patches, config.num_heads, head_dim, dtype=torch.bfloat16)
    expected = _rotate_fp32(qk, *_fp32_cos_sin(visual, grid))

    rotary = visual.blocks[0].attn.apply_rotary_emb
    assert _rel_l2(rotary.forward_native(qk, cos, sin), expected) < _MAX_REL_L2
    if current_platform.is_cuda_alike():
        # Dispatches to forward_cuda or forward_hip for the current platform.
        out = rotary(qk.cuda(), cos.cuda(), sin.cuda()).cpu()
        assert _rel_l2(out, expected) < _MAX_REL_L2
