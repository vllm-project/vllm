# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph capture/replay tests for the MiniMax M3 vision tower.

Drives the real ``EncoderCudaGraphManager`` against a thin wrapper holding
only the vision tower, in the style of
``tests/v1/cudagraph/test_encoder_cudagraph.py`` (the model class is
registered ``is_available_online=False``, so the manager-level e2e path
cannot run in CI).
"""

import pytest
import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config.vllm import VllmConfig, set_current_vllm_config
from vllm.models.minimax_m3.common.encoder_cudagraph import (
    MiniMaxM3EncoderCudaGraphMixin,
)
from vllm.models.minimax_m3.common.vision_tower import (
    MiniMaxVLVisionModel,
    MiniMaxVLVisionTransformer,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="encoder CUDA graphs need CUDA"
)

PATCH_DIM = 3 * 2 * 14 * 14  # C * temporal_patch_size * patch_size^2
_BUDGETS = [512]  # output tokens
_MAX_BATCH = 4


def _vision_config_dict(vision_segment_max_frames: int | None = None) -> dict:
    config = {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "layer_norm_eps": 1e-6,
        "num_channels": 3,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "img_token_compression_config": {
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }
    if vision_segment_max_frames is not None:
        config["vision_segment_max_frames"] = vision_segment_max_frames
    return config


def _reinit_weights(module: nn.Module) -> None:
    # vLLM linear biases are torch.empty until weight loading, so a bare run
    # sees uninitialized memory; re-init with a sane std and zero biases.
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith(".bias"):
                p.zero_()
            elif p.dim() >= 2:
                p.normal_(0, 0.02)


def _build_transformer(device, dtype, vision_segment_max_frames=None):
    # vLLM sets torch's default dtype to the model dtype during load, and the
    # tower reads torch.get_default_dtype() for backend selection; bf16 makes
    # get_vit_attn_backend pick FLASH_ATTN as in production.
    with set_default_torch_dtype(dtype), set_current_vllm_config(VllmConfig()):
        tower = MiniMaxVLVisionTransformer(
            PreTrainedConfig.from_dict(_vision_config_dict(vision_segment_max_frames)),
            require_post_norm=False,
        ).to(device=device)
    _reinit_weights(tower)
    return tower


class _ViTOnlyModel(nn.Module, MiniMaxM3EncoderCudaGraphMixin):
    """Minimal host for the encoder-CUDA-graph mixin: just the tower."""

    def __init__(self, device, dtype):
        super().__init__()
        self.multimodal_config = None
        with set_default_torch_dtype(dtype), set_current_vllm_config(VllmConfig()):
            self.vision_tower = MiniMaxVLVisionModel(
                PreTrainedConfig.from_dict(_vision_config_dict()),
                text_hidden_size=256,
            ).to(device=device)
        _reinit_weights(self.vision_tower)


def _make_manager(model, device, dtype) -> EncoderCudaGraphManager:
    """Create EncoderCudaGraphManager bypassing VllmConfig (same trick as
    tests/v1/cudagraph/test_encoder_cudagraph.py)."""
    mgr = object.__new__(EncoderCudaGraphManager)
    mgr.token_budgets = sorted(_BUDGETS)
    mgr.path_token_budgets = {"default": mgr.token_budgets}
    mgr.max_batch_size = _MAX_BATCH
    mgr.max_frames_per_batch = _MAX_BATCH * 2
    mgr.use_dp = False
    mgr.budget_graphs = {"default": {}}
    mgr.graph_pool = None
    mgr._capture_axes = ()
    mgr.graph_hits = 0
    mgr.graph_misses = 0
    mgr.log_stats_interval = 100
    mgr.model = model
    mgr.config = model.get_encoder_cudagraph_config()
    mgr.device = device
    mgr.dtype = dtype
    return mgr


def _make_mm_kwargs(grid_thw: list[list[int]], device, dtype, seed: int):
    n = sum(t * h * w for t, h, w in grid_thw)
    g = torch.Generator(device="cpu").manual_seed(seed)
    pixel_values = torch.randn(n, PATCH_DIM, generator=g, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}


def _eager_per_item(tower, mm_kwargs) -> list[torch.Tensor]:
    """Eager tower output for each item in mm_kwargs."""
    merge = tower.spatial_merge_size
    outs, start = [], 0
    for t, h, w in mm_kwargs["image_grid_thw"]:
        n = t * h * w
        with torch.inference_mode():
            outs.append(
                tower(mm_kwargs["pixel_values"][start : start + n], [[t, h, w]])
            )
        start += n
        assert outs[-1].shape[0] == n // (merge * merge)
    return outs


@pytest.mark.parametrize(
    "grid_thw,vision_segment_max_frames",
    [
        ([[1, 24, 24]], None),  # single 336x336 image
        ([[1, 24, 24], [1, 16, 16], [1, 48, 64]], None),  # mixed images
        ([[4, 48, 64]], None),  # short video (t > 1)
        ([[12, 48, 64]], 8),  # long video split by the frame limit
    ],
)
def test_encoder_metadata_forward_parity(
    grid_thw, vision_segment_max_frames, dist_init
):
    """forward(encoder_metadata=...) must match forward(grid_thw=...)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tower = _build_transformer(device, dtype, vision_segment_max_frames)
    mm_kwargs = _make_mm_kwargs(grid_thw, device, dtype, seed=0)

    with torch.inference_mode():
        eager = tower(mm_kwargs["pixel_values"], grid_thw)
        metadata = tower.prepare_encoder_metadata(grid_thw, device=device)
        via_metadata = tower(mm_kwargs["pixel_values"], None, encoder_metadata=metadata)
    torch.testing.assert_close(via_metadata, eager, rtol=0, atol=0)


class TestMiniMaxM3EncoderCudaGraph:
    @pytest.fixture
    def model_and_mgr(self, dist_init):
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        model = _ViTOnlyModel(device, dtype)
        mgr = _make_manager(model, device, dtype)
        mgr.capture(graph_pool=current_platform.graph_pool_handle())
        return model, mgr

    @pytest.mark.parametrize(
        "grid_thw",
        [
            [[1, 24, 24], [1, 16, 16], [1, 16, 16]],
            [[2, 24, 24]],
        ],
    )
    def test_execute_matches_eager(self, model_and_mgr, grid_thw):
        model, mgr = model_and_mgr
        mm_kwargs = _make_mm_kwargs(grid_thw, mgr.device, mgr.dtype, seed=1)
        result = mgr.execute(mm_kwargs)
        assert len(result) == len(grid_thw)
        assert mgr.graph_hits == len(grid_thw)
        assert mgr.graph_misses == 0

        # Replay is not bitwise vs eager: the varlen attention kernel's
        # split/scheduling decisions depend on the padded row count, so
        # reduction order differs at bf16-ulp scale (observed max 2^-6).
        for out, ref in zip(result, _eager_per_item(model.vision_tower, mm_kwargs)):
            assert out.shape == ref.shape
            torch.testing.assert_close(out, ref, rtol=1e-2, atol=2e-2)

    def test_eager_fallback_oversized_image(self, model_and_mgr):
        model, mgr = model_and_mgr
        # 64*64/4 = 1024 output tokens > max budget 512.
        grid_thw = [[1, 64, 64]]
        mm_kwargs = _make_mm_kwargs(grid_thw, mgr.device, mgr.dtype, seed=2)
        result = mgr.execute(mm_kwargs)
        assert len(result) == 1
        assert mgr.graph_misses == 1

        (ref,) = _eager_per_item(model.vision_tower, mm_kwargs)
        torch.testing.assert_close(result[0], ref, rtol=0, atol=0)

    def test_small_image_replay_matches_eager(self, model_and_mgr):
        model, mgr = model_and_mgr
        # A small image leaves nearly all captured rows in one padding sequence.
        grid_thw = [[1, 4, 4]]
        mm_kwargs = _make_mm_kwargs(grid_thw, mgr.device, mgr.dtype, seed=3)
        result = mgr.execute(mm_kwargs)
        assert len(result) == 1
        assert mgr.graph_hits == 1
        buffers = mgr.budget_graphs["default"][512].input_buffers
        assert buffers["cu_seqlens"].diff().max().item() <= buffers["max_seqlen"].item()

        (ref,) = _eager_per_item(model.vision_tower, mm_kwargs)
        torch.testing.assert_close(result[0], ref, rtol=1e-2, atol=2e-2)
