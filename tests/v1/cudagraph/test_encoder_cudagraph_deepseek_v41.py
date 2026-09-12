# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph tests for the DeepSeek-V4.1 ViT encoder.

Exercises the ``SupportsEncoderCudaGraph`` implementation in
``vllm/models/deepseek_v4_1/common/vl_cudagraph.py`` with a tiny
random-weight ViT + aligner: capture/replay via the real
``EncoderCudaGraphManager`` must match the per-image eager path.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import image_token_types
from vllm.models.deepseek_v4_1.common.vl_cudagraph import (
    DeepseekV4VLEncoderCudaGraphMixin,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

# Tiny tower: head_dim=32 keeps every ViT attention backend happy.
_CONFIG = SimpleNamespace(
    vision_dim=128,
    vision_n_heads=4,
    vision_patch_size=2,
    vision_n_layers=2,
    vision_inter_dim=256,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=2,
    hidden_size=64,
)

_BUDGETS = [16, 64]
_MAX_BATCH = 4
_DTYPE = torch.bfloat16


class _TinyDeepseekV4VLModel(
    torch.nn.Module, DeepseekV4VLEncoderCudaGraphMixin, SupportsEncoderCudaGraph
):
    """The mixin's attribute contract, minus the language model."""

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.multimodal_config = None
        self.vision = DeepseekV4ViT(config)
        self.aligner = DeepseekV4Aligner(config)
        for name in ("image_start", "image_end", "image_newline"):
            setattr(
                self,
                name,
                torch.nn.Parameter(torch.randn(config.hidden_size) * 0.02),
            )
        with torch.no_grad():
            for name, p in self.named_parameters():
                # vLLM linear weights are created empty (the loader fills
                # them); RMSNorm weights are already ones.
                if "norm" not in name:
                    p.copy_(torch.randn_like(p) * 0.02)


def _llm_grid(n_vit_h: int, n_vit_w: int) -> tuple[int, int]:
    r = _CONFIG.vision_downsample_ratio
    return (-(-n_vit_h // r), -(-n_vit_w // r))


def _make_mm_kwargs(
    grids: list[tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    p = _CONFIG.vision_patch_size
    patches = torch.randn(
        sum(h * w for h, w in grids), 3, p, p, device=device, dtype=dtype
    )
    llm_grid = [_llm_grid(h, w) for h, w in grids]
    # vit_grid/llm_grid/types are keep_on_cpu fields in the real processor.
    return {
        "patches": patches,
        "vit_grid": torch.tensor(grids, dtype=torch.int64),
        "llm_grid": torch.tensor(llm_grid, dtype=torch.int64),
        "types": torch.cat([image_token_types(lh, lw) for lh, lw in llm_grid]),
    }


@torch.inference_mode()
def _eager_reference(
    model: _TinyDeepseekV4VLModel, mm_kwargs: dict[str, Any]
) -> list[torch.Tensor]:
    """The per-image eager path from the model's ``_process_image_input``."""
    patches = mm_kwargs["patches"]
    vit_grid = mm_kwargs["vit_grid"].tolist()
    llm_grid = mm_kwargs["llm_grid"].tolist()
    types = mm_kwargs["types"]
    spans = []
    vit_offset = span_offset = 0
    for (h, w), (lh, lw) in zip(vit_grid, llm_grid, strict=True):
        n_vit = h * w
        span_len = lh * (lw + 1) + 2
        rows = model._encode_image(patches[vit_offset : vit_offset + n_vit], h, w)
        spans.append(
            model._build_image_span(rows, types[span_offset : span_offset + span_len])
        )
        vit_offset += n_vit
        span_offset += span_len
    return spans


def _make_manager(
    model: _TinyDeepseekV4VLModel,
    device: torch.device,
) -> EncoderCudaGraphManager:
    """EncoderCudaGraphManager without a full VllmConfig (same pattern as
    test_encoder_cudagraph.py)."""
    mgr = object.__new__(EncoderCudaGraphManager)
    mgr.token_budgets = sorted(_BUDGETS)
    mgr.path_token_budgets = {"default": mgr.token_budgets}
    mgr.max_batch_size = _MAX_BATCH
    mgr.max_frames_per_batch = 0
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
    mgr.dtype = _DTYPE
    return mgr


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skip if not cuda or rocm"
)
class TestDeepseekV41EncoderCudaGraph:
    @pytest.fixture(autouse=True)
    def _setup(self, dist_init):
        self.device = torch.device("cuda:0")
        with set_default_torch_dtype(_DTYPE):
            self.model = _TinyDeepseekV4VLModel(_CONFIG)
        self.model.to(device=self.device, dtype=_DTYPE)
        self.mgr = _make_manager(self.model, self.device)
        self.mgr.capture(graph_pool=current_platform.graph_pool_handle())

    def _check_matches_eager(self, grids: list[tuple[int, int]]):
        mm_kwargs = _make_mm_kwargs(grids, self.device, _DTYPE)
        expected = _eager_reference(self.model, mm_kwargs)
        with torch.inference_mode():
            result = self.mgr.execute(mm_kwargs)
        assert len(result) == len(grids)
        for i, (got, want) in enumerate(zip(result, expected, strict=True)):
            assert got.shape == want.shape, f"item {i}: {got.shape} != {want.shape}"
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)

    def test_capture_creates_one_graph_per_budget(self):
        assert set(self.mgr.budget_graphs["default"].keys()) == set(_BUDGETS)

    def test_single_even_image(self):
        self._check_matches_eager([(4, 4)])

    def test_odd_grids_hit_aligner_padding(self):
        # Odd h/w exercise the merge mask (eager path zero-pads).
        self._check_matches_eager([(3, 5)])
        self._check_matches_eager([(1, 1)])
        self._check_matches_eager([(5, 2)])

    def test_packed_multi_image_batch(self):
        # span 8 + 10 + 4 = 22 -> packed into the budget-64 graph.
        self._check_matches_eager([(4, 4), (3, 5), (1, 1)])
        assert self.mgr.graph_hits == 3

    def test_chunking_when_images_exceed_max_batch(self):
        # 6 images > max_batch_size=4 -> two packed batches.
        self._check_matches_eager([(4, 4), (2, 6), (1, 1), (3, 3), (4, 2), (2, 2)])

    def test_eager_fallback_when_image_exceeds_all_budgets(self):
        # span = 8*9+2 = 74 > max budget 64 -> eager fallback, still correct.
        self._check_matches_eager([(16, 16)])
        assert self.mgr.graph_misses == 1

    def test_mixed_batch_with_oversized_image(self):
        # The oversized image falls back to eager while the rest replay.
        self._check_matches_eager([(4, 4), (16, 16), (2, 3)])
