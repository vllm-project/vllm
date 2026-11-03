# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder CUDA graph parity tests for a tiny random PaddleOCR-VL vision tower.

The graph path precomputes the per-image metadata on the host and pads the
inputs to a fixed batch shape, so its output must match the eager path.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.paddleocr_vl.configuration_paddleocr_vl import (
    PaddleOCRVLConfig,
)

from vllm.config import MultiModalConfig, get_current_vllm_config
from vllm.model_executor.models.paddleocr_vl import (
    PaddleOCRVLForConditionalGeneration,
    Projector,
    SiglipVisionModel,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="The vision rotary embedding and CUDA graphs need CUDA or ROCm.",
)

DEVICE = torch.device("cuda")
DTYPE = torch.float32
PATCH_SIZE = 14
MAX_BATCH_SIZE = 4

# Multi-image batches as (t, h, w) patch grids; h and w are multiples of the
# 2x2 spatial merge size.
BATCH_SMALL = [(1, 2, 2), (1, 4, 2)]
BATCH_MIXED = [(1, 4, 6), (1, 8, 4), (1, 2, 2)]
BATCH_FULL = [(1, 10, 8), (1, 4, 4), (1, 2, 6), (1, 2, 2)]


@pytest.fixture
def make_model(dist_init):
    """Build a tiny PaddleOCR-VL vision tower and projector with random weights.

    Only the modules used by the encoder are built; the language model is not.
    """

    def _make(
        attn_backend: AttentionBackendEnum | None = None,
    ) -> PaddleOCRVLForConditionalGeneration:
        config = PaddleOCRVLConfig(
            vision_config=dict(
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_hidden_layers=2,
                image_size=56,
                patch_size=PATCH_SIZE,
            ),
            text_config=dict(hidden_size=32),
        )
        vllm_config = get_current_vllm_config()
        old_model_config = vllm_config.model_config
        if attn_backend is not None:
            vllm_config.model_config = SimpleNamespace(
                multimodal_config=MultiModalConfig(mm_encoder_attn_backend=attn_backend)
            )
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(DTYPE)
        try:
            with torch.device(DEVICE):
                model = object.__new__(PaddleOCRVLForConditionalGeneration)
                torch.nn.Module.__init__(model)
                model.config = config
                model.visual = SiglipVisionModel(config.vision_config)
                model.mlp_AR = Projector(config.text_config, config.vision_config)
        finally:
            torch.set_default_dtype(old_dtype)
            vllm_config.model_config = old_model_config

        if attn_backend is not None:
            assert model.visual.vision_model.encoder.attn_backend == attn_backend

        torch.manual_seed(0)
        for param in model.parameters():
            torch.nn.init.normal_(param, std=0.05)
        return model.eval()

    return _make


def make_mm_kwargs(grids: list[tuple[int, int, int]]) -> dict[str, torch.Tensor]:
    num_patches = sum(t * h * w for t, h, w in grids)
    return {
        "pixel_values": torch.randn(
            num_patches, 3, PATCH_SIZE, PATCH_SIZE, device=DEVICE, dtype=DTYPE
        ),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long),
    }


def copy_replay_into_capture_buffers(
    buffers: dict[str, torch.Tensor], replay: dict[str, torch.Tensor | None]
) -> None:
    """Copy replay values into capture buffers as EncoderCudaGraphManager does."""
    for key, buf in buffers.items():
        src = replay.get(key)
        if src is None:
            continue
        if src.ndim == 0:
            buf.copy_(src)
        else:
            EncoderCudaGraphManager._copy_padded_buffer(buf, src)


@pytest.mark.parametrize("grids", [BATCH_SMALL, BATCH_MIXED, BATCH_FULL])
@torch.inference_mode()
def test_cudagraph_forward_matches_eager(make_model, grids):
    """The graph forward on precomputed replay buffers must match eager."""
    model = make_model()
    torch.manual_seed(1)
    mm_kwargs = make_mm_kwargs(grids)

    eager = model.encoder_eager_forward(mm_kwargs)
    replay = model.prepare_encoder_cudagraph_replay_buffers(
        mm_kwargs, MAX_BATCH_SIZE, MAX_BATCH_SIZE
    ).values
    graph = model.encoder_cudagraph_forward(dict(replay))

    num_tokens = sum(t * (h // 2) * (w // 2) for t, h, w in grids)
    assert eager.shape == (num_tokens, model.config.text_config.hidden_size)
    torch.testing.assert_close(graph, eager, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_padded_capture_replay_matches_eager(make_model):
    """One captured graph, padded to a larger budget, replays different batches.

    A real ``torch.cuda.graph`` capture fails on host syncs, and replaying two
    batches of different sizes checks that the zero padding leaves the real
    rows unchanged.
    """
    model = make_model(AttentionBackendEnum.TRITON_ATTN)
    token_budget = 64
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget, MAX_BATCH_SIZE, MAX_BATCH_SIZE, DEVICE, DTYPE
    ).values
    buffers = dict(capture)

    # Warm up (Triton autotuning/compilation) before capturing.
    model.encoder_cudagraph_forward(dict(buffers))
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model.encoder_cudagraph_forward(dict(buffers))
    assert output.shape[0] == token_budget

    torch.manual_seed(1)
    for grids in (BATCH_MIXED, BATCH_FULL):
        mm_kwargs = make_mm_kwargs(grids)
        eager = model.encoder_eager_forward(mm_kwargs)
        replay = model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, MAX_BATCH_SIZE, MAX_BATCH_SIZE
        ).values
        copy_replay_into_capture_buffers(buffers, replay)
        graph.replay()
        torch.testing.assert_close(
            output[: eager.shape[0]], eager, atol=1e-4, rtol=1e-4
        )
