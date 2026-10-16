# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models import qwen3_vl
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def tp1(tmp_path, default_vllm_config):
    """Initialize a temporary single-rank tensor-parallel group."""
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path / 'distributed'}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_chunked_vision_blocks_match_unchunked(tp1, monkeypatch) -> None:
    """Large images take the chunked block path; it must not change outputs.

    The chunk boundaries fall inside the first image and across the boundary
    between the two images, so a wrong RoPE slice, a misplaced chunk or a
    wrong cu_seqlens hand-off would all show up as a mismatch.
    """
    torch.manual_seed(0)
    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=96,
        num_heads=4,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=48,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0],
        hidden_act="gelu_pytorch_tanh",
    )
    visual = Qwen3_VisionTransformer(config).float().eval()
    with torch.no_grad():
        for name, parameter in visual.named_parameters():
            # Large Q/K weights keep attention far from uniform, so wrong
            # rotary positions change the output.
            bound = 2.0 if ".attn.qkv." in name else 0.1
            parameter.uniform_(-bound, bound)
    for module in visual.modules():
        if (quant_method := getattr(module, "quant_method", None)) is not None:
            quant_method.process_weights_after_loading(module)

    grid_thw = [[1, 20, 24], [1, 8, 12]]
    num_patches = sum(t * h * w for t, h, w in grid_thw)
    pixel_values = torch.randn(num_patches, 3 * 2 * 16 * 16)

    chunked_calls = 0
    forward_chunked = qwen3_vl.Qwen3_VisionBlock._forward_chunked

    def count_chunked(self, *args, **kwargs):
        nonlocal chunked_calls
        chunked_calls += 1
        return forward_chunked(self, *args, **kwargs)

    monkeypatch.setattr(qwen3_vl.Qwen3_VisionBlock, "_forward_chunked", count_chunked)

    with torch.inference_mode():
        monkeypatch.setattr(qwen3_vl, "_VIT_CHUNK_TOKENS", num_patches)
        expected = visual(pixel_values, grid_thw)
        assert chunked_calls == 0

        monkeypatch.setattr(qwen3_vl, "_VIT_CHUNK_TOKENS", 100)
        actual = visual(pixel_values, grid_thw)
        assert chunked_calls == config.depth

    torch.testing.assert_close(actual, expected)
