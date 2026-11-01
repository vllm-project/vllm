# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-M3.1 DSpark draft attention under vLLM's torch.compile.

vLLM traces the draft model once, in the memory-profiling run, whose forward
context has no slot mapping, and reuses that graph afterwards. The draft
tokens' k/v insert must still happen on the later calls.
"""

import pytest
import torch
from torch import nn

from vllm.utils.torch_utils import set_default_torch_dtype

HEAD_DIM = 128
BLOCK_SIZE = 16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class _QueryAttention(nn.Module):
    """Stands in for the draft's Attention: returns its query, so the layer
    output shows the q the attention got."""

    def __init__(self, layer_name):
        super().__init__()
        self.layer_name = layer_name

    def forward(self, q, k, v, output_dtype=None):
        return q.to(output_dtype)


@pytest.mark.parametrize(
    "kv_cache_dtype,query_pre_quantized",
    [("auto", False), ("fp8", False), ("fp8", True)],
)
def test_draft_attention_stores_kv_after_compile(
    dist_init, monkeypatch, kv_cache_dtype, query_pre_quantized
):
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context
    from vllm.models.minimax_m3.common.dspark import MiniMaxM3DSparkAttention
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    config = MiniMaxM3TextConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=8192,
    )

    @support_torch_compile
    class DraftAttention(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
            super().__init__()
            self.self_attn = MiniMaxM3DSparkAttention(
                config,
                sliding_window=None,
                causal=False,
                cache_config=None,
                quant_config=None,
                prefix=f"{prefix}.self_attn",
                query_pre_quantized=query_pre_quantized,
            )

        def forward(
            self, positions: torch.Tensor, hidden_states: torch.Tensor
        ) -> torch.Tensor:
            return self.self_attn(positions, hidden_states)

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.NONE
        )
    )
    torch.manual_seed(0)
    with (
        set_current_vllm_config(vllm_config),
        torch.device("cuda"),
        set_default_torch_dtype(torch.bfloat16),
    ):
        model = DraftAttention(vllm_config=vllm_config, prefix="model.layers.0")
    attn = model.self_attn
    for param in (
        attn.qkv_proj.weight,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.o_proj.weight,
    ):
        param.data.normal_(std=0.05)
    layer = attn.attn
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else torch.bfloat16
    layer.kv_cache = torch.zeros(
        4, 2, BLOCK_SIZE, 2 * HEAD_DIM, dtype=cache_dtype, device="cuda"
    )
    attn.attn = _QueryAttention(layer.layer_name)

    hidden = torch.randn(7, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 127, 128, 4000, 8190, 9], device="cuda")
    slots = {layer.layer_name: torch.tensor([5, -1, 17, 18, 40, 63, 0], device="cuda")}

    with torch.inference_mode():
        with set_forward_context(None, vllm_config, slot_mapping=slots):
            expected_out = attn(positions, hidden)
        expected_cache = layer.kv_cache.clone()
        assert expected_cache.any()
        layer.kv_cache.zero_()

        # The memory-profiling run: no slot mapping, nothing is stored.
        with set_forward_context(None, vllm_config):
            model(positions, hidden)
        assert not layer.kv_cache.any()

        with set_forward_context(None, vllm_config, slot_mapping=slots):
            out = model(positions, hidden)
    assert layer.kv_cache.any(), "the compiled draft attention stored no k/v"
    got, expected, rtol = layer.kv_cache, expected_cache, 2e-2
    if kv_cache_dtype == "fp8":
        got = got.view(torch.float8_e4m3fn).float()
        expected = expected.view(torch.float8_e4m3fn).float()
        rtol = 0.13
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=rtol)
    torch.testing.assert_close(out, expected_out, atol=2e-2, rtol=2e-2)
