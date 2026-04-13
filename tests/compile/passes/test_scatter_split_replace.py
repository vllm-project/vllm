# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

import vllm
from tests.compile.backend import TestBackend
from vllm.compilation.passes.utility.scatter_split_replace import (
    ScatterSplitReplacementPass,
)
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


class ScatterSplitReplacementModel(nn.Module):
    """Model with a rope+getitem+slice_scatter+split_with_sizes sequence."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.q_size = num_heads * head_size
        self.kv_size = num_kv_heads * head_size

        self.rotary_emb = RotaryEmbedding(
            head_size,
            rotary_dim=head_size,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=True,
            dtype=dtype,
        )

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor):
        # Create copy so inplace ops do not modify the original tensors
        qkv = qkv.clone()
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        q = q + 1
        k = k + 2
        v = v + 3
        return q, k, v

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        return [
            torch.ops.aten.slice_scatter.default,
            torch.ops.aten.split_with_sizes.default,
            torch.ops.aten.getitem.default,
        ]

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.aten.getitem.default]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scatter_split_replace(dtype):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    num_heads = 8
    num_kv_heads = 4
    head_size = 64

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rotary_embedding"],
        ),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        # ScatterSplitReplacementPass requires SplitCoalescingPass to be run before it
        coalesce_pass = SplitCoalescingPass(vllm_config)
        replace_pass = ScatterSplitReplacementPass(vllm_config)
        passes = [coalesce_pass, replace_pass]
        backend = TestBackend(*passes)

        model = ScatterSplitReplacementModel(num_heads, num_kv_heads, head_size, dtype)

        T = 5
        qkv = torch.randn(
            T, num_heads * head_size + 2 * num_kv_heads * head_size, dtype=dtype
        )
        pos = torch.arange(T, dtype=torch.long)

        qkv_eager = qkv.clone()
        pos_eager = pos.clone()
        result_eager = model(qkv_eager, pos_eager)

        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(pos, 0)

        model_compiled = torch.compile(model, backend=backend)
        result_compiled = model_compiled(qkv, pos)

        for eager, compiled in zip(result_eager, result_compiled):
            torch.testing.assert_close(eager, compiled)

        assert backend.op_count(torch.ops.aten.slice_scatter.default) == 0
        assert backend.op_count(torch.ops.aten.split_with_sizes.default) == 1
