# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.kernels.core.test_rotary_embedding_mla_cache_fused import (
    assert_rope_close_rocm,
)
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.config import (
    CompilationConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    get_rope,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@support_torch_compile
class RotaryEmbeddingCompileModule(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.rotary_emb = get_rope(
            head_size=32,
            max_position=128,
            dtype=torch.float32,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
            is_neox_style=True,
        )

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        q_rot, k_rot = self.rotary_emb(positions, query, key)
        return q_rot + k_rot


@pytest.mark.skipif(current_platform.is_cpu(), reason="Requires GPU for torch.compile")
def test_rotary_embedding_torch_compile_with_custom_op(monkeypatch):
    # Ensure env toggles take effect for this test only.
    # The bytecode hook is required to detect buffer mutation in compiled code,
    # and AOT compile bypasses that hook entirely.
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "1")
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")

    device = DEVICE_TYPE
    positions = torch.arange(16, device=device)
    query = torch.randn(16, 32, device=device, dtype=torch.bfloat16)
    key = torch.randn(16, 32, device=device, dtype=torch.bfloat16)

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            custom_ops=["+rotary_embedding"],
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_num_of_warmups=0,
        ),
    )

    with set_current_vllm_config(vllm_config):
        model = RotaryEmbeddingCompileModule(vllm_config=vllm_config)
        model(positions, query, key)
        assert model._compiled_bytecode is not None
        assert "update" not in model._compiled_bytecode.co_names


class _DeepseekRopeModule(torch.nn.Module):
    def __init__(self, rotary_emb: DeepseekScalingRotaryEmbedding) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.rotary_emb(positions, query, key, offsets)


def _yarn_table_ops(graph: torch.fx.Graph) -> list[str]:
    """YaRN table ops that must stay out of the decode graph."""
    present = []
    for op in (
        torch.ops.aten.reciprocal,
        torch.ops.aten.clamp,
        torch.ops.aten.cat,
    ):
        present.extend(
            getattr(node.target, "__name__", str(node.target))
            for node in find_op_nodes(op, graph)
        )
    return present


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm AITER cached RoPE")
@pytest.mark.parametrize("is_neox_style", [False, True])
def test_deepseek_scaling_cached_aiter_rope(
    monkeypatch: pytest.MonkeyPatch, is_neox_style: bool
) -> None:
    """Cached AITER RoPE matches native YaRN, including the offsets fallback.

    VLLM_ROCM_USE_AITER_TRITON_ROPE stays off, matching the K3 recipe. The
    compiled graph must call the cached op and must not rebuild the YaRN table.
    """
    if not is_aiter_found_and_supported():
        pytest.skip("aiter is not found or not supported on this hardware")

    envs.disable_envs_cache()
    device = DEVICE_TYPE
    seq_len = 8
    num_heads = 4
    rope_dim = 64
    max_position = 128
    scaling_factor = 2.0
    dtype = torch.bfloat16
    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            custom_ops=["+rotary_embedding"],
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_num_of_warmups=0,
        ),
    )
    rope_op = torch.ops.vllm.rocm_aiter_triton_rotary_embedding

    try:
        with monkeypatch.context() as m, set_current_vllm_config(vllm_config):
            m.setenv("VLLM_ROCM_USE_AITER", "1")
            m.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "0")
            rocm_aiter_ops.refresh_env_variables()

            rope = DeepseekScalingRotaryEmbedding(
                head_size=rope_dim,
                rotary_dim=rope_dim,
                max_position_embeddings=max_position,
                base=10000,
                is_neox_style=is_neox_style,
                scaling_factor=scaling_factor,
                dtype=dtype,
            ).to(device)
            assert rope.use_aiter_cached_rope
            assert not rope.use_aiter

            cache_len = int(max_position * scaling_factor)
            positions = torch.randint(0, cache_len - 8, (seq_len,), device=device)
            query = torch.randn(
                seq_len, num_heads, rope_dim, dtype=dtype, device=device
            )
            # MLA passes k_pe as [tokens, 1, rope_dim].
            key = torch.randn(seq_len, 1, rope_dim, dtype=dtype, device=device)

            calls = {"n": 0}
            real_op = rope.rocm_aiter_triton_rotary_embedding

            def _counting_op(*args, **kwargs):
                calls["n"] += 1
                return real_op(*args, **kwargs)

            rope.rocm_aiter_triton_rotary_embedding = _counting_op

            ref_q, ref_k = rope.forward_native(positions, query.clone(), key.clone())
            out_q, out_k = rope(positions, query.clone(), key.clone())
            assert calls["n"] == 1
            assert ref_k is not None and out_k is not None
            assert_rope_close_rocm(
                out_q, ref_q, query, rope.cos_sin_cache, positions, is_neox_style
            )
            assert_rope_close_rocm(
                out_k, ref_k, key, rope.cos_sin_cache, positions, is_neox_style
            )

            offsets = torch.randint(0, 8, (seq_len,), device=device)
            calls["n"] = 0
            ref_q, ref_k = rope.forward_native(
                positions, query.clone(), key.clone(), offsets
            )
            out_q, out_k = rope(positions, query.clone(), key.clone(), offsets)
            assert calls["n"] == 0
            assert ref_k is not None and out_k is not None
            torch.testing.assert_close(out_q, ref_q)
            torch.testing.assert_close(out_k, ref_k)

            rope.rocm_aiter_triton_rotary_embedding = real_op
            module = _DeepseekRopeModule(rope)
            torch._dynamo.reset()
            backend = TestBackend()
            compiled = torch.compile(module, backend=backend)
            compiled(positions, query.clone(), key.clone())
            pre_graph = backend.graph_pre_compile.graph
            assert len(list(find_op_nodes(rope_op, pre_graph))) == 1
            assert _yarn_table_ops(pre_graph) == []

            torch._dynamo.reset()
            offset_backend = TestBackend()
            compiled_offsets = torch.compile(module, backend=offset_backend)
            compiled_offsets(positions, query.clone(), key.clone(), offsets)
            offset_graph = offset_backend.graph_pre_compile.graph
            assert len(list(find_op_nodes(rope_op, offset_graph))) == 0
            assert len(list(find_op_nodes(torch.ops.aten.mul, offset_graph))) > 0
    finally:
        torch._dynamo.reset()
        rocm_aiter_ops.refresh_env_variables()
