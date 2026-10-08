# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.kernels.core.test_rotary_embedding_mla_cache_fused import (
    assert_rope_close_rocm,
)
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.config import (
    CompilationConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer, KimiMLAAttention
from vllm.models.kimi_k3.amd.mla import apply_kimi_k3_rope
from vllm.platforms import current_platform


class _DirectMLA(KimiMLAAttention):
    def __init__(self, result: torch.Tensor) -> None:
        nn.Module.__init__(self)
        self.result = result

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.result


def _make_layer(self_attn: nn.Module) -> KimiDecoderLayer:
    layer = object.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = self_attn
    return layer


def test_mla_self_attention_returns_projection_storage_directly() -> None:
    hidden_states = torch.randn(4, 8)
    projected = torch.randn_like(hidden_states)
    layer = _make_layer(_DirectMLA(projected))

    output = layer._run_self_attn(torch.arange(4), hidden_states)

    assert output is projected
    assert output.data_ptr() == projected.data_ptr()


class _K3RopeModule(torch.nn.Module):
    def __init__(self, rotary_emb: torch.nn.Module) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb

    def forward(
        self,
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_kimi_k3_rope(self.rotary_emb, positions, q_pe, k_pe)


def _k3_rope():
    # Same selection K3 MLA makes before get_rope: YaRN, GPT-J, rope dim 64.
    return get_rope(
        64,
        max_position=256,
        is_neox_style=False,
        dtype=torch.bfloat16,
        rope_parameters={
            "rope_type": "deepseek_yarn",
            "rope_theta": 10000,
            "factor": 2.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Kimi-K3 AITER RoPE")
def test_kimi_k3_rope_uses_cached_aiter_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """K3 calls the cached AITER op. AITER off keeps the module forward."""
    if not is_aiter_found_and_supported():
        pytest.skip("aiter is not found or not supported on this hardware")

    envs.disable_envs_cache()
    device = current_platform.device_type
    seq_len = 8
    rope_dim = 64
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
            rope = _k3_rope().to(device)
            positions = torch.randint(0, 240, (seq_len,), device=device)
            query = torch.randn(seq_len, 4, rope_dim, dtype=dtype, device=device)
            key = torch.randn(seq_len, 1, rope_dim, dtype=dtype, device=device)

            calls = {"n": 0}
            real_get = rocm_aiter_ops.get_triton_rotary_embedding_op

            def _counting_get():
                op = real_get()

                def _wrapped(*args, **kwargs):
                    calls["n"] += 1
                    return op(*args, **kwargs)

                return _wrapped

            monkeypatch.setattr(
                rocm_aiter_ops, "get_triton_rotary_embedding_op", _counting_get
            )
            ref_q, ref_k = rope.forward_native(positions, query.clone(), key.clone())
            out_q, out_k = apply_kimi_k3_rope(
                rope, positions, query.clone(), key.clone()
            )
            assert calls["n"] == 1
            assert ref_k is not None
            assert_rope_close_rocm(
                out_q, ref_q, query, rope.cos_sin_cache, positions, False
            )
            assert_rope_close_rocm(
                out_k, ref_k, key, rope.cos_sin_cache, positions, False
            )
            rocm_aiter_ops.get_triton_rotary_embedding_op = real_get

            torch._dynamo.reset()
            backend = TestBackend()
            compiled = torch.compile(_K3RopeModule(rope), backend=backend)
            compiled(positions, query.clone(), key.clone())
            pre_graph = backend.graph_pre_compile.graph
            assert len(list(find_op_nodes(rope_op, pre_graph))) == 1
            for op in (
                torch.ops.aten.reciprocal,
                torch.ops.aten.clamp,
                torch.ops.aten.cat,
            ):
                assert not any(find_op_nodes(op, pre_graph))

            m.setenv("VLLM_ROCM_USE_AITER", "0")
            rocm_aiter_ops.refresh_env_variables()
            calls["n"] = 0
            ref_q, ref_k = rope.forward_native(positions, query.clone(), key.clone())
            out_q, out_k = apply_kimi_k3_rope(
                rope, positions, query.clone(), key.clone()
            )
            assert calls["n"] == 0
            assert ref_k is not None
            torch.testing.assert_close(out_q, ref_q)
            torch.testing.assert_close(out_k, ref_k)
    finally:
        torch._dynamo.reset()
        rocm_aiter_ops.refresh_env_variables()
