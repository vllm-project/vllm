# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.passes import pass_manager
from vllm.config import (
    CompilationConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@support_torch_compile
class RotaryEmbeddingCompileModule(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.rotary_emb = RotaryEmbedding(
            32,
            32,
            128,
            10000.0,
            True,
            torch.float32,
        )

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        q_rot, k_rot = self.rotary_emb(positions, query, key)
        return q_rot + k_rot


@pytest.mark.skipif(current_platform.is_cpu(), reason="Requires GPU for torch.compile")
@pytest.mark.parametrize("use_aiter", [False, True])
def test_rotary_embedding_torch_compile_with_custom_op(monkeypatch, use_aiter):
    if use_aiter and not current_platform.is_rocm():
        pytest.skip("AITER Triton RoPE is ROCm-only")
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

    with monkeypatch.context() as m:
        if use_aiter:
            m.setenv("VLLM_ROCM_USE_AITER", "1")
            m.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "1")
        else:
            m.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "0")
        rocm_aiter_ops.refresh_env_variables()
        if use_aiter:
            # Compile-pass AITER imports are selected at module load.
            importlib.reload(pass_manager)
        with set_current_vllm_config(vllm_config):
            model = RotaryEmbeddingCompileModule(vllm_config=vllm_config).to(device)
            assert bool(model.rotary_emb.use_aiter) == use_aiter
            table = model.rotary_emb.cos_sin_cache
            saved_table = table.clone()
            ref_query, ref_key = query.clone(), key.clone()
            compiled = model(positions, query, key)
            if use_aiter:
                model.rotary_emb.rocm_aiter_triton_rotary_embedding(
                    positions,
                    ref_query,
                    ref_key,
                    model.rotary_emb.head_size,
                    saved_table,
                    model.rotary_emb.is_neox_style,
                )
            else:
                ops.rotary_embedding(
                    positions,
                    ref_query,
                    ref_key,
                    model.rotary_emb.head_size,
                    saved_table,
                    model.rotary_emb.is_neox_style,
                )
            expected = ref_query + ref_key
            torch.testing.assert_close(compiled, expected, atol=1e-3, rtol=1e-3)
            assert compiled.dtype == torch.bfloat16
            assert model.rotary_emb.cos_sin_cache is table
            assert model.rotary_emb.cos_sin_cache.dtype == torch.float32
            assert torch.equal(table, saved_table)
            assert model._compiled_bytecode is not None
            assert "update" not in model._compiled_bytecode.co_names
    rocm_aiter_ops.refresh_env_variables()
