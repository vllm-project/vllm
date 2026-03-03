# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform


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

    device = "cuda"
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
