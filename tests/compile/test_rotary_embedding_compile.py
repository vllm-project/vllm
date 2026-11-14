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
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
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


QKV_HEADS, QKV_KV_HEADS, QKV_HEAD_SIZE = 8, 2, 128


@support_torch_compile
class QKVRotaryEmbeddingModule(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.rotary_emb = get_rope(
            head_size=QKV_HEAD_SIZE,
            max_position=4096,
            dtype=torch.bfloat16,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
            is_neox_style=True,
        )
        self.split_sizes = [
            QKV_HEADS * QKV_HEAD_SIZE,
            QKV_KV_HEADS * QKV_HEAD_SIZE,
            QKV_KV_HEADS * QKV_HEAD_SIZE,
        ]

    def forward(
        self, positions: torch.Tensor, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split(self.split_sizes, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="vllm_c is CUDA/ROCm")
@torch.inference_mode()
def test_rotary_embedding_kernel_lowering_has_no_copies(monkeypatch):
    """Lowering the in-place vllm_c RoPE uses its out-of-place functional_impl,
    so a compiled qkv split + RoPE runs the RoPE kernel and nothing else."""
    from torch.profiler import ProfilerActivity, profile

    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            custom_ops=["+rotary_embedding"],
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )
    vllm_config.kernel_config.ir_op_priority.rotary_embedding = ["vllm_c", "native"]

    with (
        set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        model = QKVRotaryEmbeddingModule(vllm_config=vllm_config).to(DEVICE_TYPE)
        positions = torch.randint(0, 4096, (16,), device=DEVICE_TYPE)
        qkv = torch.randn(
            16, sum(model.split_sizes), dtype=torch.bfloat16, device=DEVICE_TYPE
        )
        q, k, _ = (t.clone() for t in qkv.split(model.split_sizes, dim=-1))
        # Reference independent of the IR priority set above.
        ref_q, ref_k = RotaryEmbedding.forward_static(
            positions,
            q,
            k,
            QKV_HEAD_SIZE,
            QKV_HEAD_SIZE,
            model.rotary_emb._match_cos_sin_cache_dtype(q),
            True,
        )

        out_q, out_k, _ = model(positions, qkv)
        torch.accelerator.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(positions, qkv)
            torch.accelerator.synchronize()

    kernels = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    rope = [k for k in kernels if "rotary_embedding_out_kernel" in k]
    # Anything else (e.g. a clone of q/k) is an extra copy; memsets are benign.
    others = [k for k in kernels if k not in rope and "memset" not in k.lower()]
    assert len(rope) == 1 and not others, kernels
    torch.testing.assert_close(out_q, ref_q, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(out_k, ref_k, atol=3e-2, rtol=2e-2)
