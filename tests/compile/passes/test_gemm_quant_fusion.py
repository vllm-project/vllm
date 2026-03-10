# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm import _custom_ops as ops
from vllm.compilation.passes.fusion.gemm_quant_fusion import GemmQuantFusionPass
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()


def _make_weight(hidden_size: int) -> torch.Tensor:
    return torch.randn((hidden_size, hidden_size), device="cuda").to(FP8_DTYPE).t()


class _BaseGemmQuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, mm_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.register_buffer("weight", _make_weight(hidden_size))
        self.register_buffer(
            "scale_b",
            torch.ones((1, hidden_size), device="cuda", dtype=torch.float32),
        )
        self.register_buffer(
            "output_scale",
            torch.ones(1, device="cuda", dtype=torch.float32),
        )
        self.mm_dtype = mm_dtype

    def _gemm(self, x: torch.Tensor) -> torch.Tensor:
        scale_a = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float32)
        return ops.cutlass_scaled_mm(
            x, self.weight, scale_a, self.scale_b, self.mm_dtype, None
        )


class GemmStaticFP8QuantModel(_BaseGemmQuantModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mm_out = self._gemm(x)
        quant_out, _ = ops.scaled_fp8_quant(mm_out, scale=self.output_scale)
        return quant_out


class GemmDynamicFP8QuantModel(_BaseGemmQuantModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mm_out = self._gemm(x)
        quant_out, _ = ops.scaled_fp8_quant(mm_out)
        return quant_out


class GemmPerTokenFP8QuantModel(_BaseGemmQuantModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mm_out = self._gemm(x)
        scale = torch.ones((mm_out.shape[0], 1), device=x.device, dtype=torch.float32)
        quant_out, _ = ops.scaled_fp8_quant(
            mm_out, scale=scale, group_shape=(1, -1)
        )
        return quant_out


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.supports_fp8(),
    reason="Requires CUDA FP8 support",
)
@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="Fused CUTLASS FP8 output quantization requires SM89+",
)
@pytest.mark.parametrize(
    "model_cls, should_fuse",
    [
        (GemmStaticFP8QuantModel, True),
        (GemmDynamicFP8QuantModel, False),
        (GemmPerTokenFP8QuantModel, False),
    ],
)
def test_gemm_quant_fusion_pass(model_cls: type[torch.nn.Module], should_fuse: bool):
    config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+quant_fp8"],
            backend="eager",
            pass_config=PassConfig(fuse_gemm_quant=True, eliminate_noops=True),
        ),
    )

    with set_current_vllm_config(config):
        model = model_cls(hidden_size=128).cuda()
        x = torch.randn((32, 128), device="cuda").to(FP8_DTYPE)

        passes = [
            NoOpEliminationPass(config),
            GemmQuantFusionPass(config),
            PostCleanupPass(config),
        ]
        backend = TestBackend(*passes)

        compiled = torch.compile(model, fullgraph=True, backend=backend)
        out = compiled(x)
        assert out.dtype == FP8_DTYPE

        if should_fuse:
            backend.check_before_ops(
                [
                    torch.ops._C.cutlass_scaled_mm.default,
                    torch.ops._C.static_scaled_fp8_quant.default,
                ]
            )
            backend.check_after_ops(
                [torch.ops._C.cutlass_scaled_mm_static_fp8_quant.default]
            )
        else:
            assert (
                backend.op_count(torch.ops._C.cutlass_scaled_mm_static_fp8_quant.default)
                == 0
            )
