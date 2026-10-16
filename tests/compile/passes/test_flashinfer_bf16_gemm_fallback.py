# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.compilation.passes.utility.flashinfer_bf16_gemm_fallback import (
    FlashInferBF16GemmFallbackPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    KernelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.utils import cuda_flashinfer_bf16_gemm
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


def cutedsl_vllm_config() -> VllmConfig:
    return VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
        kernel_config=KernelConfig(linear_backend="flashinfer_cutedsl"),
    )


@pytest.mark.parametrize("m,applicable", [(32, False), (33, True)])
def test_flashinfer_bf16_gemm_fallback_range_boundary(m: int, applicable: bool):
    """The configured pass manager must run the fallback pass in the compile range
    holding M=33 but not in the one holding M=32."""
    vllm_config = cutedsl_vllm_config()
    pass_manager = PostGradPassManager()
    pass_manager.configure(vllm_config)
    (fallback_pass,) = [
        p for p in pass_manager.passes if isinstance(p, FlashInferBF16GemmFallbackPass)
    ]
    ranges = vllm_config.compilation_config.get_compile_ranges()
    (compile_range,) = [r for r in ranges if m in r]
    assert fallback_pass.is_applicable_for_range(compile_range) == applicable


@pytest.mark.parametrize("use_bias", [False, True])
def test_flashinfer_bf16_gemm_fallback(use_bias: bool):
    torch.manual_seed(0)
    linear = torch.nn.Linear(
        256, 512, bias=use_bias, dtype=torch.bfloat16, device=DEVICE_TYPE
    ).requires_grad_(False)
    x = torch.randn(64, 256, dtype=torch.bfloat16, device=DEVICE_TYPE)
    torch._dynamo.mark_dynamic(x, 0)

    def gemm(x: torch.Tensor) -> torch.Tensor:
        return cuda_flashinfer_bf16_gemm(
            linear,
            x,
            linear.weight,
            linear.bias,
            vllm_backend="flashinfer_cutedsl",
            pdl=False,
        )

    vllm_config = cutedsl_vllm_config()
    with set_current_vllm_config(vllm_config):
        backend = TestBackend(FlashInferBF16GemmFallbackPass(vllm_config))
        out = torch.compile(gemm, backend=backend)(x)

    backend.check_before_ops([torch.ops.vllm.cuda_flashinfer_bf16_gemm.default])
    mm_op = torch.ops.aten.addmm.default if use_bias else torch.ops.aten.mm.default
    backend.check_after_ops([mm_op])
    torch.testing.assert_close(out, linear(x))
