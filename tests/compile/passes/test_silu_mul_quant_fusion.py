# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools

import pytest
import torch

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.kernels.quantization.nvfp4_utils import quant_nvfp4_tensor
from tests.utils import TestFP8Layer
from vllm._aiter_ops import IS_AITER_FOUND
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.compilation.passes.fusion.act_quant_fusion import (
    FUSED_OPS,
    SILU_MUL_OP,
    ActivationQuantFusionPass,
)
from vllm.compilation.passes.fusion.rms_quant_fusion import QUANT_OPS
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.pytorch import (
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.rocm import (
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    FP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import W8A8BlockFp8LinearOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def is_nvfp4_supported():
    return current_platform.has_device_capability(100)


class TestSiluMulFp8QuantModel(torch.nn.Module):
    quant_key = kFp8StaticTensorSym

    def __init__(
        self, hidden_size: int, force_kernel: FP8ScaledMMLinearKernel, **kwargs
    ):
        super().__init__()
        self.silu_and_mul = SiluAndMul()

        self.fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
            activation_quant_key=self.quant_key,
            weight_quant_key=self.quant_key,
            force_kernel=force_kernel,
        )

        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()
        self.enable_quant_fp8_custom_op = self.fp8_linear.is_quant_fp8_enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = self.fp8_linear(y)
        return x2

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
            (
                QUANT_OPS[kFp8StaticTensorSym]
                if self.enable_quant_fp8_custom_op
                else torch.ops.aten.reciprocal
            ),
        ]

    def ops_in_model_after(self):
        return [FUSED_OPS[kFp8StaticTensorSym]]


class TestSiluMulNvfp4QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, x: torch.Tensor, **kwargs):
        super().__init__()
        from vllm.compilation.passes.fusion.act_quant_fusion import (
            silu_and_mul_nvfp4_quant_supported,
        )

        assert silu_and_mul_nvfp4_quant_supported

        self.silu_and_mul = SiluAndMul()
        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

        # create nvfp4 weight
        w = torch.rand((hidden_size, hidden_size))
        self.w, self.w_block_scale, self.w_global_scale = quant_nvfp4_tensor(w)

        # get global scale offline
        _, _, self.y_global_scale = quant_nvfp4_tensor(self.silu_and_mul(x))

        self.alpha = 1.0 / (self.w_global_scale * self.y_global_scale)

    def forward(self, x):
        y = self.silu_and_mul(x)
        y_quant, y_block_scale = scaled_fp4_quant(y, self.y_global_scale)
        out = cutlass_scaled_fp4_mm(
            a=y_quant,
            b=self.w,
            block_scale_a=y_block_scale,
            block_scale_b=self.w_block_scale,
            alpha=self.alpha,
            out_dtype=y.dtype,
        )
        return out

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
            QUANT_OPS[kNvfp4Dynamic],
        ]

    def ops_in_model_after(self):
        return [FUSED_OPS[kNvfp4Dynamic]]


class TestSiluMulGroupFp8QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
            weight_group_shape=GroupShape(128, 128),
            act_quant_group_shape=GroupShape(1, 128),
            cutlass_block_fp8_supported=False,
            # this parameter cannot always be True,
            # it depends on the VLLM_ROCM_USE_AITER
            # and VLLM_ROCM_USE_AITER_LINEAR environment variables
            use_aiter_and_is_supported=True,
        )
        self.w = torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()

        scale_hidden_size = (hidden_size + 128 - 1) // 128
        self.wscale = torch.rand(
            (scale_hidden_size, scale_hidden_size), dtype=torch.float32
        )

        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = self.w8a8_block_fp8_linear.apply(y, self.w, self.wscale)
        return x2

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
        ]

    def ops_in_model_after(self):
        return [torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant]


ROCM_KERNELS = [ROCmFP8ScaledMMLinearKernel, PerTensorTorchFP8ScaledMMLinearKernel]
CUDA_KERNELS = [
    FlashInferFP8ScaledMMLinearKernel,
    CutlassFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
]
TEST_KERNELS = ROCM_KERNELS if current_platform.is_rocm() else CUDA_KERNELS


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
@pytest.mark.parametrize(
    "model_class, enable_quant_fp8_custom_op, force_kernel",
    list(itertools.product([TestSiluMulFp8QuantModel], [True, False], TEST_KERNELS))
    + [
        (TestSiluMulNvfp4QuantModel, False, None),
        (TestSiluMulGroupFp8QuantModel, True, None),
    ],
)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_fusion_silu_and_mul_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    model_class: type[
        TestSiluMulFp8QuantModel
        | TestSiluMulNvfp4QuantModel
        | TestSiluMulGroupFp8QuantModel
    ],
    enable_silu_mul_custom_op: bool,
    enable_quant_fp8_custom_op: bool,
    force_kernel: FP8ScaledMMLinearKernel | None,
    monkeypatch: pytest.MonkeyPatch,
):
    if model_class is TestSiluMulNvfp4QuantModel and not is_nvfp4_supported():
        pytest.skip("NVFP4 is not supported on this GPU.")
    if model_class is TestSiluMulGroupFp8QuantModel and not IS_AITER_FOUND:
        pytest.skip("AITER is not supported on this GPU.")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    x = torch.rand(num_tokens, hidden_size * 2)

    # Reshape pass is needed for the fusion pass to work
    custom_ops = ["none"]
    if enable_silu_mul_custom_op:
        custom_ops.append("+silu_and_mul")
    if enable_quant_fp8_custom_op:
        custom_ops.append("+quant_fp8")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            backend="eager",  # avoid compilation for SiluAndMul and QuantFP8
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        ),
    )

    with set_current_vllm_config(config), monkeypatch.context() as m:
        fusion_passes = [ActivationQuantFusionPass(config)]
        if IS_AITER_FOUND and model_class is TestSiluMulGroupFp8QuantModel:
            from vllm._aiter_ops import rocm_aiter_ops
            from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
                RocmAiterSiluMulFp8GroupQuantFusionPass,
            )

            m.setenv("VLLM_ROCM_USE_AITER", "1")
            rocm_aiter_ops.refresh_env_variables()
            fusion_passes += [RocmAiterSiluMulFp8GroupQuantFusionPass(config)]

        passes = [NoOpEliminationPass(config), *fusion_passes, PostCleanupPass(config)]
        backend = TestBackend(*passes)
        model = model_class(hidden_size=hidden_size, force_kernel=force_kernel, x=x)

        # First dimension dynamic
        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        # Check that it gives the same answer
        if model_class == TestSiluMulFp8QuantModel:
            atol, rtol = 1e-3, 1e-3
        elif model_class == TestSiluMulNvfp4QuantModel:
            atol, rtol = 1e-1, 1e-1
        elif model_class == TestSiluMulGroupFp8QuantModel:
            atol, rtol = 5e-2, 5e-2

        torch.testing.assert_close(
            result[0].to(dtype=dtype), result2[0].to(dtype=dtype), atol=atol, rtol=rtol
        )

        assert sum([p.matched_count for p in fusion_passes]) == 1

        # In pre-nodes, quant op should be present and fused kernels should not
        backend.check_before_ops(model.ops_in_model_before())

        # In post-nodes, fused kernels should be present and quant op should not
        backend.check_after_ops(model.ops_in_model_after())
