# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from functools import partial

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
from vllm.model_executor.kernels.linear import (
    CutlassFP8ScaledMMLinearKernel,
    FlashInferFP8ScaledMMLinearKernel,
    FP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def is_nvfp4_supported():
    return current_platform.has_device_capability(100)


class TestSiluMulFp8QuantModel(torch.nn.Module):
    quant_key = kFp8StaticTensorSym

    def __init__(
        self,
        hidden_size: int,
        force_kernel: FP8ScaledMMLinearKernel,
        dtype: torch.dtype,
        **kwargs,
    ):
        super().__init__()
        self.silu_and_mul = SiluAndMul()

        self.fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
            activation_quant_key=self.quant_key,
            weight_quant_key=self.quant_key,
            force_kernel=force_kernel,
            input_dtype=dtype,
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
    act_quant_key = kFp8Dynamic128Sym

    def __init__(self, hidden_size: int, dtype: torch.dtype, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.weight_quant_key = create_fp8_quant_key(
            static=True, group_shape=GroupShape(hidden_size, hidden_size)
        )

        self.w8a8_block_fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
            weight_quant_key=self.weight_quant_key,
            activation_quant_key=self.act_quant_key,
            input_dtype=dtype,
        )
        self.w = torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()

        scale_hidden_size = (hidden_size + 128 - 1) // 128
        self.wscale = torch.rand(
            (scale_hidden_size, scale_hidden_size), dtype=torch.float32
        )

        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = self.w8a8_block_fp8_linear(y, self.w, self.wscale)
        return x2

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
        ]

    def ops_in_model_after(self):
        return [torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant]


class TestSiluMulBlockQuantModel(torch.nn.Module):
    quant_key = kFp8Dynamic128Sym

    def __init__(self, hidden_size: int, is_scale_transposed: bool = False, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.is_scale_transposed = is_scale_transposed
        self.quant_fp8 = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            column_major_scales=is_scale_transposed,
            compile_native=False,
        )

        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()
        self.enable_quant_fp8_custom_op = self.quant_fp8.enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        out, scale = self.quant_fp8(y)
        group_size = self.quant_key.scale.group_shape[1]
        scale_expanded = scale.repeat_interleave(group_size, dim=1)
        dequant = out.to(dtype=torch.float32) * scale_expanded
        return (dequant,)

    def ops_in_model_before(self):
        ops = []
        if self.enable_silu_mul_custom_op:
            ops.append(SILU_MUL_OP)
        # When silu custom op is disabled, aten.mul.Tensor also appears
        # in dequant code, so we skip checking it to avoid false positives.
        ops.append(
            QUANT_OPS[self.quant_key]
            if self.enable_quant_fp8_custom_op
            else torch.ops.aten.reciprocal.default
        )
        return ops

    def ops_in_model_after(self):
        return [FUSED_OPS[self.quant_key]]


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
        pytest.param(
            TestSiluMulNvfp4QuantModel,
            False,
            None,
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(), reason="CUDA only"
            ),
        ),
        # GroupFP8Quant fusion only works with AITER on ROCm.
        # and the enable_quant_fp8_custom_op must be True.
        pytest.param(
            TestSiluMulGroupFp8QuantModel,
            True,
            None,
            marks=pytest.mark.skipif(
                not current_platform.is_rocm(), reason="ROCm only"
            ),
        ),
        # Block quant fusion for per-group FP8 (CUDA only).
        *[
            pytest.param(
                partial(TestSiluMulBlockQuantModel, is_scale_transposed=transposed),
                True,
                None,
                marks=pytest.mark.skipif(
                    not current_platform.is_cuda(), reason="CUDA only"
                ),
                id=f"TestSiluMulBlockQuant-transposed={transposed}",
            )
            for transposed in [False, True]
        ],
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
        | TestSiluMulBlockQuantModel
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
    if (
        isinstance(model_class, partial)
        and model_class.func is TestSiluMulBlockQuantModel
        and is_deep_gemm_supported()
    ):
        pytest.skip("SiluMul+BlockQuant fusion not applicable with DeepGemm")

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
        model = model_class(
            hidden_size=hidden_size, force_kernel=force_kernel, x=x, dtype=dtype
        )

        # First dimension dynamic
        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        # Check that it gives the same answer
        if isinstance(model, TestSiluMulFp8QuantModel):
            atol, rtol = 1e-3, 1e-3
        elif isinstance(model, TestSiluMulNvfp4QuantModel):
            atol, rtol = 1e-1, 1e-1
        elif isinstance(
            model, (TestSiluMulGroupFp8QuantModel, TestSiluMulBlockQuantModel)
        ):
            atol, rtol = 5e-2, 5e-2

        torch.testing.assert_close(
            result[0].to(dtype=dtype), result2[0].to(dtype=dtype), atol=atol, rtol=rtol
        )

        assert sum([p.matched_count for p in fusion_passes]) == 1

        # In pre-nodes, quant op should be present and fused kernels should not
        backend.check_before_ops(model.ops_in_model_before())

        # In post-nodes, fused kernels should be present and quant op should not
        backend.check_after_ops(model.ops_in_model_after())
