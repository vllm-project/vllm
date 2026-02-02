# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

import vllm.config
import vllm.plugins
from tests.compile.backend import TestBackend
from tests.utils import TestBlockFP8Layer, TestFP8Layer
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.compilation.passes.fusion.matcher_utils import QUANT_OPS
from vllm.compilation.passes.fusion.rms_quant_fusion import (
    FUSED_OPS,
    FusedRMSQuantKey,
    RMSNormQuantFusionPass,
    fused_add_rms_norm_nvfp4_quant_supported,
    rms_norm_nvfp4_quant_supported,
)
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.rocm import (
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    FP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kNvfp4Dynamic,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

# Kernel and group_shape combinations: (kernel, group_shape)
# CUDA kernels
CUDA_KERNEL_GROUPSHAPE_COMBINATIONS = [
    # FlashInferFP8ScaledMMLinearKernel supports both per-tensor only
    (FlashInferFP8ScaledMMLinearKernel, GroupShape.PER_TENSOR),
    # CutlassFP8ScaledMMLinearKernel supports both per-tensor and per-token
    (CutlassFP8ScaledMMLinearKernel, GroupShape.PER_TOKEN),
    (CutlassFP8ScaledMMLinearKernel, GroupShape.PER_TENSOR),
    # PerTensorTorchFP8ScaledMMLinearKernel only supports per-tensor
    (PerTensorTorchFP8ScaledMMLinearKernel, GroupShape.PER_TENSOR),
    # ChannelWiseTorchFP8ScaledMMLinearKernel only supports per-token
    (ChannelWiseTorchFP8ScaledMMLinearKernel, GroupShape.PER_TOKEN),
    # Blockwise group shapes (no kernel abstraction)
    (None, GroupShape(1, 128)),
    (None, GroupShape(1, 64)),
]

# ROCm kernels
ROCM_KERNEL_GROUPSHAPE_COMBINATIONS = [
    # ROCmFP8ScaledMMLinearKernel supports per-tensor only
    (ROCmFP8ScaledMMLinearKernel, GroupShape.PER_TENSOR),
    # RowWiseTorchFP8ScaledMMLinearKernel only supports per-token
    (RowWiseTorchFP8ScaledMMLinearKernel, GroupShape.PER_TOKEN),
    # ChannelWiseTorchFP8ScaledMMLinearKernel only supports per-token
    (ChannelWiseTorchFP8ScaledMMLinearKernel, GroupShape.PER_TOKEN),
    # Blockwise group shapes (no kernel abstraction)
    (None, GroupShape(1, 128)),
    (None, GroupShape(1, 64)),
]

KERNEL_GROUPSHAPE_COMBINATIONS = (
    CUDA_KERNEL_GROUPSHAPE_COMBINATIONS
    if current_platform.is_cuda()
    else ROCM_KERNEL_GROUPSHAPE_COMBINATIONS
)

# For Aiter tests we toggle use_aiter_quant_op
AITER_KERNEL_GROUPSHAPE_COMBINATIONS = [
    # Per-token with ROCmFP8ScaledMMLinearKernel
    (ROCmFP8ScaledMMLinearKernel, GroupShape.PER_TENSOR, False),
    # Per-token with RowWiseTorchFP8ScaledMMLinearKernel
    (RowWiseTorchFP8ScaledMMLinearKernel, GroupShape.PER_TOKEN, True),
    (RowWiseTorchFP8ScaledMMLinearKernel, GroupShape.PER_TOKEN, False),
    # Per-token with ChannelWiseTorchFP8ScaledMMLinearKernel
    (ChannelWiseTorchFP8ScaledMMLinearKernel, GroupShape.PER_TOKEN, True),
    (ChannelWiseTorchFP8ScaledMMLinearKernel, GroupShape.PER_TOKEN, False),
    # Blockwise (no kernel abstraction)
    (None, GroupShape(1, 128), True),
]


class TestModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        force_kernel: FP8ScaledMMLinearKernel | None,
        group_shape: GroupShape,
        use_aiter_fusion: bool = False,
        use_aiter_quant: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fp8_linear_layers: list[torch.nn.Module]
        self.group_shape = group_shape
        self.use_aiter_quant_op = use_aiter_quant
        self.use_aiter_fusion = use_aiter_fusion
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]
        self.enable_rms_norm_custom_op = self.norm[0].enabled()

        # Determine if blockwise based on group_shape
        is_blockwise = group_shape.is_per_group()

        if is_blockwise:
            act_quant_scale_desc = ScaleDesc(torch.float32, False, group_shape)
            self.activation_quant_key = QuantKey(
                dtype=FP8_DTYPE, scale=act_quant_scale_desc, symmetric=True
            )
            self.fp8_linear_layers = [
                TestBlockFP8Layer(
                    weight_shape=(hidden_size, hidden_size),
                    group_shape=group_shape,
                    cutlass_block_fp8_supported=cutlass_block_fp8_supported(),
                    use_aiter_and_is_supported=use_aiter_quant,
                    transpose_weights=use_aiter_fusion,
                )
                for _ in range(3)
            ]

            self.enable_quant_fp8_custom_op = (
                False
                if use_aiter_quant
                else self.fp8_linear_layers[0].linear_op.input_quant_op.enabled()
            )

        else:
            is_static = group_shape == GroupShape.PER_TENSOR
            act_quant_scale_desc = ScaleDesc(torch.float32, is_static, group_shape)
            w_quant_scale_desc = ScaleDesc(torch.float32, True, group_shape)
            self.activation_quant_key = QuantKey(
                dtype=FP8_DTYPE, scale=act_quant_scale_desc, symmetric=True
            )
            self.weight_quant_key = QuantKey(
                dtype=FP8_DTYPE, scale=w_quant_scale_desc, symmetric=True
            )
            self.fp8_linear_layers = [
                TestFP8Layer(
                    weight_shape=(hidden_size, hidden_size),
                    activation_quant_key=self.activation_quant_key,
                    weight_quant_key=self.weight_quant_key,
                    force_kernel=force_kernel,
                )
                for _ in range(3)
            ]

            # Enable aiter quantization if requested
            for layer in self.fp8_linear_layers:
                layer.kernel.quant_fp8.use_aiter = use_aiter_quant

            self.enable_quant_fp8_custom_op = self.fp8_linear_layers[
                0
            ].is_quant_fp8_enabled()

    def forward(self, x):
        # avoid having graph input be an arg to a pattern directly
        x = resid = torch.relu(x)
        y = self.norm[0](x)

        x2 = self.fp8_linear_layers[0](y)
        # make sure resid is used for replacement to work
        y2, resid = self.norm[1](x2, resid)

        x3 = self.fp8_linear_layers[1](y2)

        y3, resid = self.norm[2](x3, resid)  # use resid here

        x4 = self.fp8_linear_layers[2](y3)

        y4, resid = self.norm[3](x4, resid)  # use resid here
        return y4

    def ops_in_model_before(self):
        if self.group_shape.is_per_group():
            # Blockwise path
            if self.use_aiter_fusion and self.use_aiter_quant_op:
                return [rocm_aiter_ops.get_group_quant_op()]
            if self.use_aiter_fusion:
                return [torch.ops.vllm.triton_per_token_group_quant_fp8.default]
        else:
            if self.use_aiter_quant_op:
                return [rocm_aiter_ops.get_per_token_quant_op()]

        # Common path
        return (
            [QUANT_OPS[self.activation_quant_key]]
            if self.enable_quant_fp8_custom_op
            else [torch.ops.aten.reciprocal]
        )

    def ops_in_model_after(self):
        if self.use_aiter_fusion:
            if self.group_shape.is_per_group():
                # Blockwise aiter fusion
                from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
                    AiterFusedAddRMSFp8GroupQuantPattern,
                    AiterRMSFp8GroupQuantPattern,
                )

                return [
                    AiterFusedAddRMSFp8GroupQuantPattern.FUSED_OP,
                    AiterRMSFp8GroupQuantPattern.FUSED_OP,
                ]
            else:
                # Per-token aiter fusion
                from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
                    AiterFusedAddRMSNormDynamicQuantPattern,
                    AiterRMSNormDynamicQuantPattern,
                )

                return [
                    AiterFusedAddRMSNormDynamicQuantPattern.FUSED_OP,
                    AiterRMSNormDynamicQuantPattern.FUSED_OP,
                ]

        # Regular fusion
        return [
            FUSED_OPS[FusedRMSQuantKey(self.activation_quant_key, True)],
            FUSED_OPS[FusedRMSQuantKey(self.activation_quant_key, False)],
        ]

    def ops_in_model_before_partial(self):
        return (
            [RMS_OP, RMS_ADD_OP]
            if self.enable_rms_norm_custom_op
            else [torch.ops.aten.rsqrt]
        )


def _run_fusion_test(
    model,
    fusion_pass,
    vllm_config,
    dtype,
    hidden_size,
    num_tokens,
):
    """Helper function for common fusion test logic.

    Must be called within vllm_config context.
    """
    noop_pass = NoOpEliminationPass(vllm_config)
    cleanup_pass = PostCleanupPass(vllm_config)

    backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)
    backend2 = TestBackend(noop_pass, cleanup_pass)

    x = torch.rand(num_tokens, hidden_size)
    torch._dynamo.mark_dynamic(x, 0)

    model_fused = torch.compile(model, backend=backend)
    result_fused = model_fused(x)

    model_unfused = torch.compile(model, backend=backend2)
    result_unfused = model_unfused(x)

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(result_fused, result_unfused, atol=ATOL, rtol=RTOL)

    assert fusion_pass.matched_count == 3
    backend.check_before_ops(model.ops_in_model_before())
    backend.check_after_ops(model.ops_in_model_after())

    return backend, backend2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("kernel_groupshape", KERNEL_GROUPSHAPE_COMBINATIONS)
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.parametrize("enable_quant_fp8_custom_op", [True, False])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only test on CUDA and ROCm"
)
def test_fusion_rmsnorm_quant(
    dtype,
    hidden_size,
    num_tokens,
    eps,
    kernel_groupshape,
    enable_rms_norm_custom_op,
    enable_quant_fp8_custom_op,
):
    force_kernel, group_shape = kernel_groupshape

    if not enable_quant_fp8_custom_op and group_shape.is_per_group():
        pytest.skip("Unsupported unwrapped quant fp8 op for blockwise quantization")

    if group_shape == GroupShape(1, 64) and (
        cutlass_block_fp8_supported() or is_deep_gemm_supported()
    ):
        pytest.skip("Unsupported group shape 64 for CUTLASS/DeepGemm")

    custom_ops = []
    if enable_rms_norm_custom_op:
        custom_ops.append("+rms_norm")
    if enable_quant_fp8_custom_op:
        custom_ops.append("+quant_fp8")

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(
                fuse_norm_quant=True, fuse_act_quant=True, eliminate_noops=True
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        # Setup device before model creation
        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(1)

        fusion_pass = RMSNormQuantFusionPass(vllm_config)

        model = TestModel(
            hidden_size=hidden_size,
            eps=eps,
            force_kernel=force_kernel,
            group_shape=group_shape,
            use_aiter_fusion=False,
            use_aiter_quant=False,
        )

        backend, _ = _run_fusion_test(
            model, fusion_pass, vllm_config, dtype, hidden_size, num_tokens
        )
        backend.check_before_ops(
            model.ops_in_model_before_partial(), fully_replaced=False
        )

        # If RMSNorm custom op is disabled (native/torch impl used),
        # there's a risk that the fused add doesn't get included in the
        # replacement and only the rms part gets fused with quant.
        # Hence, we check only 2 add nodes are left (final fused rmsnorm add).
        if not enable_rms_norm_custom_op:
            n_add_nodes = lambda g: sum(1 for _ in find_op_nodes(torch.ops.aten.add, g))
            # 7 = 1 (RMS) + 3x2 (3xRMS_ADD, 2 each)
            assert n_add_nodes(backend.graph_pre_pass) == 7
            assert n_add_nodes(backend.graph_post_pass) == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize(
    "kernel_groupshape_quant", AITER_KERNEL_GROUPSHAPE_COMBINATIONS
)
@pytest.mark.skipif(
    (not current_platform.is_rocm() or not IS_AITER_FOUND),
    reason="Only test on ROCm with aiter package installed",
)
def test_aiter_fusion_rmsnorm_quant(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    eps: float,
    kernel_groupshape_quant: tuple,
    monkeypatch: pytest.MonkeyPatch,
):
    force_kernel, group_shape, use_aiter_quant_op = kernel_groupshape_quant
    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(fuse_norm_quant=True, eliminate_noops=True),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            RocmAiterRMSNormQuantFusionPass,
        )

        m.setenv("VLLM_ROCM_USE_AITER", "1")

        rocm_aiter_ops.refresh_env_variables()

        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(1)

        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)

        model = TestModel(
            hidden_size=hidden_size,
            eps=eps,
            force_kernel=force_kernel,
            group_shape=group_shape,
            use_aiter_fusion=True,  # Always use aiter fusion ops in aiter test
            use_aiter_quant=use_aiter_quant_op,  # Toggle aiter quantization
        )

        _run_fusion_test(
            model, fusion_pass, vllm_config, dtype, hidden_size, num_tokens
        )


def is_nvfp4_supported():
    return current_platform.has_device_capability(100)


def quant_nvfp4_tensor(a: torch.Tensor):
    from vllm.scalar_type import scalar_types

    FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(a).max().to(
        torch.float32
    )
    a_quant, a_block_scale = scaled_fp4_quant(a, a_global_scale)
    return a_quant, a_block_scale, a_global_scale


class TestRMSNormNvfp4QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float, x: torch.Tensor):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.enable_rms_norm_custom_op = self.norm.enabled()

        w = torch.rand((hidden_size, hidden_size))
        self.w, self.w_block_scale, self.w_global_scale = quant_nvfp4_tensor(w)

        y = self.norm(x)
        _, _, self.y_global_scale = quant_nvfp4_tensor(y)
        self.alpha = 1.0 / (self.w_global_scale * self.y_global_scale)

    def forward(self, x):
        y = self.norm(x)
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
            RMS_OP if self.enable_rms_norm_custom_op else torch.ops.aten.rsqrt,
            QUANT_OPS[kNvfp4Dynamic],
        ]

    def ops_in_model_after(self):
        return [FUSED_OPS[FusedRMSQuantKey(kNvfp4Dynamic, False)]]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA")
def test_fusion_rmsnorm_nvfp4_quant(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    eps: float,
    enable_rms_norm_custom_op: bool,
):
    if not is_nvfp4_supported():
        pytest.skip("NVFP4 is not supported on this GPU.")

    if not rms_norm_nvfp4_quant_supported:
        pytest.skip("rms_norm_nvfp4_quant op is not available.")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    x = torch.rand(num_tokens, hidden_size)

    custom_ops = ["none"]
    if enable_rms_norm_custom_op:
        custom_ops.append("+rms_norm")

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            backend="eager",
            pass_config=PassConfig(fuse_norm_quant=True, eliminate_noops=True),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RMSNormQuantFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)

        model = TestRMSNormNvfp4QuantModel(hidden_size, eps, x)

        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        atol, rtol = 1e-1, 1e-1
        torch.testing.assert_close(result, result2, atol=atol, rtol=rtol)

        assert fusion_pass.matched_count == 1
        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())


class TestFusedAddRMSNormNvfp4QuantModel(torch.nn.Module):
    """Test model for fused_add_rms_norm + nvfp4 quant fusion."""

    def __init__(self, hidden_size: int, eps: float, x: torch.Tensor):
        super().__init__()
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(3)]
        self.enable_rms_norm_custom_op = self.norm[0].enabled()

        w = torch.rand((hidden_size, hidden_size))
        self.w, self.w_block_scale, self.w_global_scale = quant_nvfp4_tensor(w)

        # Compute global scale from a sample forward pass
        resid = torch.zeros_like(x)
        y, _ = self.norm[0](x, resid)
        _, _, self.y_global_scale = quant_nvfp4_tensor(y)
        self.alpha = 1.0 / (self.w_global_scale * self.y_global_scale)

    def forward(self, x):
        # Avoid having graph input be an arg to a pattern directly
        x = torch.relu(x)
        resid = torch.tanh(x)

        # First: fused_add_rms_norm + nvfp4 quant
        y, resid = self.norm[0](x, resid)
        y_quant, y_block_scale = scaled_fp4_quant(y, self.y_global_scale)
        out1 = cutlass_scaled_fp4_mm(
            a=y_quant,
            b=self.w,
            block_scale_a=y_block_scale,
            block_scale_b=self.w_block_scale,
            alpha=self.alpha,
            out_dtype=y.dtype,
        )

        # Second: fused_add_rms_norm + nvfp4 quant
        y2, resid = self.norm[1](out1, resid)
        y2_quant, y2_block_scale = scaled_fp4_quant(y2, self.y_global_scale)
        out2 = cutlass_scaled_fp4_mm(
            a=y2_quant,
            b=self.w,
            block_scale_a=y2_block_scale,
            block_scale_b=self.w_block_scale,
            alpha=self.alpha,
            out_dtype=y2.dtype,
        )

        # Third: fused_add_rms_norm + nvfp4 quant
        y3, resid = self.norm[2](out2, resid)
        y3_quant, y3_block_scale = scaled_fp4_quant(y3, self.y_global_scale)
        out3 = cutlass_scaled_fp4_mm(
            a=y3_quant,
            b=self.w,
            block_scale_a=y3_block_scale,
            block_scale_b=self.w_block_scale,
            alpha=self.alpha,
            out_dtype=y3.dtype,
        )

        return out3

    def ops_in_model_before(self):
        return [
            RMS_ADD_OP if self.enable_rms_norm_custom_op else torch.ops.aten.rsqrt,
            QUANT_OPS[kNvfp4Dynamic],
        ]

    def ops_in_model_after(self):
        return [FUSED_OPS[FusedRMSQuantKey(kNvfp4Dynamic, True)]]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA")
def test_fusion_fused_add_rmsnorm_nvfp4_quant(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    eps: float,
    enable_rms_norm_custom_op: bool,
):
    if not is_nvfp4_supported():
        pytest.skip("NVFP4 is not supported on this GPU.")

    if not fused_add_rms_norm_nvfp4_quant_supported:
        pytest.skip("fused_add_rms_norm_nvfp4_quant op is not available.")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    x = torch.rand(num_tokens, hidden_size)

    custom_ops = ["none"]
    if enable_rms_norm_custom_op:
        custom_ops.append("+rms_norm")

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            backend="eager",
            pass_config=PassConfig(fuse_norm_quant=True, eliminate_noops=True),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RMSNormQuantFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)

        model = TestFusedAddRMSNormNvfp4QuantModel(hidden_size, eps, x)

        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        atol, rtol = 1e-1, 1e-1
        torch.testing.assert_close(result, result2, atol=atol, rtol=rtol)

        assert fusion_pass.matched_count == 2
        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
