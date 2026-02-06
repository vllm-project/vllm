# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

import vllm.config
import vllm.plugins
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.compilation.fusion import FUSED_OPS, FusedRMSQuantKey, RMSNormQuantFusionPass
from vllm.compilation.fx_utils import find_op_nodes
from vllm.compilation.matcher_utils import QUANT_OPS
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (  # noqa: E501
    _KernelT,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cuda import (
    CudaFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
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
from vllm.model_executor.layers.quantization.kernels.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

from ..utils import TestFP8Layer
from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()

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
    # Blockwise group shapes
    (CudaFp8BlockScaledMMKernel, GroupShape(1, 128)),
    (CudaFp8BlockScaledMMKernel, GroupShape(1, 64)),
    (CutlassFp8BlockScaledMMKernel, GroupShape(1, 128)),
    (CutlassFp8BlockScaledMMKernel, GroupShape(1, 64)),
    (TritonFp8BlockScaledMMKernel, GroupShape(1, 128)),
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
    (TritonFp8BlockScaledMMKernel, GroupShape(1, 128)),
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
    # Blockwise
    (AiterFp8BlockScaledMMKernel, GroupShape(1, 128), True),
]


class TestModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        force_kernel: type[_KernelT] | None,
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
            block_size = group_shape.col
            self.activation_quant_key = create_fp8_quant_key(
                static=False, group_shape=group_shape
            )
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(block_size, block_size)
            )

        else:
            is_static = group_shape == GroupShape.PER_TENSOR
            self.activation_quant_key = create_fp8_quant_key(
                is_static, group_shape=group_shape
            )
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=group_shape
            )

        self.fp8_linear_layers = [
            TestFP8Layer(
                weight_shape=(hidden_size, hidden_size),
                activation_quant_key=self.activation_quant_key,
                weight_quant_key=self.weight_quant_key,
                force_kernel=force_kernel,
                transpose_weights=use_aiter_fusion,
            )
            for _ in range(3)
        ]

        # Enable aiter quantization if requested
        for layer in self.fp8_linear_layers:
            layer.kernel.input_quant_op.use_aiter = use_aiter_quant

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
                from vllm.compilation.rocm_aiter_fusion import (
                    AiterFusedAddRMSFp8GroupQuantPattern,
                    AiterRMSFp8GroupQuantPattern,
                )

                return [
                    AiterFusedAddRMSFp8GroupQuantPattern.FUSED_OP,
                    AiterRMSFp8GroupQuantPattern.FUSED_OP,
                ]
            else:
                # Per-token aiter fusion
                from vllm.compilation.rocm_aiter_fusion import (
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
        from vllm.compilation.rocm_aiter_fusion import RocmAiterRMSNormQuantFusionPass

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
