# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.plugins
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
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.flashinfer import (
    FlashInferScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.pytorch import (
    ChannelWiseTorchScaledMMLinearKernel,
    PerTensorTorchScaledMMLinearKernel,
    RowWiseTorchScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.rocm import (
    ROCmScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
    FP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.model_executor.layers.quantization.utils.w8a8_utils import cutlass_block_fp8_supported

from ..utils import TestFP8Layer
from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()

RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default


class W8A8Fp8LinearWrapper:
    """
    Wrapper class for W8A8BlockFp8LinearOp that provides a callable interface
    and the is_quant_fp8_enabled() method required by tests.

    This class creates a W8A8 (weight-8bit, activation-8bit) FP8 linear operation
    with blockwise quantization support.
    """
    def __init__(
        self,
        group_shape: GroupShape,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
    ):
        """
        Initialize the W8A8 FP8 linear wrapper.

        Args:
            group_shape: The quantization group shape for activations.
                         For blockwise quantization, this is typically (1, block_size).
            weight: The FP8 quantized weight tensor.
            weight_scale: The per-group scaling factors for dequantizing the weights.
            input_scale: The per-group scaling factors for quantizing the input activations.
                         Can be None for dynamic quantization.
        """
        # Create the blockwise FP8 linear operator
        # Note: weight_group_shape uses a square group (group_shape[1], group_shape[1])
        # to match the expected weight layout for blockwise quantization
        self.linear_op = W8A8BlockFp8LinearOp(
            weight_group_shape=GroupShape(group_shape[1], group_shape[1]),
            act_quant_group_shape=group_shape,
            cutlass_block_fp8_supported=cutlass_block_fp8_supported(),
            use_aiter_and_is_supported=False,
        )
        self.weight = weight
        self.weight_scale = weight_scale
        self.input_scale = input_scale

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Make the wrapper callable like the original partial function."""
        return self.linear_op.apply(
            input=input,
            weight=self.weight,
            weight_scale=self.weight_scale,
            input_scale=self.input_scale,
            bias=None
        )

    def is_quant_fp8_enabled(self) -> bool:
        """Check if FP8 quantization custom op is enabled."""
        return self.linear_op.input_quant_op.enabled()

class TestModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        force_kernel: FP8ScaledMMLinearKernel,
        group_shape: GroupShape,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]
        static = group_shape == GroupShape.PER_TENSOR

        act_quant_scale_desc = ScaleDesc(torch.float32, static, group_shape)
        w_quant_scale_desc = ScaleDesc(torch.float32, True, group_shape)
        self.activation_quant_key = QuantKey(
            dtype=FP8_DTYPE, scale=act_quant_scale_desc, symmetric=True
        )
        self.weight_quant_key = QuantKey(
            dtype=FP8_DTYPE, scale=w_quant_scale_desc, symmetric=True
        )

        if group_shape.is_per_tensor():
            self.wscale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]
        elif group_shape.is_per_group():
            self.wscale = [
                torch.rand(
                    (hidden_size // group_shape[1], hidden_size // group_shape[1]),
                    dtype=torch.float32,
                )
                for _ in range(3)
            ]
        else:
            self.wscale = [
                torch.rand((hidden_size, 1), dtype=torch.float32) for _ in range(3)
            ]

        if static:
            self.act_scale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]

        else:
            self.act_scale = [None for _ in range(3)]

        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE) for _ in range(3)
        ]
        
        if not group_shape.is_per_group():
            self.w = [self.w[0].t() for _ in range(3)]

        self.enable_rms_norm_custom_op = self.norm[0].enabled()
 
        if group_shape.is_per_group():
            self.fp8_linear_layers = [
                W8A8Fp8LinearWrapper(
                    group_shape=group_shape,
                    weight=self.w[i],
                    weight_scale=self.wscale[i],
                    input_scale=self.act_scale[i],
                )
                for i in range(3)
            ]
            self.enable_quant_fp8_custom_op = self.fp8_linear_layers[
                0
            ].is_quant_fp8_enabled()
        else:
            self.fp8_linear_layers = [
                TestFP8Layer(
                    self.activation_quant_key,
                    self.weight_quant_key,
                    self.w[i],
                    self.wscale[i],
                    input_scale=self.act_scale[i],
                    force_kernel=force_kernel,
                )
                for i in range(3)
            ]
            self.enable_quant_fp8_custom_op = self.fp8_linear_layers[
                0
            ].is_quant_fp8_enabled()

        self.enable_rms_norm_custom_op = self.norm[0].enabled()
        self.group_shape = group_shape

        
    
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

    def ops_in_model_after(self):
        return [
            FUSED_OPS[FusedRMSQuantKey(self.activation_quant_key, True)],
            FUSED_OPS[FusedRMSQuantKey(self.activation_quant_key, False)],
        ]

    def ops_in_model_before(self):
        return (
            [QUANT_OPS[self.activation_quant_key]]
            if self.enable_quant_fp8_custom_op
            else [torch.ops.aten.reciprocal]
        )

    def ops_in_model_before_partial(self):
        return (
            [RMS_OP, RMS_ADD_OP]
            if self.enable_rms_norm_custom_op
            else [torch.ops.aten.rsqrt]
        )


ROCM_FP8_KERNELS = [
    ROCmScaledMMLinearKernel,
    PerTensorTorchScaledMMLinearKernel,
    RowWiseTorchScaledMMLinearKernel,
    ChannelWiseTorchScaledMMLinearKernel,
]

CUDA_FP8_KERNELS = [
    FlashInferScaledMMLinearKernel,
    CutlassFP8ScaledMMLinearKernel,
    PerTensorTorchScaledMMLinearKernel,
    ChannelWiseTorchScaledMMLinearKernel,
]

# Blockwise group shapes that use W8A8BlockFp8LinearOp
BLOCKWISE_GROUP_SHAPES = [
    GroupShape(1, 128),
    GroupShape(1, 64),
]

# Non-blockwise group shapes that use FP8ScaledMMLinearKernel
NON_BLOCKWISE_GROUP_SHAPES = [
    GroupShape.PER_TOKEN,
    GroupShape.PER_TENSOR,
]


def _generate_kernel_groupshape_combinations():
    """
    Generate valid (kernel, group_shape) combinations for testing.

    Returns:
        List of (kernel, group_shape) tuples where:
        - Blockwise group shapes use None as kernel (W8A8BlockFp8LinearOp doesn't use FP8ScaledMMLinearKernel)
        - Non-blockwise group shapes are paired with compatible kernels
    """
    combinations = []

    kernels = CUDA_FP8_KERNELS if current_platform.is_cuda() else ROCM_FP8_KERNELS

    # Non-blockwise group shapes with FP8ScaledMMLinearKernel
    for kernel in kernels:
        for group_shape in NON_BLOCKWISE_GROUP_SHAPES:
            # PerTensorTorchScaledMMLinearKernel only works with PER_TENSOR
            if kernel == PerTensorTorchScaledMMLinearKernel and group_shape != GroupShape.PER_TENSOR:
                continue
            # ChannelWiseTorchScaledMMLinearKernel only works with PER_TOKEN
            if kernel == ChannelWiseTorchScaledMMLinearKernel and group_shape != GroupShape.PER_TOKEN:
                continue
            # RowWiseTorchScaledMMLinearKernel only works with PER_TOKEN
            if kernel == RowWiseTorchScaledMMLinearKernel and group_shape != GroupShape.PER_TOKEN:
                continue
            combinations.append((kernel, group_shape))

    # Blockwise group shapes don't use FP8ScaledMMLinearKernel, so kernel is None
    for group_shape in BLOCKWISE_GROUP_SHAPES:
        combinations.append((None, group_shape))

    return combinations


# Generate valid combinations of (kernel, group_shape)
KERNEL_GROUPSHAPE_COMBINATIONS = _generate_kernel_groupshape_combinations()


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
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    # Unpack the (kernel, group_shape) combination
    force_kernel, group_shape = kernel_groupshape

    if not enable_quant_fp8_custom_op and group_shape.is_per_group():
        pytest.skip("Unsupported unwrapped quant fp8 op for blockwise quantization")

    # Skip test for 64-bit group shape when running with cutlass or deepgemm
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
        # Reshape pass is needed for the fusion pass to work
        noop_pass = NoOpEliminationPass(vllm_config)
        fusion_pass = RMSNormQuantFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)
        backend2 = TestBackend(noop_pass, cleanup_pass)
        model = TestModel(hidden_size, eps, force_kernel, group_shape)

        # # skip the test if we cannot force the kernel for non-blockwise group shapes
        # if force_kernel is not None:
        #     selected_kernels = [layer.kernel for layer in model.fp8_linear_layers]
        #     if not any(isinstance(kernel, force_kernel) for kernel in selected_kernels):
        #         pytest.skip(f"{force_kernel.__name__} couldn't be forced")

        # First dimension dynamic
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
        backend.check_before_ops(
            model.ops_in_model_before_partial(), fully_replaced=False
        )
        backend.check_after_ops(model.ops_in_model_after())

        # If RMSNorm custom op is disabled (native/torch impl used),
        # there's a risk that the fused add doesn't get included in the
        # replacement and only the rms part gets fused with quant.
        # Hence, we check only 2 add nodes are left (final fused rmsnorm add).
        if not enable_rms_norm_custom_op:
            n_add_nodes = lambda g: sum(1 for _ in find_op_nodes(torch.ops.aten.add, g))
            # 7 = 1 (RMS) + 3x2 (3xRMS_ADD, 2 each)
            assert n_add_nodes(backend.graph_pre_pass) == 7
            assert n_add_nodes(backend.graph_post_pass) == 2
