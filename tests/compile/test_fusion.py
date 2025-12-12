# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

from ..utils import TestBlockFP8Layer, TestFP8Layer
from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()

RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default


class TestModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        force_kernel: FP8ScaledMMLinearKernel | None,
        group_shape: GroupShape,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group_shape = group_shape
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]
        self.enable_rms_norm_custom_op = self.norm[0].enabled()

        is_static = group_shape == GroupShape.PER_TENSOR
        act_quant_scale_desc = ScaleDesc(torch.float32, is_static, group_shape)
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
        else:  # PER_TOKEN
            self.wscale = [
                torch.rand((hidden_size, 1), dtype=torch.float32) for _ in range(3)
            ]

        self.act_scale = (
            [torch.rand(1, dtype=torch.float32) for _ in range(3)]
            if is_static
            else [None for _ in range(3)]
        )

        # Initialize weights
        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE) for _ in range(3)
        ]
        if not group_shape.is_per_group():
            self.w = [self.w[0].t() for _ in range(3)]

        if group_shape.is_per_group():
            self.fp8_linear_layers = [
                TestBlockFP8Layer(
                    group_shape=group_shape,
                    weight=self.w[i],
                    weight_scale=self.wscale[i],
                    input_scale=self.act_scale[i],
                )
                for i in range(3)
            ]
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


BLOCKWISE_GROUP_SHAPES = [
    GroupShape(1, 128),
    GroupShape(1, 64),
]

NON_BLOCKWISE_GROUP_SHAPES = [
    GroupShape.PER_TOKEN,
    GroupShape.PER_TENSOR,
]


def _generate_kernel_groupshape_combinations():
    """
    Generate valid (kernel, group_shape) combinations for testing.
    """
    combinations = []

    kernels = CUDA_FP8_KERNELS if current_platform.is_cuda() else ROCM_FP8_KERNELS

    for kernel in kernels:
        for group_shape in NON_BLOCKWISE_GROUP_SHAPES:
            if (
                kernel == PerTensorTorchScaledMMLinearKernel
                and group_shape != GroupShape.PER_TENSOR
            ):
                continue
            if (
                kernel == ChannelWiseTorchScaledMMLinearKernel
                and group_shape != GroupShape.PER_TOKEN
            ):
                continue
            if (
                kernel == RowWiseTorchScaledMMLinearKernel
                and group_shape != GroupShape.PER_TOKEN
            ):
                continue
            combinations.append((kernel, group_shape))

    # Blockwise group shapes don't use FP8ScaledMMLinearKernel, so kernel is None
    for group_shape in BLOCKWISE_GROUP_SHAPES:
        combinations.append((None, group_shape))

    return combinations


KERNEL_GROUPSHAPE_COMBINATIONS = _generate_kernel_groupshape_combinations()


class TestRmsnormGroupFp8QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float, **kwargs):
        super().__init__()

        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()
            for _ in range(3)
        ]

        scale_hidden_size = (hidden_size + 128 - 1) // 128
        self.wscale = [
            torch.rand((scale_hidden_size, scale_hidden_size), dtype=torch.float32)
            for _ in range(3)
        ]

        self.norm_weight = [torch.ones(hidden_size) for _ in range(4)]
        self.eps = eps

        self.w8a8_block_fp8_linear = [
            TestBlockFP8Layer(
                GroupShape(128, 128),
                self.w[i],
                self.wscale[i],
                cutlass_block_fp8_supported=False,
                use_aiter_and_is_supported=True,
            )
            for i in range(3)
        ]

    def forward(self, x):
        # avoid having graph input be an arg to a pattern directly
        x = resid = torch.relu(x)
        y = rocm_aiter_ops.rms_norm(x, self.norm_weight[0], self.eps)

        x2 = self.w8a8_block_fp8_linear[0](y)
        # make sure resid is used for replacement to work
        y2, resid = rocm_aiter_ops.rms_norm2d_with_add(
            x2, resid, self.norm_weight[1], self.eps
        )

        x3 = self.w8a8_block_fp8_linear[1](y2)

        y3, resid = rocm_aiter_ops.rms_norm2d_with_add(
            x3, resid, self.norm_weight[2], self.eps
        )

        x4 = self.w8a8_block_fp8_linear[2](y3)

        y4, resid = rocm_aiter_ops.rms_norm2d_with_add(
            x4, resid, self.norm_weight[3], self.eps
        )
        return y4

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.rocm_aiter_rms_norm,
            torch.ops.vllm.rocm_aiter_group_fp8_quant,
        ]

    def ops_in_model_before_partial(self):
        return []

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant,
            torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant,
        ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("kernel_groupshape", KERNEL_GROUPSHAPE_COMBINATIONS)
@pytest.mark.parametrize(
    "model_class, enable_rms_norm_custom_op, enable_quant_fp8_custom_op",
    list(itertools.product([TestModel], [True, False], [True, False]))
    + [(TestRmsnormGroupFp8QuantModel, False, False)],
)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only test on CUDA and ROCm"
)
def test_fusion_rmsnorm_quant(
    dtype,
    hidden_size,
    num_tokens,
    eps,
    kernel_groupshape,
    model_class,
    enable_rms_norm_custom_op,
    enable_quant_fp8_custom_op,
):
    if model_class is TestRmsnormGroupFp8QuantModel and not IS_AITER_FOUND:
        pytest.skip("AITER is not supported on this GPU.")

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
        if model_class is TestRmsnormGroupFp8QuantModel:
            from vllm.compilation.rocm_aiter_fusion import (
                RocmAiterRMSNormFp8GroupQuantFusionPass,
            )

            fusion_pass = RocmAiterRMSNormFp8GroupQuantFusionPass(vllm_config)
        else:
            fusion_pass = RMSNormQuantFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)
        backend2 = TestBackend(noop_pass, cleanup_pass)
        model = TestModel(hidden_size, eps, force_kernel, group_shape)

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
        if (
            not enable_rms_norm_custom_op
            and model_class is not TestRmsnormGroupFp8QuantModel
        ):
            n_add_nodes = lambda g: sum(1 for _ in find_op_nodes(torch.ops.aten.add, g))
            # 7 = 1 (RMS) + 3x2 (3xRMS_ADD, 2 each)
            assert n_add_nodes(backend.graph_pre_pass) == 7
            assert n_add_nodes(backend.graph_post_pass) == 2
