# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import pytest
import torch

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
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    maybe_create_device_identity,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

from ..utils import override_cutlass_fp8_supported
from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()

RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default


class TestModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_shape: GroupShape,
        cuda_force_torch: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cuda_force_torch = cuda_force_torch
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]
        if group_shape.is_per_group():
            self.wscale = [
                torch.rand(
                    (hidden_size // group_shape[1], hidden_size // group_shape[1]),
                    dtype=torch.float32,
                )
                for _ in range(3)
            ]
        else:
            self.wscale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]
        static = group_shape == GroupShape.PER_TENSOR
        quant_scale = ScaleDesc(torch.float32, static, group_shape)
        self.quant_key = QuantKey(dtype=FP8_DTYPE, scale=quant_scale, symmetric=True)
        if static:
            self.scale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]
        else:
            self.scale = [None for _ in range(3)]
        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE) for _ in range(3)
        ]
        if not group_shape.is_per_group():
            self.w = [self.w[0].t() for _ in range(3)]

        if group_shape.is_per_group():
            self.fp8_linear = W8A8BlockFp8LinearOp(
                weight_group_shape=GroupShape(group_shape[1], group_shape[1]),
                act_quant_group_shape=group_shape,
                cutlass_block_fp8_supported=cutlass_block_fp8_supported(),
                use_aiter_and_is_supported=False,
            )
            self.enable_quant_fp8_custom_op = self.fp8_linear.input_quant_op.enabled()
        else:
            with override_cutlass_fp8_supported(not cuda_force_torch):
                self.fp8_linear = Fp8LinearOp(
                    act_quant_static=static,
                    act_quant_group_shape=group_shape,
                )
                self.enable_quant_fp8_custom_op = self.fp8_linear.quant_fp8.enabled()

        self.enable_rms_norm_custom_op = self.norm[0].enabled()
        self.group_shape = group_shape

    def forward(self, x):
        # avoid having graph input be an arg to a pattern directly
        x = resid = torch.relu(x)
        y = self.norm[0](x)

        x2 = self.fp8_linear.apply(
            y, self.w[0], self.wscale[0], input_scale=self.scale[0]
        )
        # make sure resid is used for replacement to work
        y2, resid = self.norm[1](x2, resid)

        x3 = self.fp8_linear.apply(
            y2, self.w[1], self.wscale[1], input_scale=self.scale[1]
        )

        y3, resid = self.norm[2](x3, resid)  # use resid here

        x4 = self.fp8_linear.apply(
            y3, self.w[2], self.wscale[2], input_scale=self.scale[2]
        )

        y4, resid = self.norm[3](x4, resid)  # use resid here
        return y4

    def ops_in_model_after(self):
        return [
            FUSED_OPS[FusedRMSQuantKey(self.quant_key, True)],
            FUSED_OPS[FusedRMSQuantKey(self.quant_key, False)],
        ]

    def ops_in_model_before(self):
        return (
            [QUANT_OPS[self.quant_key]]
            if self.enable_quant_fp8_custom_op
            else [torch.ops.aten.reciprocal]
        )

    def ops_in_model_before_partial(self):
        return (
            [RMS_OP, RMS_ADD_OP]
            if self.enable_rms_norm_custom_op
            else [torch.ops.aten.rsqrt]
        )


GROUP_SHAPES = [
    GroupShape.PER_TOKEN,
    GroupShape.PER_TENSOR,
    GroupShape(1, 128),
    GroupShape(1, 64),
]


class TestRmsnormGroupFp8QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float, **kwargs):
        super().__init__()
        self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
            weight_group_shape=GroupShape(128, 128),
            act_quant_group_shape=GroupShape(1, 128),
            cutlass_block_fp8_supported=False,
            use_aiter_and_is_supported=True,
        )
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

    def forward(self, x):
        # avoid having graph input be an arg to a pattern directly
        x = resid = torch.relu(x)
        y = rocm_aiter_ops.rms_norm(x, self.norm_weight[0], self.eps)

        x2 = self.w8a8_block_fp8_linear.apply(y, self.w[0], self.wscale[0])
        # make sure resid is used for replacement to work
        y2, resid = rocm_aiter_ops.rms_norm2d_with_add(
            x2, resid, self.norm_weight[1], self.eps
        )

        x3 = self.w8a8_block_fp8_linear.apply(y2, self.w[1], self.wscale[1])

        y3, resid = rocm_aiter_ops.rms_norm2d_with_add(
            x3, resid, self.norm_weight[2], self.eps
        )

        x4 = self.w8a8_block_fp8_linear.apply(y3, self.w[2], self.wscale[2])

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
@pytest.mark.parametrize("group_shape", GROUP_SHAPES)
@pytest.mark.parametrize(
    "model_class, enable_rms_norm_custom_op, enable_quant_fp8_custom_op",
    list(itertools.product([TestModel], [True, False], [True, False]))
    + [(TestRmsnormGroupFp8QuantModel, False, False)],
)
# cuda_force_torch used to test torch code path on platforms that
# cutlass_fp8_supported() == True.
@pytest.mark.parametrize(
    "cuda_force_torch", [True, False] if cutlass_fp8_supported() else [True]
)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only test on CUDA and ROCm"
)
def test_fusion_rmsnorm_quant(
    dtype,
    hidden_size,
    num_tokens,
    eps,
    group_shape,
    model_class,
    enable_rms_norm_custom_op,
    enable_quant_fp8_custom_op,
    cuda_force_torch,
):
    if model_class is TestRmsnormGroupFp8QuantModel and not IS_AITER_FOUND:
        pytest.skip("AITER is not supported on this GPU.")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)
    maybe_create_device_identity()  # needed for certain non-cutlass fp8 paths

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
        model = model_class(
            hidden_size=hidden_size,
            eps=eps,
            group_shape=group_shape,
            cuda_force_torch=cuda_force_torch,
        )
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
