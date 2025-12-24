# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


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
        use_aiter: bool = False,
        cuda_force_torch: bool = False,
        use_aiter_quant_op: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_aiter = use_aiter
        self.use_aiter_quant_op = use_aiter_quant_op
        self.cuda_force_torch = cuda_force_torch
        self.group_shape = group_shape
        self.enable_quant_fp8_custom_op = None  # Will be set later if applicable

        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]

        # Setup quantization scale descriptor
        static = group_shape == GroupShape.PER_TENSOR and not use_aiter
        quant_scale = ScaleDesc(torch.float32, static, group_shape)
        self.quant_key = QuantKey(dtype=FP8_DTYPE, scale=quant_scale, symmetric=True)

        # Setup scales
        if static:
            self.scale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]
        else:
            self.scale = [None for _ in range(3)]

        # Setup weights
        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE) for _ in range(3)
        ]
        if not group_shape.is_per_group() or use_aiter:
            self.w = [self.w[0].t() for _ in range(3)]

        # Setup weight scales
        if group_shape.is_per_group():
            scale_size = (
                (hidden_size + 128 - 1) // 128
                if use_aiter
                else hidden_size // group_shape[1]
            )
            wscale_shape: tuple[int, ...] = (scale_size, scale_size)
        else:
            wscale_shape = (1,)
        self.wscale = [torch.rand(wscale_shape, dtype=torch.float32) for _ in range(3)]

        # Setup FP8 linear operation
        is_per_group = group_shape.is_per_group()
        if is_per_group and use_aiter:
            self.fp8_linear = W8A8BlockFp8LinearOp(
                weight_group_shape=GroupShape(128, 128),
                act_quant_group_shape=group_shape,
                use_aiter_and_is_supported=use_aiter_quant_op,
            )
            # AITER blockwise doesn't use enable_quant_fp8_custom_op
        elif is_per_group:
            self.fp8_linear = W8A8BlockFp8LinearOp(
                weight_group_shape=GroupShape(group_shape[1], group_shape[1]),
                act_quant_group_shape=group_shape,
                cutlass_block_fp8_supported=cutlass_block_fp8_supported(),
                use_aiter_and_is_supported=False,
            )
            self.enable_quant_fp8_custom_op = self.fp8_linear.input_quant_op.enabled()
        elif use_aiter:
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=False,
                act_quant_group_shape=group_shape,
            )
            self.fp8_linear.quant_fp8.use_aiter = use_aiter_quant_op
            self.enable_quant_fp8_custom_op = self.fp8_linear.quant_fp8.enabled()
        else:
            with override_cutlass_fp8_supported(not cuda_force_torch):
                self.fp8_linear = Fp8LinearOp(
                    act_quant_static=static,
                    act_quant_group_shape=group_shape,
                )
                self.enable_quant_fp8_custom_op = self.fp8_linear.quant_fp8.enabled()

        self.enable_rms_norm_custom_op = self.norm[0].enabled()

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

    def ops_in_model_before(self):
        if (
            self.use_aiter
            and self.group_shape.is_per_group()
            and current_platform.is_fp8_fnuz()
        ):
            return [rocm_aiter_ops.get_group_quant_op()]
        if self.use_aiter and self.group_shape.is_per_group():
            return [torch.ops.vllm.triton_per_token_group_quant_fp8.default]
        if self.use_aiter and self.use_aiter_quant_op:
            return [rocm_aiter_ops.get_per_token_quant_op()]
        if self.use_aiter:
            return [QUANT_OPS[self.quant_key]]
        if self.enable_quant_fp8_custom_op:
            return [QUANT_OPS[self.quant_key]]
        return [torch.ops.aten.reciprocal]

    def ops_in_model_after(self):
        if self.use_aiter and self.group_shape.is_per_group():
            from vllm.compilation.rocm_aiter_fusion import (
                AiterFusedAddRMSFp8GroupQuantPattern,
                AiterRMSFp8GroupQuantPattern,
            )

            return [
                AiterFusedAddRMSFp8GroupQuantPattern.FUSED_OP,
                AiterRMSFp8GroupQuantPattern.FUSED_OP,
            ]
        if self.use_aiter:
            from vllm.compilation.rocm_aiter_fusion import (
                AiterFusedAddRMSNormDynamicQuantPattern,
                AiterRMSNormDynamicQuantPattern,
            )

            return [
                AiterFusedAddRMSNormDynamicQuantPattern.FUSED_OP,
                AiterRMSNormDynamicQuantPattern.FUSED_OP,
            ]
        return [
            FUSED_OPS[FusedRMSQuantKey(self.quant_key, True)],
            FUSED_OPS[FusedRMSQuantKey(self.quant_key, False)],
        ]

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
@pytest.mark.parametrize("group_shape", GROUP_SHAPES)
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.parametrize("enable_quant_fp8_custom_op", [True, False])
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
    enable_rms_norm_custom_op,
    enable_quant_fp8_custom_op,
    cuda_force_torch,
):
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
        maybe_create_device_identity()

        fusion_pass = RMSNormQuantFusionPass(vllm_config)
        model = TestModel(
            hidden_size=hidden_size,
            eps=eps,
            group_shape=group_shape,
            use_aiter=False,
            cuda_force_torch=cuda_force_torch,
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


GROUP_SHAPE_QUANT_OPS_MATCHS = [
    (GroupShape.PER_TOKEN, True),
    (GroupShape.PER_TOKEN, False),
    (GroupShape(1, 128), True),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize(
    "group_shape, use_aiter_quant_op", GROUP_SHAPE_QUANT_OPS_MATCHS
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
    group_shape: GroupShape,
    use_aiter_quant_op: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(fuse_norm_quant=True, eliminate_noops=True),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from vllm.compilation.rocm_aiter_fusion import RocmAiterRMSNormFusionPass

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(1)
        maybe_create_device_identity()

        fusion_pass = RocmAiterRMSNormFusionPass(vllm_config)
        model = TestModel(
            hidden_size=hidden_size,
            eps=eps,
            group_shape=group_shape,
            use_aiter=True,
            use_aiter_quant_op=use_aiter_quant_op,
        )

        _run_fusion_test(
            model, fusion_pass, vllm_config, dtype, hidden_size, num_tokens
        )
