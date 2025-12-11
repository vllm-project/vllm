# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import pytest
import torch
from torch._ops import OpOverload

import vllm.plugins
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fusion import (
    QUANT_OPS,
    FusedRMSQuantKey,
)
from vllm.compilation.fx_utils import find_auto_fn, find_auto_fn_maybe, is_func
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.rocm_aiter_fusion import (
    AiterFusedAddRMSFp8GroupQuantPattern,
    AiterFusedAddRMSNormDynamicQuantPattern,
    AiterRMSFp8GroupQuantPattern,
    AiterRMSNormDynamicQuantPattern,
    AiterRMSNormQuantPattern,
    RocmAiterRMSNormFusionPass,
)
from vllm.config import CompilationConfig, CompilationMode, PassConfig, VllmConfig
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
    maybe_create_device_identity,
)
from vllm.platforms import current_platform

from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()


class TestModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        is_block_linear: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(3)]
        group_shape = GroupShape.PER_TOKEN
        # AITER RMSNorm fusion pass does not support static quantization at the moment.
        quant_scale = ScaleDesc(torch.float32, static=False, group_shape=group_shape)
        self.key = QuantKey(dtype=FP8_DTYPE, scale=quant_scale, symmetric=True)

        self.w = [
            torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()
            for _ in range(2)
        ]
        self.scale = [None for _ in range(2)]

        self.is_block_linear = is_block_linear

        if is_block_linear:
            scale_hidden_size = (hidden_size + 128 - 1) // 128
            self.wscale = [
                torch.rand(
                    size=(scale_hidden_size, scale_hidden_size), dtype=torch.float32
                )
                for _ in range(2)
            ]
            self.linear = W8A8BlockFp8LinearOp(
                weight_group_shape=GroupShape(128, 128),
                act_quant_group_shape=GroupShape(1, 128),
                cutlass_block_fp8_supported=False,
                use_aiter_and_is_supported=True,
            )
        else:
            self.wscale = [
                torch.rand(size=(hidden_size, 1), dtype=torch.float32) for _ in range(2)
            ]
            self.linear = Fp8LinearOp(
                act_quant_static=False,
                act_quant_group_shape=group_shape,
            )

    def forward(self, x):
        resid = torch.sqrt(x)
        y = self.norm[0](x)

        x2 = self.linear.apply(y, self.w[0], self.wscale[0], input_scale=self.scale[0])
        # make sure resid is used for replacement to work
        y2, resid = self.norm[1](x2, resid)

        x3 = self.linear.apply(y2, self.w[1], self.wscale[1], input_scale=self.scale[1])
        y3, resid = self.norm[2](x3, resid)  # use resid here
        return y3

    def ops_in_model_before(self) -> Sequence[OpOverload]:
        if self.is_block_linear:
            return [RocmAiterRMSNormFusionPass.AITER_GROUP_FP8_QUANT_OP]
        return [QUANT_OPS[self.key]]

    def ops_in_model_after(self) -> Sequence[OpOverload]:
        if self.is_block_linear:
            return [
                AiterFusedAddRMSFp8GroupQuantPattern.RMS_ADD_GROUP_QUANT_OP,
                AiterRMSFp8GroupQuantPattern.RMS_GROUP_QUANT_OP,
            ]

        ROCM_AITER_FUSED_OPS = (
            AiterFusedAddRMSNormDynamicQuantPattern.FUSED_OPS
            | AiterRMSNormDynamicQuantPattern.FUSED_OPS
        )

        return [
            ROCM_AITER_FUSED_OPS[FusedRMSQuantKey(self.key, False)],
            ROCM_AITER_FUSED_OPS[FusedRMSQuantKey(self.key, True)],
        ]

    def ops_in_model(self):
        return [AiterRMSNormQuantPattern.RMS_ADD_OP]

    def ops_not_in_model(self):
        return []


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("enable_block_linear", [True, False])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="Only test on ROCm")
def test_fusion_rmsnorm_quant(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    eps: float,
    enable_block_linear: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)
    maybe_create_device_identity()  # needed for certain non-cutlass fp8 paths

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            compilation_config=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(enable_fusion=True, enable_noop=True),
        )
    )
    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_USE_AITER_RMSNORM", "1")

        # Reshape pass is needed for the fusion pass to work
        noop_pass = NoOpEliminationPass(vllm_config)
        fusion_pass = RocmAiterRMSNormFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)

        model = TestModel(hidden_size, eps, is_block_linear=enable_block_linear)

        # First dimension dynamic
        x = torch.rand(num_tokens, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)

        result2 = model2(x)

        ATOL, RTOL = (1e-2, 1e-2)

        torch.testing.assert_close(result, result2, atol=ATOL, rtol=RTOL)

        assert fusion_pass.matched_count == 2

        # In pre-nodes, fp8 quant should be there and fused kernels should not
        backend.check_before_ops(model.ops_in_model_before())

        # In post-nodes, fused kernels should be there and fp8 quant should not
        backend.check_after_ops(model.ops_in_model_after())


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="Only test on ROCm")
def test_fix_functionalization(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    eps: float,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(enable_fusion=True, enable_noop=True),
        )
    )
    with monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_USE_AITER_RMSNORM", "1")

        # Reshape pass is needed for the fusion pass to work
        noop_pass = NoOpEliminationPass(vllm_config)
        fusion_pass = RocmAiterRMSNormFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        passes = [noop_pass, fusion_pass, cleanup_pass]
        func_pass = FixFunctionalizationPass(vllm_config)

        backend_no_func = TestBackend(*passes)
        backend_func = TestBackend(*passes, func_pass)

        model = TestModel(hidden_size, eps)

        # First dimension dynamic
        x = torch.rand(num_tokens, hidden_size)

        torch.compile(model, backend=backend_no_func)(x)
        torch.compile(model, backend=backend_func)(x)

        # check if the functionalization pass is applied
        for op in model.ops_in_model():
            find_auto_fn(backend_no_func.graph_post_pass.nodes, op)
            assert find_auto_fn_maybe(backend_func.graph_post_pass.nodes, op) is None

        # make sure the ops were all de-functionalized
        found = dict()
        for node in backend_func.graph_post_pass.nodes:
            for op in model.ops_in_model():
                if is_func(node, op):
                    found[op] = True
            for op in model.ops_not_in_model():
                if is_func(node, op):
                    found[op] = True
        assert all(found[op] for op in model.ops_in_model())
        assert all(not found.get(op) for op in model.ops_not_in_model())
