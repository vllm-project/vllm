# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from tests.compile.backend import TestBackend
from tests.utils import TestFP8Layer
from vllm.compilation.passes.fusion.act_quant_fusion import (
    ActivationQuantFusionPass,
)
from vllm.compilation.passes.fusion.rms_quant_fusion import RMSNormQuantFusionPass
from vllm.compilation.passes.fx_utils import find_auto_fn, find_auto_fn_maybe, is_func
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

TEST_FP8 = current_platform.supports_fp8()
FP8_DTYPE = current_platform.fp8_dtype()


class TestSiluMul(torch.nn.Module):
    quant_key = kFp8StaticTensorSym

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        if TEST_FP8:
            self.fp8_linear = TestFP8Layer(
                weight_shape=(hidden_size, hidden_size),
                activation_quant_key=self.quant_key,
                weight_quant_key=self.quant_key,
            )

    def forward(self, x):
        y = self.silu_and_mul(x)
        if TEST_FP8:
            return self.fp8_linear(y)
        else:
            return y

    def example_inputs(self, num_tokens=32, hidden_size=128):
        return (torch.rand(num_tokens, hidden_size * 2),)

    def ops_in_model(self, do_fusion):
        if TEST_FP8 and do_fusion:
            return [torch.ops._C.silu_and_mul_quant.default]
        else:
            return [torch.ops._C.silu_and_mul.default]

    def ops_not_in_model(self):
        return []


class TestFusedAddRMSNorm(torch.nn.Module):
    quant_key = kFp8StaticTensorSym

    def __init__(self, hidden_size=16, intermediate_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = torch.nn.Parameter(
            torch.empty((intermediate_size, hidden_size))
        )
        self.norm = RMSNorm(intermediate_size, 1e-05)
        self.norm.weight = torch.nn.Parameter(torch.ones(intermediate_size))

        torch.nn.init.normal_(self.gate_proj, std=0.02)

        if TEST_FP8:
            self.fp8_linear = TestFP8Layer(
                weight_shape=(hidden_size, intermediate_size),
                activation_quant_key=self.quant_key,
                weight_quant_key=self.quant_key,
            )

    def forward(self, hidden_states, residual):
        # Reshape input
        view = hidden_states.reshape(-1, self.hidden_size)

        # matrix multiplication
        permute = self.gate_proj.permute(1, 0)
        mm = torch.mm(view, permute)

        # layer normalization
        norm_output, residual_output = self.norm(mm, residual)

        if TEST_FP8:
            # scaled_mm with static input quantization
            fp8_linear_result = self.fp8_linear(norm_output)

            return fp8_linear_result, residual_output

        else:
            return norm_output, residual_output

    def example_inputs(self, batch_size=8, hidden_size=16, seq_len=16):
        hidden_states = torch.randn((batch_size * seq_len, hidden_size))
        residual = torch.randn((batch_size * seq_len, hidden_size))
        return (hidden_states, residual)

    def ops_in_model(self, do_fusion):
        if TEST_FP8 and do_fusion:
            return [torch.ops._C.fused_add_rms_norm_static_fp8_quant.default]
        else:
            return [torch.ops._C.fused_add_rms_norm.default]

    def ops_not_in_model(self):
        return []


class TestRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim=64, max_position=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters={"rope_type": "default", "rope_theta": base},
        )

    def forward(self, positions, q, k):
        q_rotated, k_rotated = self.rotary_emb(positions, q, k)
        return q_rotated, k_rotated

    def example_inputs(self, num_tokens=32, head_dim=64):
        positions = torch.arange(num_tokens, dtype=torch.long)
        q = torch.randn(num_tokens, head_dim)
        k = torch.randn(num_tokens, head_dim)
        return (positions, q, k)

    def ops_in_model(self, do_fusion):
        return [torch.ops._C.rotary_embedding.default]

    def ops_not_in_model(self):
        return []


class TestRotaryEmbeddingSliceScatter(torch.nn.Module):
    def __init__(self, head_dim=64, num_heads=4, max_position=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.hidden_size = head_dim * num_heads

        self.qkv_proj = torch.nn.Linear(
            self.hidden_size, self.hidden_size * 3, bias=False
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters={"rope_type": "default", "rope_theta": base},
        )

    def forward(self, positions, hidden_states):
        # Simulate the pattern: mm -> split_with_sizes -> rotary_embedding
        # -> slice_scatter -> split_with_sizes

        qkv = self.qkv_proj(hidden_states)
        split_sizes = [self.hidden_size, self.hidden_size, self.hidden_size]
        q, k, v = torch.split(qkv, split_sizes, dim=-1)

        q_rotated, k_rotated = self.rotary_emb(positions, q, k)

        qkv_updated = torch.cat([q_rotated, k_rotated, v], dim=-1)
        return qkv_updated

    def example_inputs(self, num_tokens=32, head_dim=64, num_heads=4):
        hidden_size = head_dim * num_heads
        positions = torch.arange(num_tokens, dtype=torch.long)
        hidden_states = torch.randn(num_tokens, hidden_size)
        return (positions, hidden_states)

    def ops_in_model(self, do_fusion):
        return [torch.ops._C.rotary_embedding.default]

    def ops_not_in_model(self):
        return [torch.ops.aten.slice_scatter.default]


class TestFunctionWithMutatedArgsAndReturn(torch.nn.Module):
    OP_REGISTERED = False

    def __init__(self):
        super().__init__()
        self.register_test_custom_op()

    @classmethod
    def register_test_custom_op(cls):
        if not cls.OP_REGISTERED:

            def function_with_mutated_args_and_return_impl(
                x: torch.Tensor,
            ) -> torch.Tensor:
                ret = x + 1
                x.add_(2)
                return ret

            def function_with_mutated_args_and_return_fake(
                x: torch.Tensor,
            ) -> torch.Tensor:
                return torch.empty_like(x)

            direct_register_custom_op(
                op_name="function_with_mutated_args_and_return",
                op_func=function_with_mutated_args_and_return_impl,
                mutates_args=["x"],
                fake_impl=function_with_mutated_args_and_return_fake,
            )

            cls.OP_REGISTERED = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Clone x to avoid mutating the original tensor
        ret = torch.ops.vllm.function_with_mutated_args_and_return(x)
        return x, ret

    def example_inputs(self, num_tokens=32):
        hidden_states = torch.randn(num_tokens)
        return (hidden_states,)

    def ops_in_model(self, do_fusion):
        return [torch.ops.vllm.function_with_mutated_args_and_return.default]

    def ops_not_in_model(self):
        return []


MODELS_AND_DO_FUSION = {
    TestSiluMul: [True, False],
    TestFusedAddRMSNorm: [True, False],
    TestRotaryEmbedding: [False],
    TestRotaryEmbeddingSliceScatter: [False],
    TestFunctionWithMutatedArgsAndReturn: [False],
}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "model_class, do_fusion",
    [
        (model_class, do_fusion)
        for model_class, fusions in MODELS_AND_DO_FUSION.items()
        for do_fusion in fusions
    ],
)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
def test_fix_functionalization(
    model_class: torch.nn.Module, do_fusion: bool, dtype: torch.dtype
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            custom_ops=["all"],
            pass_config=PassConfig(
                fuse_norm_quant=do_fusion,
                fuse_act_quant=do_fusion,
                eliminate_noops=True,
            ),
        ),
    )

    with set_current_vllm_config(vllm_config):
        assert RMSNorm.enabled()
        noop_pass = NoOpEliminationPass(vllm_config)
        fusion_pass = RMSNormQuantFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        act_quant_fusion_pass = ActivationQuantFusionPass(vllm_config)

        passes = (
            [noop_pass, fusion_pass, act_quant_fusion_pass, cleanup_pass]
            if do_fusion
            else [noop_pass, cleanup_pass]
        )
        func_pass = FixFunctionalizationPass(vllm_config)

        backend_func = TestBackend(*passes, func_pass)
        backend_no_func = TestBackend(*passes)

        model = model_class()
        inputs_func = model.example_inputs()
        inputs_no_func = copy.deepcopy(inputs_func)
        model_func = model_class()
        model_no_func = copy.deepcopy(model_func)
        model_func = torch.compile(model_func, backend=backend_func)
        model_no_func = torch.compile(model_no_func, backend=backend_no_func)
        model_func(*inputs_func)
        model_no_func(*inputs_no_func)

        # check if the functionalization pass is applied
        for op in model.ops_in_model(do_fusion):
            find_auto_fn(backend_no_func.graph_post_pass.nodes, op)
            assert find_auto_fn_maybe(backend_func.graph_post_pass.nodes, op) is None

        # make sure the ops were all de-functionalized
        found = dict()
        for node in backend_func.graph_post_pass.nodes:
            for op in model.ops_in_model(do_fusion):
                if is_func(node, op):
                    found[op] = True
            for op in model.ops_not_in_model():
                if is_func(node, op):
                    found[op] = True
        assert all(found[op] for op in model.ops_in_model(do_fusion))
        assert all(not found.get(op) for op in model.ops_not_in_model())

        # TODO (Rohan138): compare the outputs from model_func and model_no_func
        # currently runs into errors while comparing `TestFusedAddRMSNorm`
        # Linked issue: https://github.com/vllm-project/vllm/issues/34996
        # torch.testing.assert_close(outputs_func, outputs_no_func)
