# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.activation_quant_fusion import ActivationQuantFusionPass
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fusion import RMSNormQuantFusionPass
from vllm.compilation.fx_utils import find_auto_fn, find_auto_fn_maybe, is_func
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform

from .backend import TestBackend

TEST_FP8 = current_platform.supports_fp8()
FP8_DTYPE = current_platform.fp8_dtype()


class TestSiluMul(torch.nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.wscale = torch.rand(1, dtype=torch.float32)
        self.scale = torch.rand(1, dtype=torch.float32)

        if TEST_FP8:
            self.w = torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=True,
                act_quant_group_shape=GroupShape.PER_TENSOR,
            )

    def forward(self, x):
        y = self.silu_and_mul(x)
        if TEST_FP8:
            x2 = self.fp8_linear.apply(y, self.w, self.wscale, input_scale=self.wscale)
            return x2
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
            self.fp8_linear = Fp8LinearOp(act_quant_static=True)

            self.scale = torch.rand(1, dtype=torch.float32)
            self.w = torch.rand(hidden_size, intermediate_size).to(dtype=FP8_DTYPE).t()
            self.wscale = torch.rand(1, dtype=torch.float32)

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
            fp8_linear_result = self.fp8_linear.apply(
                norm_output,
                self.w,
                self.wscale,
                input_scale=self.scale.to(norm_output.device),
            )

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


MODELS = [
    TestSiluMul,
    TestFusedAddRMSNorm,
    TestRotaryEmbedding,
    TestRotaryEmbeddingSliceScatter,
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("do_fusion", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda", reason="Only test on CUDA")
def test_fix_functionalization(
    model_class: torch.nn.Module, do_fusion: bool, dtype: torch.dtype
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

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
        torch.compile(model, backend=backend_func)(*model.example_inputs())
        torch.compile(model, backend=backend_no_func)(*model.example_inputs())

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
