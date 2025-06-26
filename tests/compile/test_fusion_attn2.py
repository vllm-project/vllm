# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import LLM, SamplingParams
import vllm.envs as envs
import vllm.plugins
from vllm.compilation.fusion import (FUSED_OPS, QUANT_OPS, FusedRMSQuantKey,
                                     GroupShape, QuantKey)
from vllm.compilation.fusion_attn import AttnFusionPass
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import (CompilationConfig, CompilationLevel, PassConfig,
                         VllmConfig, ModelConfig)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_FP8_SUPPORTED, Fp8LinearOp, maybe_create_device_identity)
from vllm.attention import Attention, AttentionType
from vllm.platforms import current_platform
from vllm.forward_context import set_forward_context
from .backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()


class TestModel(torch.nn.Module):

    def __init__(self, cutlass_fp8_enabled: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = Attention(
            32,
            128,
            0.08,
            num_kv_heads=8,
        )
        self.fp8_linear = Fp8LinearOp(
            cutlass_fp8_supported=cutlass_fp8_enabled,
            use_per_token_if_dynamic=True)
        self.w = torch.rand(4096, 4096).to(dtype=FP8_DTYPE).t()
        self.wscale = torch.rand(1, dtype=torch.float32)
        self.scale = torch.rand(1, dtype=torch.float32)
        self.key = QuantKey(dtype=FP8_DTYPE,
                            static=True,
                            group_shape=GroupShape.PER_TENSOR,
                            symmetric=True)

    def forward(self, q, kv):
        y = self.attn(q, kv, kv)
        output = self.fp8_linear.apply(
            y,
            weight=self.w,
            weight_scale=self.wscale,
            out_dtype=torch.bfloat16,
            input_scale=self.scale,
        )
        return output

    def ops_in_model_before(self):
        return [QUANT_OPS[self.key]]

    def ops_in_model_after(self):
        return []


def test_fusion_attn2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V1", str(int(1)))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(1)
    maybe_create_device_identity()  # needed for certain non-cutlass fp8 paths

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        full_cuda_graph=True,
    ),
                             model_config=ModelConfig(
                                 model="amd/Llama-3.1-8B-Instruct-FP8-KV",
                                 dtype=torch.bfloat16,
                             ))
    vllm_config.compilation_config.pass_config = \
        PassConfig(enable_attn_fusion=True, enable_noop=True,
                    dump_graph_stages=["before_attn_fusion", "after_attn_fusion"])
    llm = LLM(model="amd/Llama-3.1-8B-Instruct-FP8-KV", enforce_eager=True)
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    orig_model = model_runner.model
    passes = [NoOpEliminationPass(vllm_config), AttnFusionPass(vllm_config)]

    backend_func = TestBackend(*passes)
    model_runner.model = torch.compile(orig_model,
                                       fullgraph=True,
                                       backend=backend_func)
    llm.generate("Hello world", sampling_params=SamplingParams(max_tokens=1))

    # # In pre-nodes, fp8 quant should be there and fused kernels should not
    backend_func.check_before_ops(
        [torch.ops._C.static_scaled_fp8_quant.default])

    # # In post-nodes, fused kernels should be there and fp8 quant should not
    backend_func.check_after_ops([])
