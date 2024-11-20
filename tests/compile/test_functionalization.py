import pytest
import torch

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.fusion import (FusionPass, find_auto_fn,
                                     find_auto_fn_maybe)
from vllm.compilation.reshapes import RedundantReshapesPass
from vllm.compilation.vllm_inductor_pass import is_func
from vllm.config import CompilationConfig

from .backend import TestBackend

OPS_IN_MODEL = [
    torch.ops._C.rotary_embedding.default,
    torch.ops._C.fused_add_rms_norm.default,
    torch.ops._C.silu_and_mul.default,
]

RMS_OP = torch.ops._C.rms_norm.default

RMS_QUANT_OPS = {
    "static_fp8": [
        torch.ops._C.rms_norm_static_fp8_quant.default,
        torch.ops._C.fused_add_rms_norm_static_fp8_quant.default
    ],
}

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@pytest.mark.parametrize("model",
                         ["nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e"])
@pytest.mark.parametrize("do_fusion", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE != "cuda",
                    reason="Only test on CUDA")
def test_fix_functionalization(model: str, do_fusion: bool):
    torch.set_default_device("cuda")

    config = CompilationConfig.PassConfig(enable_fusion=do_fusion,
                                          enable_reshape=True)
    reshape_pass = RedundantReshapesPass(config)
    fusion_pass = FusionPass.instance(config)

    passes = [reshape_pass, fusion_pass] if do_fusion else [reshape_pass]
    func_pass = FixFunctionalizationPass(config)
    backend_func = TestBackend(*passes, func_pass)
    backend_no_func = TestBackend(*passes)

    # instantiate a full engine and manually compile the model 2x
    # (with and without FixFunctionalizationPass)
    llm = LLM(model=model, enforce_eager=True)
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    orig_model = model_runner.model
    # TODO mark inputs dynamic? (currently torch.compile is triggered 4x)
    # Can only do that by using the decorator but then we'd have to instantiate
    # 2 LLM instances.

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)
    model_runner.model = torch.compile(orig_model,
                                       fullgraph=True,
                                       backend=backend_func)
    gen_func = llm.generate(prompts, sampling_params)

    model_runner.model = torch.compile(orig_model,
                                       fullgraph=True,
                                       backend=backend_no_func)
    gen_no_func = llm.generate(prompts, sampling_params)

    for output_func, output_no_func in zip(gen_func, gen_no_func):
        assert output_func.outputs[0].text == output_no_func.outputs[0].text

    # OPS_IN_MODEL always appear. RMS_OP is fused away if we run fusion,
    # and replaced by fused quantized ops in RMS_QUANT_OPS.
    ops = OPS_IN_MODEL + (RMS_QUANT_OPS["static_fp8"]
                          if do_fusion else [RMS_OP])

    for op in ops:
        find_auto_fn(backend_no_func.graph_post_pass.nodes, op)
        assert find_auto_fn_maybe(backend_func.graph_post_pass.nodes,
                                  op) is None  # noqa: E501

    # make sure the ops were all de-functionalized
    found = dict()
    for node in backend_func.graph_post_pass.nodes:
        for op in ops:
            if is_func(node, op):
                found[op] = True
    assert all(found[op] for op in ops)
