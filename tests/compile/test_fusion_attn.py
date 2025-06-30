# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import Optional

import pytest
import torch._dynamo

from tests.compile.backend import LazyInitPass, TestBackend, TestPassManager
from tests.models.utils import check_outputs_equal
from vllm import LLM, SamplingParams
from vllm.compilation.fusion import QUANT_OPS, QuantKey, kFp8StaticTensorSym
from vllm.compilation.fusion_attn import ATTN_OP, AttnFusionPass
from vllm.compilation.fx_utils import find_op_nodes
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.vllm_inductor_pass import PrinterInductorPass
from vllm.config import (CompilationConfig, CompilationLevel, ModelConfig,
                         PassConfig, VllmConfig, get_current_vllm_config,
                         set_current_vllm_config)
from vllm.platforms import current_platform


def check(test_pass_manager: TestPassManager, quant_key: QuantKey,
          compile_config: CompilationConfig):

    # check support
    attn_fusion_supported = [
        layer.impl.fused_output_quant_supported(quant_key.dtype,
                                                quant_key.static,
                                                quant_key.group_shape)
        for key, layer in compile_config.static_forward_context.items()
    ]
    if any(attn_fusion_supported):
        # Check quant ops
        test_pass_manager.check_before_ops([QUANT_OPS[quant_key]],
                                           fully_replaced=False)

    # attention ops present in both, just output_scale param changes
    attn_nodes_pre = list(
        find_op_nodes(ATTN_OP, test_pass_manager.graph_pre_pass))
    attn_nodes_post = list(
        find_op_nodes(ATTN_OP, test_pass_manager.graph_post_pass))
    assert len(attn_nodes_pre) == len(attn_nodes_post)

    for i in range(len(attn_nodes_pre)):
        assert attn_nodes_pre[i].kwargs["output_scale"] is None
        fused = attn_nodes_post[i].kwargs["output_scale"] is not None
        assert fused == attn_fusion_supported[i], \
            f"Node {i} {'' if fused else 'not '} expected " \
            f"to have fused output quant"


backend: Optional[TestBackend] = None
backend_unfused: Optional[TestBackend] = None


@pytest.mark.parametrize(
    "model, quant_key",
    [("amd/Llama-3.1-8B-Instruct-FP8-KV", kFp8StaticTensorSym)])
@pytest.mark.parametrize(
    "use_triton_fa", [True, False] if current_platform.is_rocm() else [False])
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="Only test CUDA and ROCm")
def test_attention_fusion_v0(example_prompts, monkeypatch, model: str,
                             quant_key: QuantKey, use_triton_fa: bool):
    # Clean Dynamo cache to avoid reusing other test cases
    # (for some reason the reset at the end is not enough)
    torch._dynamo.reset()

    # Use global backends
    global backend, backend_unfused

    monkeypatch.setenv("VLLM_USE_V1", "0")
    monkeypatch.setenv("VLLM_USE_TRITON_FLASH_ATTN", str(int(use_triton_fa)))

    # Prompt 4 seems too open-ended, differs between fused and unfused
    # (both outputs look reasonable though)
    prompts = example_prompts[:4] + example_prompts[5:]

    compile_config = CompilationConfig(
        # DYNAMO_AS_IS triggers custom backend & does full Dynamo compilation
        # DYNAMO_ONCE does not properly propagate shapes.
        level=CompilationLevel.DYNAMO_AS_IS,
        backend="tests.compile.test_fusion_attn.backend_unfused",
        custom_ops=["+quant_fp8"],
    )
    model_config = ModelConfig(model=model, dtype="bfloat16")
    vllm_config = VllmConfig(compilation_config=compile_config,
                             model_config=model_config)
    backend_unfused = TestBackend(NoOpEliminationPass(vllm_config))

    llm = LLM(model,
              enforce_eager=True,
              compilation_config=compile_config,
              gpu_memory_utilization=0.4,
              max_model_len=2048)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=10,
                                     top_p=0.95)

    unfused_output = llm.generate(prompts, sampling_params)
    backend_unfused = None  # Reset backend to make sure llm gets released
    del llm

    compile_config = CompilationConfig(
        # DYNAMO_AS_IS triggers custom backend & does full Dynamo compilation
        # DYNAMO_ONCE does not properly propagate shapes.
        level=CompilationLevel.DYNAMO_AS_IS,
        backend="tests.compile.test_fusion_attn.backend",
        custom_ops=["+quant_fp8"],
    )
    vllm_config = VllmConfig(compilation_config=compile_config,
                             model_config=model_config)

    # AttnFusionPass needs attention layers to be registered in config upon init
    # so we initialize it during compilation.
    attn_pass = lambda *args, **kw: AttnFusionPass(vllm_config)(*args, **kw)
    backend = TestBackend(NoOpEliminationPass(vllm_config),
                          attn_pass,
                          check_fn=functools.partial(
                              check,
                              quant_key=quant_key,
                              compile_config=compile_config))
    llm2 = LLM(model,
               enforce_eager=True,
               compilation_config=compile_config,
               gpu_memory_utilization=0.4,
               max_model_len=2048)

    # check outputs
    fused_output = llm2.generate(prompts, sampling_params)

    # transform outputs to format expected by check_outputs_equal
    sample_outs = lambda s: (list(s.token_ids), s.text)
    outs_lst = lambda ros: [sample_outs(ro.outputs[0]) for ro in ros]

    check_outputs_equal(
        outputs_0_lst=outs_lst(unfused_output),
        outputs_1_lst=outs_lst(fused_output),
        name_0="unfused",
        name_1="fused",
    )
    del llm2
    # Clean Dynamo cache to avoid polluting other case(s)
    torch._dynamo.reset()

    # Reset backend to make sure llm2 gets released
    backend = None


@pytest.mark.parametrize(
    "model, quant_key",
    [("amd/Llama-3.1-8B-Instruct-FP8-KV", kFp8StaticTensorSym)])
@pytest.mark.parametrize("use_split_attention", [True, False])
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="Only test CUDA and ROCm")
def test_attention_fusion_v1(example_prompts, monkeypatch, model: str,
                             quant_key: QuantKey, use_split_attention: bool):
    # Clean Dynamo cache to avoid reusing other test cases
    # (for some reason the reset at the end is not enough)
    torch._dynamo.reset()

    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_V1_USE_PREFILL_DECODE_ATTENTION",
                       str(int(use_split_attention)))
    # avoid pickling issues for custom pass
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    #
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    # Certain promps seem differ slightly between fused and unfused
    # (both outputs look reasonable though)
    prompts = example_prompts[:4] + example_prompts[5:6]

    compile_config = CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        full_cuda_graph=True,
        pass_config=PassConfig(
            enable_noop=True,
            enable_attn_fusion=False,
        ),
        use_cudagraph=False,
    )
    model_config = ModelConfig(dtype="bfloat16")
    vllm_config = VllmConfig(compilation_config=compile_config,
                             model_config=model_config)

    vllm_config.compilation_config.inductor_compile_config = {
        "post_grad_custom_post_pass":
        PrinterInductorPass("test_print", vllm_config, always=True)
    }
    with set_current_vllm_config(vllm_config):
        backend = TestBackend()  # also force disable caches
        llm = LLM(model,
                  compilation_config=compile_config,
                  gpu_memory_utilization=0.5,
                  max_model_len=2048)
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=10,
                                         top_p=0.95)
        unfused_output = llm.generate(prompts, sampling_params)
        del llm

    compile_config = CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        full_cuda_graph=True,
        pass_config=PassConfig(
            enable_noop=True,
            enable_attn_fusion=False,  # Added manually for checking
        ),
        use_cudagraph=False,
    )
    vllm_config = VllmConfig(compilation_config=compile_config,
                             model_config=model_config)

    with set_current_vllm_config(vllm_config):
        # Upon init, AttnFusionPass needs attention layers to be registered
        # in config so we initialize it during compilation.

        # Backend registers its pass manager into current config
        # also sets force_disable_caches=True
        # TODO maybe add register method
        backend = TestBackend(  # noqa: F841
            LazyInitPass(AttnFusionPass, vllm_config),
            check_fn=functools.partial(
                check,
                quant_key=quant_key,
                compile_config=get_current_vllm_config().compilation_config))
        llm2 = LLM(model,
                   compilation_config=compile_config,
                   gpu_memory_utilization=0.5,
                   max_model_len=2048)

        # check outputs
        fused_output = llm2.generate(prompts, sampling_params)
        del llm2

    # transform outputs to format expected by check_outputs_equal
    sample_outs = lambda s: (list(s.token_ids), s.text)
    outs_lst = lambda ros: [sample_outs(ro.outputs[0]) for ro in ros]

    check_outputs_equal(
        outputs_0_lst=outs_lst(unfused_output),
        outputs_1_lst=outs_lst(fused_output),
        name_0="unfused",
        name_1="fused",
    )

    # Clean Dynamo cache to avoid polluting other case(s)
    torch._dynamo.reset()
