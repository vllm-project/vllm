# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.compile.backend import TestBackend
from tests.models.utils import check_outputs_equal
from vllm import LLM, SamplingParams
from vllm.compilation.fusion import QUANT_OPS, kFp8StaticTensorSym
from vllm.compilation.fusion_attn import ATTN_OP, AttnFusionPass
from vllm.compilation.fx_utils import find_op_nodes
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import CompilationConfig, CompilationLevel, VllmConfig

MODEL = "amd/Llama-3.1-8B-Instruct-FP8-KV"
test_backend = TestBackend()


@pytest.mark.parametrize("num_prompts", [4])
def test_attention_fusion2(example_prompts, num_prompts):
    # TODO(luka): use all prompts if possible (after recompilation resolved)
    # https://github.com/vllm-project/vllm/issues/19391
    prompts = example_prompts[:num_prompts]

    # For some reason, if we compile the first LLM with Dynamo,
    # the second time the compiler is not invoked.
    # Hence, we use eager mode as the baseline.
    llm = LLM(MODEL,
              enforce_eager=True,
              compilation_config=CompilationConfig(
                  level=CompilationLevel.NO_COMPILATION),
              gpu_memory_utilization=0.9)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=10,
                                     top_p=0.95)

    unfused_output = llm.generate(prompts, sampling_params)
    del llm

    compile_config = CompilationConfig(
        # DYNAMO_AS_IS triggers custom backend & does full Dynamo compilation
        level=CompilationLevel.DYNAMO_AS_IS,
        backend="tests.compile.test_fusion_attn.test_backend",
        use_cudagraph=False,
    )
    vllm_config = VllmConfig(compilation_config=compile_config)

    # AttnFusionPass needs attention layers to be registered in config upon init
    # so we initialize it during compilation.
    attn_pass = lambda *args, **kw: AttnFusionPass(vllm_config)(*args, **kw)
    test_backend.custom_passes += [NoOpEliminationPass(vllm_config), attn_pass]
    llm2 = LLM(MODEL,
               compilation_config=compile_config,
               gpu_memory_utilization=0.9)

    # Check quant ops
    test_backend.check_before_ops([QUANT_OPS[kFp8StaticTensorSym]],
                                  fully_replaced=False)

    # attention ops present in both, just output_scale param changes
    attn_nodes_pre = list(find_op_nodes(ATTN_OP, test_backend.graph_pre_pass))
    attn_nodes_post = list(find_op_nodes(ATTN_OP,
                                         test_backend.graph_post_pass))
    assert len(attn_nodes_pre) == len(attn_nodes_post)

    for pre_node, post_node in zip(attn_nodes_pre, attn_nodes_post):
        assert pre_node.kwargs["output_scale"] is None
        assert post_node.kwargs["output_scale"] is not None

    # check outputs
    fused_output = llm2.generate(prompts, sampling_params)

    req_outs = lambda ro: [(list(s.token_ids), s.text) for s in ro.outputs]
    outs_lst = lambda ros: [tuple(zip(*req_outs(ro))) for ro in ros]

    check_outputs_equal(
        outputs_0_lst=outs_lst(unfused_output),
        outputs_1_lst=outs_lst(fused_output),
        name_0="unfused",
        name_1="fused",
    )
