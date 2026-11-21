# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.models.utils import check_logprobs_close
from vllm.config import CompilationConfig, CUDAGraphMode, LoggingConfig
from vllm.platforms import current_platform

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea:",
    "List three prime numbers greater than 100:",
    "Q: What is 17 times 23?\nA:",
]

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")


def _compilation_config(splitting_ops, use_inductor_graph_partition):
    return CompilationConfig(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=splitting_ops,
        use_inductor_graph_partition=use_inductor_graph_partition,
    )


def _moe_splitting_ops(use_inductor_graph_partition):
    splitting_ops = list(CompilationConfig._attention_ops)
    if not use_inductor_graph_partition:
        # Only added by default when splitting_ops is None.
        splitting_ops += [
            "vllm::unified_kv_cache_update",
            "vllm::unified_mla_kv_cache_update",
        ]
    return splitting_ops + ["vllm::moe_forward", "vllm::moe_forward_shared"]


@pytest.mark.parametrize("use_inductor_graph_partition", [False, True])
def test_moe_forward_as_splitting_op(vllm_runner, use_inductor_graph_partition):
    """A splitting op runs outside the captured pieces, so the piece after
    `vllm::moe_forward` must not read the MoE output from the address it had
    at capture time (#61150)."""
    outputs = []
    for splitting_ops in (None, _moe_splitting_ops(use_inductor_graph_partition)):
        with vllm_runner(
            "ibm-granite/granite-3.0-1b-a400m-instruct",
            max_model_len=1024,
            compilation_config=_compilation_config(
                splitting_ops, use_inductor_graph_partition
            ),
        ) as llm:
            outputs.append(
                llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="default_splitting_ops",
        name_1="moe_splitting_ops",
    )


def _shrink_qwen3_5_moe(hf_config):
    hf_config.get_text_config().update(
        {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "mtp_num_hidden_layers": 0,
        }
    )
    return hf_config


def test_moe_forward_shared_as_splitting_op(vllm_runner):
    """Same as above for `vllm::moe_forward_shared`. With DEBUG logging,
    CUDAGraphWrapper asserts that replayed inputs keep their capture
    addresses, so dummy weights are enough."""
    with vllm_runner(
        "Qwen/Qwen3.5-35B-A3B",
        load_format="dummy",
        hf_overrides=_shrink_qwen3_5_moe,
        max_model_len=1024,
        enable_chunked_prefill=None,
        limit_mm_per_prompt={"image": 0, "video": 0},
        logging_config=LoggingConfig(log_level="DEBUG"),
        compilation_config=_compilation_config(_moe_splitting_ops(False), False),
    ) as llm:
        llm.generate_greedy(PROMPTS, max_tokens=8)
