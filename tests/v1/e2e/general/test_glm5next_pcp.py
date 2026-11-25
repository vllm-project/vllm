# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight regression for head-sharded KDA, chunking, graphs and MTP."""

import os

import pytest

from tests.models.utils import check_logprobs_close
from tests.utils import create_new_process_for_each_test


def _indexer_pcp_sizes(worker):
    from vllm.v1.attention.backends.mla.indexer import Glm5NextIndexerMetadataBuilder

    runner = worker.model_runner
    groups = list(runner.attn_groups)
    if runner.speculator is not None:
        groups += list(runner.speculator.attn_groups)
    sizes = []
    for row in groups:
        for group in row:
            builder = group.get_metadata_builder()
            if isinstance(builder, Glm5NextIndexerMetadataBuilder):
                for item in (builder, builder.pcp_write_metadata_builder):
                    if item is not None:
                        sizes.append(
                            item.vllm_config.parallel_config.prefill_context_parallel_size
                        )
    return sizes


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.parametrize(
    "graph_mode", ["NONE", "PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"]
)
@pytest.mark.parametrize("num_spec", [0, 3])
@create_new_process_for_each_test()
def test_glm5next_pcp_matches_tp(vllm_runner, graph_mode, num_spec):
    model = os.getenv("GLM53_FLASH_MODEL", "zai-org/GLM-5.3-Flash")
    prompts = [
        "Question: Alice has 17 apples and buys 8 more. How many apples? Answer:",
        "The following numbers form a sequence: "
        + "1, 2, 3, 4, " * 151
        + "\nQuestion: What is 13 plus 29? Answer:",
        "Question: A box has 7 rows of 9 pencils. How many pencils? Answer:",
    ]
    options = dict(
        language_model_only=True,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        block_size=256,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        seed=0,
    )
    with vllm_runner(
        model, tensor_parallel_size=8, enforce_eager=True, **options
    ) as baseline:
        tokenizer = baseline.llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        references = [
            baseline.generate_greedy_logprobs(prompts, 32, 5) for _ in range(2)
        ]
    speculative_config = (
        {"method": "mtp", "num_speculative_tokens": num_spec} if num_spec else None
    )
    with vllm_runner(
        model,
        tensor_parallel_size=4,
        prefill_context_parallel_size=2,
        enforce_eager=graph_mode == "NONE",
        compilation_config={
            "cudagraph_mode": graph_mode,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
        },
        speculative_config=speculative_config,
        **options,
    ) as pcp:
        for sizes in pcp.llm.collective_rpc(_indexer_pcp_sizes):
            assert sizes and all(size == 2 for size in sizes)
        for reference in references:
            actual = pcp.generate_greedy_logprobs(prompts, 32, 5)
            check_logprobs_close(
                outputs_0_lst=reference,
                outputs_1_lst=actual,
                name_0="TP8",
                name_1="TP4-PCP2",
            )
