import gc

from typing import List

import pytest
import torch

from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

MODELS = [
    "JackFram/llama-68m",
    "facebook/opt-125m",
]

TEST_PROMPTS = [
    # pylint: disable=line-too-long
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    # Different between page attention and flash attention.
    # "Describe the basic components of a neural network and how it can be trained.",
    "Write a short story about a robot that dreams for the first time.",
    "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.",
    "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'",
]


# TODO(sang): Add chunked prefill parameters.
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("block_size", [256])
def test_models(
    vllm_runner,
    model: str,
    dtype: str,
    max_tokens: int,
    block_size: int,
) -> None:
    """ verify the flash attention has the same output
    as page attention """
    print("loading page attention models..")
    pg_model = vllm_runner(model, dtype=dtype)
    expected_outputs = []

    print("generating tokens...")
    expected_outputs.extend(pg_model.generate_greedy(TEST_PROMPTS, max_tokens))
    print("generating tokens finished")

    del pg_model

    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    flash_attn_model = vllm_runner(model,
                                   dtype=dtype,
                                   flash_style=True,
                                   block_size=block_size)
    flash_attn_output_by_batches = []
    for i in range(10):
        prompts = [TEST_PROMPTS[j % len(TEST_PROMPTS)] for j in range(i)]
        flash_attn_output_by_batches.append(
            flash_attn_model.generate_greedy(prompts, max_tokens))

    del flash_attn_model

    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    for flash_attn_outputs in flash_attn_output_by_batches:
        for i in range(len(flash_attn_outputs)):
            fa_output_ids, fa_output_str = flash_attn_outputs[i]
            vllm_output_ids, vllm_output_str = expected_outputs[
                i % len(expected_outputs)]
            assert fa_output_ids == vllm_output_ids, (
                f"Test{i}:\flash ids: {fa_output_ids}\nvLLM ids: {vllm_output_ids}"
                f"Test{i}:\nflash output: {fa_output_str!r}\nvLLM output: {vllm_output_str!r}"
            )
