import gc

import pytest
import torch

from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

MODELS = [
    "JackFram/llama-68m",
    # "facebook/opt-125m",
]


# SANG-TODO enforce_eager = True and chunked prefill currently doesn't work.
# TODO(sang): Add chunked prefill parameters.
# @pytest.mark.parametrize("model", MODELS)
# @pytest.mark.parametrize("dtype", ["half"])
# @pytest.mark.parametrize("max_tokens", [128])
# @pytest.mark.parametrize("max_chunked_prefill_len", [-1, 16, 64])
# @pytest.mark.parametrize("max_num_prompt_seqs", [1, 2, 100])
# @pytest.mark.parametrize("block_size", [32])
# @pytest.mark.parametrize("tensor_parallel_size", [1, 2])
# @pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("max_chunked_prefill_len", [500])
@pytest.mark.parametrize("max_num_prompt_seqs", [256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("enforce_eager", [False])
def test_models(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    max_chunked_prefill_len: int,
    max_num_prompt_seqs: int,
    block_size: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
) -> None:
    """ verify the flash attention has the same output
    as page attention """
    if torch.cuda.device_count() < tensor_parallel_size:
        pytest.skip(
            f"{torch.cuda.device_count()=} is smaller than {tensor_parallel_size=}"
        )
    print("loading page attention models..")
    pg_model = vllm_runner(model, dtype=dtype, enforce_eager=enforce_eager)
    expected_outputs = []

    print("generating tokens...")
    expected_outputs.extend(
        pg_model.generate_greedy(example_prompts, max_tokens))
    print("generating tokens finished")

    del pg_model

    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    flash_attn_output_by_batches = []
    flash_attn_model = vllm_runner(
        model,
        dtype=dtype,
        block_size=block_size,
        tensor_parallel_size=tensor_parallel_size,
        max_chunked_prefill_len=max_chunked_prefill_len,
        enforce_eager=enforce_eager)
    for i in range(5, 6):
        prompts = [example_prompts[j % len(example_prompts)] for j in range(i)]
        breakpoint()
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
            # print("expected, ",vllm_output_str, "\n")
            # print("actual:, ", fa_output_str, "\n")
            assert fa_output_ids == vllm_output_ids, (
                f"Test{i}:\nflash ids: {fa_output_ids}\nvLLM ids: {vllm_output_ids}"
                f"Test{i}:\nflash output: {fa_output_str!r}\nvLLM output: {vllm_output_str!r}"
            )
