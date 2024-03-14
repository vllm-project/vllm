import gc

import pytest
from tests.conftest import example_prompts
import torch

from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from vllm import SamplingParams

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    # "JackFram/llama-68m",
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
@pytest.mark.parametrize("max_chunked_prefill_len", [16])
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("num", [3])
def test_models(
    example_prompts,
    vllm_runner,
    model: str,
    dtype: str,
    max_tokens: int,
    max_chunked_prefill_len: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    num,
) -> None:
    """ verify the chunked prefill attention has the same output as vLLM."""
    if torch.cuda.device_count() < tensor_parallel_size:
        pytest.skip(
            f"{torch.cuda.device_count()=} is smaller than {tensor_parallel_size=}"
        )

    def evaluate(init_llm):
        llm = init_llm()
        outputs = llm.generate_greedy(example_prompts, max_tokens=max_tokens)
        token_ids_list = []
        output_str_list = []

        for i in range(len(outputs)):
            token_ids = outputs[i][0]
            output_str = outputs[i][1]
            token_ids_list.append(token_ids)
            output_str_list.append(output_str)

        # clean up.
        del llm
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_default_dtype(torch.float32)
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()

        return token_ids_list, output_str_list

    vllm_token_ids, vllm_str = evaluate(lambda: vllm_runner(model, dtype=dtype, enforce_eager=enforce_eager))
    chunked_prefill_token_ids, chunked_str = evaluate(lambda: vllm_runner(
        model,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_chunked_prefill_len=max_chunked_prefill_len,
        enforce_eager=enforce_eager))

    for i in range(len(vllm_token_ids)):
        print(f"TEST {i}")
        print(f"{len(vllm_token_ids[i])=} {vllm_token_ids[i]=}\n{vllm_str[i]=}")
        print(f"{len(chunked_prefill_token_ids[i])=} {chunked_prefill_token_ids[i]=}\n{chunked_str[i]=}\n")
        assert vllm_token_ids[i] == chunked_prefill_token_ids[i]

