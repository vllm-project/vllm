"""Compare the outputs of a sparse model vs sparse model running dense.
Note: sparse kernels do not have bitwise correctness vs the dense models. 
As a result, in this test, we just confirm that the top selected tokens of the 
sparse models are in the top N selections of same model running dense.

Run `pytest tests/models/test_compressed.py`.
"""

import pytest

from tests.models.utils import check_logprobs_close

MAX_MODEL_LEN = 1024
MODEL_FORMAT_PAIRS = [
    ("nm-testing/TinyLlama-1.1B-Chat-v1.0-pruned2.4",
     "semi_structured_sparse_w16a16"),
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50", "sparse_w16a16"),
]


@pytest.mark.parametrize("model_format_pairs", MODEL_FORMAT_PAIRS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    vllm_runner,
    example_prompts,
    model_format_pairs,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_name, sparsity = model_format_pairs

    sparse_model = vllm_runner(model_name=model_name,
                               sparsity=sparsity,
                               dtype=dtype,
                               max_model_len=MAX_MODEL_LEN)
    sparse_outputs = sparse_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del sparse_model

    dense_model = vllm_runner(model_name=model_name,
                              sparsity=None,
                              dtype=dtype,
                              max_model_len=MAX_MODEL_LEN)
    dense_outputs = dense_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del dense_model

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=dense_outputs,
        outputs_1_lst=sparse_outputs,
        name_0="dense",
        name_1="sparse",
    )
