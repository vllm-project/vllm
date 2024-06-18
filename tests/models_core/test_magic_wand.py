"""Compare the outputs of a sparse model vs sparse model running dense.
Note: sparse kernels do not have bitwise correctness vs the dense models. 
As a result, in this test, we just confirm that the top selected tokens of the 
sparse models are in the top N selections of same model running dense.

Run `pytest tests/models_core/test_magic_wand.py`.
"""

import gc

import pytest

from tests.models.utils import check_logprobs_close
from tests.nm_utils.utils_skip import should_skip_test_group

if should_skip_test_group(group_name="TEST_MODELS_CORE"):
    pytest.skip("TEST_MODELS_CORE=DISABLE, skipping core model test group",
                allow_module_level=True)

MAX_MODEL_LEN = 1024
MODEL_FORMAT_EXTRABLOCKS = [
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50", "sparse_w16a16", 1500),
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned2.4",
     "semi_structured_sparse_w16a16", 1500),
]


@pytest.mark.parametrize("model_format_extrablocks", MODEL_FORMAT_EXTRABLOCKS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_magic_wand(
    vllm_runner,
    example_prompts,
    model_format_extrablocks,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_name, sparsity, num_extra_blocks = model_format_extrablocks
    dense_model = vllm_runner(model_name=model_name,
                              enforce_eager=True,
                              sparsity=None,
                              dtype=dtype,
                              max_model_len=1024)
    dense_gpu_alloc = (
        dense_model.model.llm_engine.scheduler.block_manager.gpu_allocator)
    dense_num_kv_blocks = dense_gpu_alloc.num_blocks
    dense_outputs = dense_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del dense_model
    gc.collect()

    sparse_model = vllm_runner(
        model_name=model_name,
        enforce_eager=True,
        sparsity=sparsity,
        dtype=dtype,
        max_model_len=1024,
    )
    sparse_gpu_alloc = (
        sparse_model.model.llm_engine.scheduler.block_manager.gpu_allocator)
    sparse_num_kv_blocks = sparse_gpu_alloc.num_blocks
    sparse_outputs = sparse_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)
    del sparse_model

    # Confirm the memory is saved.
    assert sparse_num_kv_blocks > dense_num_kv_blocks + num_extra_blocks, (
        f"Test{model_name}: Sparse model KV cache size {sparse_num_kv_blocks} "
        f"not bigger than dense model KV cache size {dense_num_kv_blocks} + "
        f"expected num_extra_blocks {num_extra_blocks}")

    # Confirm the generations are similar.
    check_logprobs_close(
        outputs_0_lst=dense_outputs,
        outputs_1_lst=sparse_outputs,
        name_0="dense",
        name_1="sparse",
    )
