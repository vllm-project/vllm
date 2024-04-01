"""Checks the memory usage of the sparse model is < memory usage of the
dense model by checking that the number of KV cache blocks is
bigger for the sparse model rather than the dense model. vLLM pre-allocates
the memory for the KV-cache after checking availability once the model
is loaded. This implies that using a compressed model should give more space
for the KV cache and thus more allocated blocks.

Run `pytest tests/models/test_sparse_memory.py --forked`.
"""

import gc

import pytest
import torch

MODEL_FORMAT_EXTRABLOCKS = [
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50", "sparse_w16a16", 2000),
    ("nm-testing/OpenHermes-2.5-Mistral-7B-pruned2.4",
     "semi_structured_sparse_w16a16", 2000),
]


@pytest.mark.parametrize("model_format_extrablocks", MODEL_FORMAT_EXTRABLOCKS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_models(
    vllm_runner_nm,
    example_prompts,
    model_format_extrablocks,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_name, sparsity, num_extra_blocks = model_format_extrablocks
    dense_model = vllm_runner_nm(model_name=model_name,
                                 sparsity=None,
                                 dtype=dtype,
                                 max_model_len=1024)
    dense_num_kv_blocks = (dense_model.model.llm_engine.scheduler.
                           block_manager.gpu_allocator.num_blocks)

    del dense_model
    torch.cuda.empty_cache()
    gc.collect()

    sparse_model = vllm_runner_nm(model_name=model_name,
                                  sparsity=sparsity,
                                  dtype=dtype,
                                  max_model_len=1024)
    sparse_num_kv_blocks = (sparse_model.model.llm_engine.scheduler.
                            block_manager.gpu_allocator.num_blocks)

    del sparse_model
    torch.cuda.empty_cache()
    gc.collect()

    assert sparse_num_kv_blocks > dense_num_kv_blocks + num_extra_blocks, (
        f"Test{model_name}: Sparse model KV cache size {sparse_num_kv_blocks} "
        f"not bigger than dense model KV cache size {dense_num_kv_blocks} + "
        f"expected num_extra_blocks {num_extra_blocks}")
