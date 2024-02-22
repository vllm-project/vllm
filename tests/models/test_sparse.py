"""Compare the outputs of a sparse model running sparse compared to running dense.
Note: running sparse vs dense does not have bitwise correctness due to differences 
in computation orders. As a result, in this test, we just confirm that the top 
selected tokens of the model running sparse and running dense are in the top 3 
selections of eachother.
Note: When running sparse we internally use locks to synchronize the threads. This 
can result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.
Run `pytest tests/models/test_sparse.py`.
"""

import pytest
import torch
from dataclasses import dataclass
from vllm.model_executor.layers.sparsity import SemiStructuredSparseW16A16Config

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]
semi_structured_supported = (
    SemiStructuredSparseW16A16Config.get_min_capability() >= capability)


@dataclass
class SparseModel:
    name: str
    sparsity: str


models = [
    SparseModel("nm-testing/OpenHermes-2.5-Mistral-7B-pruned50",
                "sparse_w16a16"),
    SparseModel(
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-pruned2.4",
        "semi_structured_sparse_w16a16"
        if semi_structured_supported else "sparse_w16a16"),
]


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("model", models)
#@pytest.mark.parametrize("dtype", ["half", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [3])
def test_models(
    vllm_runner,
    example_prompts,
    model: SparseModel,
    #dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    sparse_model = vllm_runner(model.name,
                               sparsity=model.sparsity,
                               enforce_eager=True)
    sparse_outputs = sparse_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    # Note: not sure why, but deleting just the model on Ada Lovelace
    #   does not free the GPU memory. On Ampere, deleting the just model
    #   frees the memory.
    del sparse_model.model.llm_engine.driver_worker
    del sparse_model

    dense_model = vllm_runner(model.name, sparsity=None, enforce_eager=True)
    dense_outputs = dense_model.generate_greedy_logprobs(
        example_prompts, max_tokens, num_logprobs)

    # Note: not sure why, but deleting just the model on Ada Lovelace
    #   does not free the GPU memory. On Ampere, deleting the just model
    #   frees the memory.
    del dense_model.model.llm_engine.driver_worker
    del dense_model

    # loop through the prompts
    for prompt_idx in range(len(example_prompts)):
        dense_output_ids, dense_output_str, dense_logprobs = dense_outputs[
            prompt_idx]
        sparse_output_ids, sparse_output_str, sparse_logprobs = sparse_outputs[
            prompt_idx]

        for idx, (dense_output_id, sparse_output_id) in enumerate(
                zip(dense_output_ids, sparse_output_ids)):
            # If sequence is not an exact match,
            if sparse_output_id != dense_output_id:
                # Each predicted token must be in top 5 of the other's
                assert dense_output_id in sparse_logprobs[idx], (
                    f"Test{prompt_idx}:\ndense:\t{dense_output_str!r}\nsparse:\t{sparse_output_str!r}"
                )
                assert sparse_output_id in dense_logprobs[idx], (
                    f"Test{prompt_idx}:\ndense:\t{dense_output_str!r}\nsparse:\t{sparse_output_str!r}"
                )

                # Break out since sequences will now diverge.
                break
