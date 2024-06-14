"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_llama_embedding.py`.
"""
import pytest
import torch
import torch.nn.functional as F

from tests.nm_utils.utils_skip import should_skip_test_group

if should_skip_test_group(group_name="TEST_MODELS"):
    pytest.skip("TEST_MODELS=DISABLE, skipping model test group",
                allow_module_level=True)

MODELS = [
    "intfloat/e5-mistral-7b-instruct",
]


def compare_embeddings(embeddings1, embeddings2):
    similarities = [
        F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=0)
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with hf_runner(model, dtype=dtype, is_embedding_model=True) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.encode(example_prompts)

    similarities = compare_embeddings(hf_outputs, vllm_outputs)
    all_similarities = torch.stack(similarities)
    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
