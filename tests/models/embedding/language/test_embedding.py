"""Compare the embedding outputs of HF and vLLM models.

Run `pytest tests/models/embedding/language/test_embedding.py`.
"""
import pytest

from ..utils import check_embeddings_close

# Model, Guard
MODELS = [
    "intfloat/e5-mistral-7b-instruct",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-multilingual-gemma2",
]

ENCODER_ONLY = [
    "BAAI/bge-base-en-v1.5",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    monkeypatch,
    hf_runner,
    vllm_runner,
    example_prompts,
    model,
    dtype: str,
) -> None:
    if model in ENCODER_ONLY:
        monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "XFORMERS")

    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and vllm_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    with hf_runner(model, dtype=dtype,
                   is_sentence_transformer=True) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with vllm_runner(model, dtype=dtype, max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.encode(example_prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )
