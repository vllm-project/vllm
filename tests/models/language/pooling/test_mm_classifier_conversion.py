# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.platforms import current_platform


def update_config(config):
    config.text_config.update({
        "architectures": ["Gemma3ForSequenceClassification"],
        "classifier_from_token": ["A", "B", "C", "D", "E"],
        "method":
        "no_post_processing",
        "id2label": {
            "A": "Chair",
            "B": "Couch",
            "C": "Table",
            "D": "Bed",
            "E": "Cupboard"
        },
    })
    return config


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("google/gemma-3-4b-it",
                     marks=[pytest.mark.core_model, pytest.mark.cpu_model]),
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_models(
    vllm_runner,
    model: str,
    dtype: str,
    monkeypatch,
) -> None:
    if current_platform.is_rocm():
        # ROCm Triton FA does not currently support sliding window attention
        # switch to use ROCm CK FA backend
        monkeypatch.setenv("VLLM_USE_TRITON_FLASH_ATTN", "False")

    messages = [{
        "role":
        "system",
        "content":
        """
    You are a helpful assistant. You will be given a product description
    which may also include an image. Classify the following product into
    one of the categories:

    A = chair
    B = couch
    C = table
    D = bed
    E = cupboard

    You'll answer with exactly one letter (A, B, C, D, or E)."""
    }, {
        "role":
        "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url":
                "https://upload.wikimedia.org/wikipedia/commons/c/c6/Set_of_fourteen_side_chairs_MET_DP110780.jpg"
            }
        }, {
            "type": "text",
            "text": "A fine 19th century piece of furniture."
        }]
    }]

    with vllm_runner(model_name=model,
                     runner="pooling",
                     task="classify",
                     convert="classify",
                     load_format="auto",
                     hf_overrides=update_config,
                     override_pooler_config={"pooling_type": "LAST"},
                     max_model_len=512,
                     enforce_eager=True,
                     tensor_parallel_size=1,
                     disable_log_stats=True,
                     dtype=dtype) as vllm_model:

        llm = vllm_model.get_llm()
        prompts = llm.preprocess_chat(messages)

        result = llm.classify(prompts)
        assert result[0].outputs.probs[0] > 0.95
        assert all(c < 0.05 for c in result[0].outputs.probs[1:])
