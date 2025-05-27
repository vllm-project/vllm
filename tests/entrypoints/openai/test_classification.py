# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

from vllm.entrypoints.openai.protocol import ClassificationResponse

from ...utils import RemoteOpenAIServer

MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"
DTYPE = "float32"  # Use float32 to avoid NaN issue


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--dtype",
        DTYPE,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_single_input_classification(server: RemoteOpenAIServer,
                                     model_name: str):
    input_text = "This product was excellent and exceeded my expectations"

    classification_response = requests.post(
        server.url_for("classify"),
        json={
            "model": model_name,
            "input": input_text
        },
    )

    classification_response.raise_for_status()
    output = ClassificationResponse.model_validate(
        classification_response.json())

    assert output.object == "list"
    assert output.model == MODEL_NAME
    assert len(output.data) == 1
    assert hasattr(output.data[0], "label")
    assert hasattr(output.data[0], "probs")


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_multiple_inputs_classification(server: RemoteOpenAIServer,
                                        model_name: str):
    input_texts = [
        "The product arrived on time and works perfectly",
        "I'm very satisfied with my purchase, would buy again",
        "The customer service was helpful and resolved my issue quickly",
        "This product broke after one week, terrible quality",
        "I'm very disappointed with this purchase, complete waste of money",
        "The customer service was rude and unhelpful",
    ]

    classification_response = requests.post(
        server.url_for("classify"),
        json={
            "model": model_name,
            "input": input_texts
        },
    )
    output = ClassificationResponse.model_validate(
        classification_response.json())

    assert len(output.data) == len(input_texts)
    for i, item in enumerate(output.data):
        assert item.index == i
        assert hasattr(item, "label")
        assert hasattr(item, "probs")
        assert len(item.probs) == item.num_classes
        assert item.label in ["Default", "Spoiled"]


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_truncate_prompt_tokens(server: RemoteOpenAIServer, model_name: str):
    long_text = "hello " * 600

    classification_response = requests.post(
        server.url_for("classify"),
        json={
            "model": model_name,
            "input": long_text,
            "truncate_prompt_tokens": 5
        },
    )

    classification_response.raise_for_status()
    output = ClassificationResponse.model_validate(
        classification_response.json())

    assert len(output.data) == 1
    assert output.data[0].index == 0
    assert hasattr(output.data[0], "probs")
    assert output.usage.prompt_tokens == 5
    assert output.usage.total_tokens == 5


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_invalid_truncate_prompt_tokens_error(server: RemoteOpenAIServer,
                                              model_name: str):
    classification_response = requests.post(
        server.url_for("classify"),
        json={
            "model": model_name,
            "input": "test",
            "truncate_prompt_tokens": 513
        },
    )

    error = classification_response.json()
    assert classification_response.status_code == 400
    assert error["object"] == "error"
    assert "truncate_prompt_tokens" in error["message"]


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_empty_input_error(server: RemoteOpenAIServer, model_name: str):
    classification_response = requests.post(
        server.url_for("classify"),
        json={
            "model": model_name,
            "input": ""
        },
    )

    error = classification_response.json()
    assert classification_response.status_code == 400
    assert error["object"] == "error"


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_batch_classification_empty_list(server: RemoteOpenAIServer,
                                         model_name: str):
    classification_response = requests.post(
        server.url_for("classify"),
        json={
            "model": model_name,
            "input": []
        },
    )
    classification_response.raise_for_status()
    output = ClassificationResponse.model_validate(
        classification_response.json())

    assert output.object == "list"
    assert isinstance(output.data, list)
    assert len(output.data) == 0
