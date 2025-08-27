# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests
import torch
import torch.nn.functional as F

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
    assert "truncate_prompt_tokens" in error["error"]["message"]


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
    assert "error" in error


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


@pytest.mark.asyncio
async def test_invocations(server: RemoteOpenAIServer):
    request_args = {
        "model": MODEL_NAME,
        "input": "This product was excellent and exceeded my expectations"
    }

    classification_response = requests.post(server.url_for("classify"),
                                            json=request_args)
    classification_response.raise_for_status()

    invocation_response = requests.post(server.url_for("invocations"),
                                        json=request_args)
    invocation_response.raise_for_status()

    classification_output = classification_response.json()
    invocation_output = invocation_response.json()

    assert classification_output.keys() == invocation_output.keys()
    for classification_data, invocation_data in zip(
            classification_output["data"], invocation_output["data"]):
        assert classification_data.keys() == invocation_data.keys()
        assert classification_data["probs"] == pytest.approx(
            invocation_data["probs"], rel=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_activation(server: RemoteOpenAIServer, model_name: str):
    input_text = ["This product was excellent and exceeded my expectations"]

    async def get_outputs(activation):
        response = requests.post(server.url_for("classify"),
                                 json={
                                     "model": model_name,
                                     "input": input_text,
                                     "activation": activation
                                 })
        outputs = response.json()
        return torch.tensor([x['probs'] for x in outputs["data"]])

    default = await get_outputs(activation=None)
    w_activation = await get_outputs(activation=True)
    wo_activation = await get_outputs(activation=False)

    assert torch.allclose(default, w_activation,
                          atol=1e-2), "Default should use activation."
    assert not torch.allclose(
        w_activation, wo_activation,
        atol=1e-2), "wo_activation should not use activation."
    assert torch.allclose(
        F.softmax(wo_activation, dim=-1), w_activation, atol=1e-2
    ), "w_activation should be close to activation(wo_activation)."


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_pooling(server: RemoteOpenAIServer, model_name: str):
    # pooling api uses ALL pooling, which does not support chunked prefill.
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": "test",
            "encoding_format": "float"
        },
    )
    assert response.json()["error"]["type"] == "BadRequestError"
