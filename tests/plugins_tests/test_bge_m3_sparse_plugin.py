# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import requests

# Test configuration for BGE-M3 sparse plugin
from tests.plugins.bge_m3_sparse_plugin.bge_m3_sparse_processor.types import (
    SparseEmbeddingResponse,
)
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse

model_config = {
    "model_name": "BAAI/bge-m3",
    "plugin": "bge_m3_sparse_plugin",
    "test_input": "What is the capital of France?",
    "hf_overrides": {"architectures": ["BgeM3EmbeddingModel"], "head_dtype": "float16"},
}


@pytest.fixture(scope="function")
def server():
    args = [
        "--runner",
        "pooling",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--hf_overrides",
        json.dumps(model_config["hf_overrides"]),
        "--io-processor-plugin",
        model_config["plugin"],
    ]

    with RemoteOpenAIServer(model_config["model_name"], args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_bge_m3_sparse_plugin_online(server: RemoteOpenAIServer):
    """Test BGE-M3 sparse plugin in online mode via API."""
    request_payload = {
        "model": model_config["model_name"],
        "data": {"input": model_config["test_input"], "return_tokens": True},
    }

    ret = requests.post(
        server.url_for("pooling"),
        json=request_payload,
    )

    response = ret.json()

    # Verify the request response is in the correct format
    assert (parsed_response := IOProcessorResponse(**response))

    # Verify the output is formatted as expected for this plugin
    assert parsed_response.data
    assert len(parsed_response.data) > 0

    data_entry = parsed_response.data[0]
    assert data_entry.object == "sparse-embedding"
    assert hasattr(data_entry, "sparse_embedding")

    # Verify sparse embedding format
    sparse_embedding = data_entry.sparse_embedding
    assert isinstance(sparse_embedding, list)
    if sparse_embedding:
        entry = sparse_embedding[0]
        assert hasattr(entry, "token_id")
        assert hasattr(entry, "weight")
        # When return_tokens=True, token should be present
        assert entry.token is not None
        assert isinstance(entry.token_id, int)
        assert isinstance(entry.weight, float)
        assert entry.weight >= 0  # SPLADE outputs are non-negative

    # Verify usage information
    assert parsed_response.usage
    assert parsed_response.usage.prompt_tokens > 0
    assert parsed_response.usage.total_tokens == parsed_response.usage.prompt_tokens


@pytest.mark.asyncio
async def test_bge_m3_sparse_plugin_online_no_tokens(server: RemoteOpenAIServer):
    """Test BGE-M3 sparse plugin in online mode without returning tokens."""
    request_payload = {
        "model": model_config["model_name"],
        "input": model_config["test_input"],
        "return_tokens": False,
    }

    ret = requests.post(
        server.url_for("pooling"),
        json=request_payload,
    )

    response = ret.json()
    print(f"Response: {response}")

    # Verify the request response is in the correct format
    assert (parsed_response := SparseEmbeddingResponse(**response))

    # Verify sparse embedding format
    sparse_embedding = parsed_response.data[0].sparse_embedding
    if sparse_embedding:
        entry = sparse_embedding[0]
        # When return_tokens=False, token should be None
        assert entry.token is None


@pytest.mark.parametrize(
    "return_tokens",
    [True, False],
)
def test_bge_m3_sparse_plugin_offline(vllm_runner, return_tokens: bool):
    """Test BGE-M3 sparse plugin in offline mode."""
    prompt = {
        "data": {
            "input": model_config["test_input"],
            "return_tokens": return_tokens,
        }
    }

    with vllm_runner(
        model_config["model_name"],
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=32,
        io_processor_plugin=model_config["plugin"],
        hf_overrides=model_config["hf_overrides"],
        default_torch_num_threads=1,
    ) as llm_runner:
        llm = llm_runner.get_llm()
        output = llm.encode(prompt, pooling_task="token_classify")

    print(f"Output: {output}")

    # Verify output structure
    assert len(output) > 0
    result = output[0]
    assert hasattr(result, "outputs")

    # The outputs should be SparseEmbeddingResponse
    response = result.outputs
    assert hasattr(response, "data")
    assert hasattr(response, "usage")

    # Verify response data
    data_entry = response.data[0]
    assert data_entry.object == "sparse-embedding"
    assert hasattr(data_entry, "sparse_embedding")

    # Verify sparse embedding format
    sparse_embedding = data_entry.sparse_embedding
    assert isinstance(sparse_embedding, list)
    if sparse_embedding:
        entry = sparse_embedding[0]
        assert hasattr(entry, "token_id")
        assert hasattr(entry, "weight")
        assert isinstance(entry.token_id, int)
        assert isinstance(entry.weight, float)
        assert entry.weight >= 0  # SPLADE outputs are non-negative

        # Verify token presence based on return_tokens
        if return_tokens:
            # For offline mode, tokens might be None depending on renderer
            # but token_id and weight should always be present
            assert isinstance(entry.token_id, int)
            assert isinstance(entry.weight, (float, int))


def test_bge_m3_sparse_plugin_offline_multiple_inputs(vllm_runner):
    """Test BGE-M3 sparse plugin with multiple inputs in offline mode."""
    prompts = {
        "data": {
            "input": [
                "What is the capital of France?",
                "What is the capital of Germany?",
                "What is the capital of Spain?",
            ],
            "return_tokens": True,
        }
    }

    with vllm_runner(
        model_config["model_name"],
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=32,
        io_processor_plugin=model_config["plugin"],
        hf_overrides=model_config["hf_overrides"],
        default_torch_num_threads=1,
    ) as llm_runner:
        llm = llm_runner.get_llm()
        outputs = llm.encode(prompts, pooling_task="token_classify")

    print(f"Outputs: {outputs}")

    # Verify output structure
    assert len(outputs) == 3
    for i, output in enumerate(outputs):
        result = output
        assert hasattr(result, "outputs")

        response = result.outputs
        assert len(response.data) > 0

        # Each output should have sparse embeddings
        sparse_embedding = response.data[0].sparse_embedding
        assert isinstance(sparse_embedding, list)

        # Verify usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens
