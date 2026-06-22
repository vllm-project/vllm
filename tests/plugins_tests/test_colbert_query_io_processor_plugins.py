# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import TypedDict

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse


# Test configuration for ColBERT query plugin
class ModelConfig(TypedDict):
    model_name: str
    plugin: str
    query_input: str
    document_input: str
    hf_overrides: str
    embedding_dim: int
    query_maxlen: int


model_config: ModelConfig = {
    "model_name": "jinaai/jina-colbert-v2",
    "plugin": "colbert_query_plugin",
    "query_input": "What is machine learning?",
    "document_input": "Machine learning is a subset of artificial intelligence.",
    "hf_overrides": json.dumps({"architectures": ["ColBERTJinaRobertaModel"]}),
    "embedding_dim": 128,
    "query_maxlen": 32,
}


def _get_attr_or_val(obj: object | dict, key: str):
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return getattr(obj, key, None)


def _check_token_embeddings(entry, expected_input_type: str):
    assert _get_attr_or_val(entry, "object") == "embedding"
    assert _get_attr_or_val(entry, "input_type") == expected_input_type

    embedding = _get_attr_or_val(entry, "embedding")
    assert isinstance(embedding, list) and len(embedding) > 0
    for token_embedding in embedding:
        assert isinstance(token_embedding, list)
        assert len(token_embedding) == model_config["embedding_dim"]
    return embedding


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--enforce-eager",
        "--max-num-seqs",
        "32",
        "--trust-remote-code",
        "--hf_overrides",
        model_config["hf_overrides"],
        "--io-processor-plugin",
        model_config["plugin"],
    ]

    with RemoteOpenAIServer(model_config["model_name"], args) as remote_server:
        yield remote_server


def _post_pooling(server: RemoteOpenAIServer, data: dict):
    request_payload = {
        "model": model_config["model_name"],
        "task": "plugin",
        "data": data,
    }
    ret = requests.post(server.url_for("pooling"), json=request_payload)
    ret.raise_for_status()
    response = ret.json()
    parsed_response = IOProcessorResponse(**response).data
    assert parsed_response
    return parsed_response


def test_colbert_query_plugin_query_online(server: RemoteOpenAIServer):
    """Queries are expanded to exactly query_maxlen token vectors."""
    parsed_response = _post_pooling(
        server, {"input": model_config["query_input"], "input_type": "query"}
    )

    data = _get_attr_or_val(parsed_response, "data")
    assert len(data) == 1

    embedding = _check_token_embeddings(data[0], "query")
    assert len(embedding) == model_config["query_maxlen"]

    usage = _get_attr_or_val(parsed_response, "usage")
    assert _get_attr_or_val(usage, "prompt_tokens") == model_config["query_maxlen"]


def test_colbert_query_plugin_document_online(server: RemoteOpenAIServer):
    """Documents return one vector per token, with no mask expansion."""
    parsed_response = _post_pooling(
        server, {"input": model_config["document_input"], "input_type": "document"}
    )

    data = _get_attr_or_val(parsed_response, "data")
    assert len(data) == 1

    embedding = _check_token_embeddings(data[0], "document")
    # No query expansion: number of vectors tracks the input length.
    assert len(embedding) != model_config["query_maxlen"]

    usage = _get_attr_or_val(parsed_response, "usage")
    assert _get_attr_or_val(usage, "prompt_tokens") == len(embedding)


def test_colbert_query_plugin_missing_input_type_online(server: RemoteOpenAIServer):
    """input_type is required; omitting it is rejected."""
    request_payload = {
        "model": model_config["model_name"],
        "task": "plugin",
        "data": {"input": model_config["document_input"]},
    }
    ret = requests.post(server.url_for("pooling"), json=request_payload)
    assert ret.status_code == 400


def test_colbert_query_plugin_batch_online(server: RemoteOpenAIServer):
    """A list input returns one entry per prompt."""
    queries = ["What is machine learning?", "What is deep learning?"]
    parsed_response = _post_pooling(server, {"input": queries, "input_type": "query"})

    data = _get_attr_or_val(parsed_response, "data")
    assert len(data) == len(queries)
    for i, entry in enumerate(data):
        assert _get_attr_or_val(entry, "index") == i
        embedding = _check_token_embeddings(entry, "query")
        assert len(embedding) == model_config["query_maxlen"]


@pytest.mark.parametrize("input_type", ["query", "document"])
def test_colbert_query_plugin_offline(vllm_runner, input_type: str):
    """Test the ColBERT query plugin in offline mode."""
    input_text = (
        model_config["query_input"]
        if input_type == "query"
        else model_config["document_input"]
    )
    prompt = {
        "data": {
            "input": input_text,
            "input_type": input_type,
        }
    }

    with vllm_runner(
        model_config["model_name"],
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=32,
        trust_remote_code=True,
        io_processor_plugin=model_config["plugin"],
        hf_overrides=json.loads(model_config["hf_overrides"]),
        default_torch_num_threads=1,
    ) as llm_runner:
        llm = llm_runner.get_llm()
        pooler_output = llm.encode(prompt, pooling_task="plugin")

    response = pooler_output[0].outputs
    assert len(response.data) == 1

    embedding = _check_token_embeddings(response.data[0], input_type)
    if input_type == "query":
        assert len(embedding) == model_config["query_maxlen"]
    else:
        assert len(embedding) != model_config["query_maxlen"]

    assert response.usage.prompt_tokens == len(embedding)
    assert response.usage.total_tokens == response.usage.prompt_tokens


def test_colbert_query_plugin_offline_multiple_inputs(vllm_runner):
    """Test the ColBERT query plugin with multiple inputs in offline mode."""
    queries = [
        "What is machine learning?",
        "What is deep learning?",
        "Why?",
    ]
    prompts = {
        "data": {
            "input": queries,
            "input_type": "query",
        }
    }

    with vllm_runner(
        model_config["model_name"],
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=32,
        trust_remote_code=True,
        io_processor_plugin=model_config["plugin"],
        hf_overrides=json.loads(model_config["hf_overrides"]),
        default_torch_num_threads=1,
    ) as llm_runner:
        llm = llm_runner.get_llm()
        pooler_output = llm.encode(prompts, pooling_task="plugin")

    response = pooler_output[0].outputs
    assert len(response.data) == len(queries)

    for i, entry in enumerate(response.data):
        assert entry.index == i
        embedding = _check_token_embeddings(entry, "query")
        assert len(embedding) == model_config["query_maxlen"]

    expected_tokens = model_config["query_maxlen"] * len(queries)
    assert response.usage.prompt_tokens == expected_tokens
    assert response.usage.total_tokens == response.usage.prompt_tokens
