# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

from vllm.entrypoints.openai.protocol import (RerankDocumentObject,
                                              RerankImageURL, RerankResponse)

from ...utils import RemoteOpenAIServer

MODEL_NAME = "BAAI/bge-reranker-base"
DTYPE = "bfloat16"


# Unit tests for RerankDocumentObject
class TestRerankDocumentObject:

    def test_text_only(self):
        doc = RerankDocumentObject(text="Hello world")
        assert doc.text == "Hello world"
        assert doc.image_url is None
        assert doc.get_image_url_str() is None

    def test_image_url_string(self):
        doc = RerankDocumentObject(
            image_url="https://example.com/image.jpg")
        assert doc.text is None
        assert doc.get_image_url_str() == "https://example.com/image.jpg"

    def test_image_url_object(self):
        doc = RerankDocumentObject(
            image_url=RerankImageURL(url="https://example.com/image.jpg"))
        assert doc.get_image_url_str() == "https://example.com/image.jpg"

    def test_local_path_conversion(self):
        doc = RerankDocumentObject(image_url="/path/to/image.jpg")
        assert doc.get_image_url_str() == "file:///path/to/image.jpg"

    def test_file_url_passthrough(self):
        doc = RerankDocumentObject(image_url="file:///path/to/image.jpg")
        assert doc.get_image_url_str() == "file:///path/to/image.jpg"

    def test_data_url_passthrough(self):
        data_url = "data:image/jpeg;base64,/9j/4AAQ..."
        doc = RerankDocumentObject(image_url=data_url)
        assert doc.get_image_url_str() == data_url

    def test_text_and_image(self):
        doc = RerankDocumentObject(
            text="A description",
            image_url="https://example.com/image.jpg")
        assert doc.text == "A description"
        assert doc.get_image_url_str() == "https://example.com/image.jpg"

    def test_validation_error_empty(self):
        with pytest.raises(ValueError,
                           match="At least one of 'text' or 'image_url'"):
            RerankDocumentObject()


@pytest.fixture(scope="module")
def server():
    args = ["--enforce-eager", "--max-model-len", "100", "--dtype", DTYPE]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_texts(server: RemoteOpenAIServer, model_name: str):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].relevance_score >= 0.9
    assert rerank.results[1].relevance_score <= 0.01


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_top_n(server: RemoteOpenAIServer, model_name: str):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.", "Cross-encoder models are neat"
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                        "top_n": 2
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].relevance_score >= 0.9
    assert rerank.results[1].relevance_score <= 0.01


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_max_model_len(server: RemoteOpenAIServer, model_name: str):

    query = "What is the capital of France?" * 100
    documents = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents
                                    })
    assert rerank_response.status_code == 400
    # Assert just a small fragments of the response
    assert "Please reduce the length of the input." in \
        rerank_response.text


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_document_objects(server: RemoteOpenAIServer, model_name: str):
    """Test reranking with document objects instead of plain strings."""
    query = "What is the capital of France?"
    # Use document objects with text field
    documents = [
        {"text": "The capital of Brazil is Brasilia."},
        {"text": "The capital of France is Paris."},
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2
    # The Paris document should be ranked higher
    assert rerank.results[0].relevance_score >= 0.9
    assert rerank.results[0].document.text == "The capital of France is Paris."


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_mixed_documents(server: RemoteOpenAIServer, model_name: str):
    """Test reranking with a mix of string and document object formats."""
    query = "What is the capital of France?"
    # Mix of plain strings and document objects
    documents = [
        "The capital of Brazil is Brasilia.",  # plain string
        {"text": "The capital of France is Paris."},  # document object
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].relevance_score >= 0.9


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_query_as_object(server: RemoteOpenAIServer, model_name: str):
    """Test reranking with query as a document object."""
    query = {"text": "What is the capital of France?"}
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].relevance_score >= 0.9


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_image_requires_multimodal_model(server: RemoteOpenAIServer,
                                                 model_name: str):
    """Test that image documents require a multimodal model."""
    query = "What does this image show?"
    documents = [
        {"image_url": "https://example.com/image1.jpg"},
        {"image_url": "https://example.com/image2.jpg"},
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                    })
    # Should fail because the model is not multimodal
    assert rerank_response.status_code == 400
    assert "multimodal" in rerank_response.text.lower() or \
           "vision" in rerank_response.text.lower()
