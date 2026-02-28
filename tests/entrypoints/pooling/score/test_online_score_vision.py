# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import requests

from tests.utils import VLLM_PATH, RemoteOpenAIServer
from vllm.entrypoints.pooling.score.protocol import RerankResponse, ScoreResponse
from vllm.multimodal.utils import encode_image_url, fetch_image
from vllm.platforms import current_platform

MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"
HF_OVERRIDES = {
    "architectures": ["Qwen3VLForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}

ROCM_ATTN_BACKENDS = [
    "ROCM_ATTN",
    "ROCM_AITER_FA",
    "TRITON_ATTN",
    "FLEX_ATTENTION",
]

ATTN_BACKENDS = ROCM_ATTN_BACKENDS if current_platform.is_rocm() else []

# Per-backend tolerance with explicit entries; "default" is the fallback
BACKEND_TOL: dict[str, float] = {
    "default": 0.05,  # 5% tolerance for other backends (e.g. FLASH_ATTN)
    # Relaxed tolerances for ROCm attn
    # See: https://github.com/vllm-project/vllm/issues/35569
    "ROCM_ATTN": 0.09,  # observed max: 8.45%
    "ROCM_AITER_FA": 0.045,  # observed max: 5.07%
    "TRITON_ATTN": 0.045,  # observed max: 4.59%
    "FLEX_ATTENTION": 0.045,  # observed max: 7.39%
}

# ROCm: disable skinny GEMM to avoid non-deterministic results from
# atomic reductions in wvSplitKrc kernel.
# See: https://github.com/vllm-project/vllm/pull/33493#issuecomment-3906083975
ROCM_ENV_OVERRIDES = (
    {"VLLM_ROCM_USE_SKINNY_GEMM": "0"} if current_platform.is_rocm() else {}
)
# ROCm: disable prefix caching and eliminate batch variance to reduce
# test flakiness.
ROCM_EXTRA_ARGS = (
    ["--no-enable-prefix-caching", "--max-num-seqs", "1"]
    if current_platform.is_rocm()
    else []
)


def get_tol(backend: str) -> float:
    return BACKEND_TOL.get(backend, BACKEND_TOL["default"])


def assert_score(actual: float, expected: float, backend: str, label: str):
    tol = get_tol(backend)
    diff = abs(actual - expected)
    rel_diff = diff / abs(expected) if expected != 0 else diff
    print(
        f"[{backend}] {label}: actual={actual:.6f} expected={expected:.6f} "
        f"diff={diff:.6f} rel_diff={rel_diff:.4f} tol={tol}"
    )
    assert actual == pytest.approx(expected, rel=tol), (
        f"[{backend}] {label}: score mismatch — "
        f"actual={actual:.6f}, expected={expected:.6f}, "
        f"rel_diff={rel_diff:.4f}, tol={tol}"
    )


query = "A cat standing in the snow."
document = "This product was excellent and exceeded my expectations."
image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
documents = [
    {
        "type": "text",
        "text": document,
    },
    {
        "type": "image_url",
        "image_url": {"url": image_url},
    },
    {
        "type": "image_url",
        "image_url": {"url": encode_image_url(fetch_image(image_url))},
    },
]

TEXT_VS_TEXT = 0.10040374100208282
TEXT_VS_IMAGE = 0.7423753142356873
TEXT_VS_TEXT_PLUS_IMAGE = 0.5298863053321838


@pytest.fixture(scope="module", params=ATTN_BACKENDS)
def server(request):
    backend = request.param
    print(f"\n=== Starting server with attention backend: {backend} ===")
    args = [
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--chat-template",
        str(VLLM_PATH / "examples/pooling/score/template/qwen3_vl_reranker.jinja"),
        "--attention-config",
        json.dumps({"backend": backend}),
    ] + ROCM_EXTRA_ARGS

    with RemoteOpenAIServer(
        MODEL_NAME, args, override_hf_configs=HF_OVERRIDES, env_dict=ROCM_ENV_OVERRIDES
    ) as remote_server:
        remote_server.attn_backend = backend
        print(f"=== Server ready with backend: {backend} ===")
        yield remote_server


def test_score_api_queries_str_documents_str(server: RemoteOpenAIServer):
    backend = server.attn_backend
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": document,
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1
    assert score.usage.prompt_tokens == 81
    assert_score(score.data[0].score, TEXT_VS_TEXT, backend, "text_vs_text")


def test_score_api_queries_str_documents_text_content(server: RemoteOpenAIServer):
    backend = server.attn_backend
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": {"content": [documents[0]]},
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1
    assert score.usage.prompt_tokens == 81
    assert_score(score.data[0].score, TEXT_VS_TEXT, backend, "text_vs_text")


def test_score_api_queries_str_documents_image_url_content(server: RemoteOpenAIServer):
    backend = server.attn_backend
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": {"content": [documents[1]]},
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1
    assert score.usage.prompt_tokens == 98
    assert_score(score.data[0].score, TEXT_VS_IMAGE, backend, "text_vs_image")


def test_score_api_queries_str_documents_image_base64_content(
    server: RemoteOpenAIServer,
):
    backend = server.attn_backend
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": {"content": [documents[2]]},
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1
    assert score.usage.prompt_tokens == 98
    assert_score(score.data[0].score, TEXT_VS_IMAGE, backend, "text_vs_image_base64")


def test_score_api_queries_str_documents_image_url_plus_text_content(
    server: RemoteOpenAIServer,
):
    backend = server.attn_backend
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": {"content": [documents[0], documents[1]]},
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1
    assert score.usage.prompt_tokens == 108
    assert_score(
        score.data[0].score, TEXT_VS_TEXT_PLUS_IMAGE, backend, "text_vs_text_plus_image"
    )


def test_score_api_queries_str_documents_list(server: RemoteOpenAIServer):
    backend = server.attn_backend
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": [
                document,
                {"content": [documents[0]]},
                {"content": [documents[1]]},
                {"content": [documents[0], documents[1]]},
            ],
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 4
    assert score.usage.prompt_tokens == 368
    assert_score(score.data[0].score, TEXT_VS_TEXT, backend, "list[0]_text_vs_text")
    assert_score(score.data[1].score, TEXT_VS_TEXT, backend, "list[1]_text_vs_text")
    assert_score(score.data[2].score, TEXT_VS_IMAGE, backend, "list[2]_text_vs_image")
    assert_score(
        score.data[3].score,
        TEXT_VS_TEXT_PLUS_IMAGE,
        backend,
        "list[3]_text_vs_text_plus_image",
    )


def test_rerank_api_queries_str_documents_list(server: RemoteOpenAIServer):
    backend = server.attn_backend
    rerank_response = requests.post(
        server.url_for("rerank"),
        json={
            "model": MODEL_NAME,
            "query": query,
            "documents": [
                document,
                {"content": [documents[0]]},
                {"content": [documents[1]]},
                {"content": [documents[0], documents[1]]},
            ],
        },
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.model is not None
    assert rerank.usage is not None
    assert len(rerank.results) == 4

    rerank.results.sort(key=lambda x: x.index)
    assert_score(
        rerank.results[0].relevance_score,
        TEXT_VS_TEXT,
        backend,
        "rerank[0]_text_vs_text",
    )
    assert_score(
        rerank.results[1].relevance_score,
        TEXT_VS_TEXT,
        backend,
        "rerank[1]_text_vs_text",
    )
    assert_score(
        rerank.results[2].relevance_score,
        TEXT_VS_IMAGE,
        backend,
        "rerank[2]_text_vs_image",
    )
    assert_score(
        rerank.results[3].relevance_score,
        TEXT_VS_TEXT_PLUS_IMAGE,
        backend,
        "rerank[3]_text_vs_text_plus_image",
    )


def test_score_api_queries_list_documents_list(server: RemoteOpenAIServer):
    backend = server.attn_backend
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": [query] * 4,
            "documents": [
                document,
                {"content": [documents[0]]},
                {"content": [documents[1]]},
                {"content": [documents[0], documents[1]]},
            ],
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 4
    assert score.usage.prompt_tokens == 368
    assert_score(score.data[0].score, TEXT_VS_TEXT, backend, "paired[0]_text_vs_text")
    assert_score(score.data[1].score, TEXT_VS_TEXT, backend, "paired[1]_text_vs_text")
    assert_score(score.data[2].score, TEXT_VS_IMAGE, backend, "paired[2]_text_vs_image")
    assert_score(
        score.data[3].score,
        TEXT_VS_TEXT_PLUS_IMAGE,
        backend,
        "paired[3]_text_vs_text_plus_image",
    )
