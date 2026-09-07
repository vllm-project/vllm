# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.benchmarks.lib.endpoint_request_func import _validate_api_url


@pytest.mark.parametrize(
    ("path", "suffix"),
    [
        ("/v1/completions", "completions"),
        ("/v1/chat/completions", "chat/completions"),
        ("/v1/embeddings", "embeddings"),
        ("/v1/audio/transcriptions", {"transcriptions", "translations"}),
        ("/v1/audio/translations", {"transcriptions", "translations"}),
        ("/rerank", "rerank"),
        ("/pooling", "pooling"),
        ("/start_profile", "completions"),
    ],
)
@pytest.mark.parametrize("query", ["", "?api-version=2026-01-01", "#benchmark"])
def test_validate_api_url_uses_path(path, suffix, query):
    _validate_api_url(f"https://example.com{path}{query}", "Test API", suffix)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1/invalid",
        "https://example.com/v1/invalid?next=completions",
        "https://example.com/v1/invalid#completions",
    ],
)
def test_validate_api_url_rejects_wrong_path(url):
    with pytest.raises(ValueError, match="Test API URL must end"):
        _validate_api_url(url, "Test API", "completions")
