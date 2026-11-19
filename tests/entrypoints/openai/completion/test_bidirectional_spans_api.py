# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `bidirectional_spans` request field on the OpenAI-compatible APIs."""

import json
import os
import tempfile

import pytest
import requests
from pydantic import TypeAdapter, ValidationError

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationCompletionRequest,
    ClassificationRequest,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingCompletionRequest,
    EmbeddingRequest,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    PoolingCompletionRequest,
    PoolingRequest,
)
from vllm.exceptions import VLLMValidationError
from vllm.renderers.inputs.preprocess import pair_bidirectional_spans
from vllm.transformers_utils.repo_utils import get_non_weight_snapshot_path

MODEL_NAME = "google/gemma-4-E2B-it"
SPANS = {
    "text_config": {
        "bidirectional_span_layer_types": ["full_attention"],
        "bidirectional_span_max_length": 64,
    }
}
IDS = list(range(100, 140))
CHAT = [{"role": "user", "content": "hi"}]


def test_the_field_holds_spans_for_one_prompt():
    request = CompletionRequest(model="m", prompt=IDS, bidirectional_spans=[[0, 2]])
    assert request.bidirectional_spans == [(0, 2)]


def test_the_field_holds_one_list_of_spans_per_prompt():
    request = CompletionRequest(
        model="m",
        prompt=[IDS[:3], IDS[:4]],
        bidirectional_spans=[[[0, 2]], [[0, 3], [3, 4]]],
    )
    assert request.bidirectional_spans == [[(0, 2)], [(0, 3), (3, 4)]]


@pytest.mark.parametrize("bound", [True, "4", 4.0])
def test_span_bounds_must_be_integers(bound):
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt=IDS, bidirectional_spans=[[0, bound]])


def test_spans_are_paired_with_a_token_id_prompt():
    (single,) = pair_bidirectional_spans([1, 2, 3, 4], [(0, 2)])
    assert single == {"prompt_token_ids": [1, 2, 3, 4], "bidirectional_spans": [(0, 2)]}


def test_each_token_id_prompt_of_a_list_is_paired_with_its_spans():
    batch = pair_bidirectional_spans([[1, 2], [3, 4, 5]], [[(0, 2)], []])
    assert [p["bidirectional_spans"] for p in batch] == [[(0, 2)], []]


@pytest.mark.parametrize(
    ("prompt", "spans"),
    [
        ("some text", [(0, 2)]),
        (["a", "b"], [[(0, 1)], [(0, 1)]]),
        ([[1, 2], [3, 4]], [(0, 2)]),
        ([[1, 2], [3, 4]], [[(0, 2)]]),
    ],
)
def test_spans_without_matching_token_id_prompts_are_refused(prompt, spans):
    with pytest.raises(VLLMValidationError, match="bidirectional_spans"):
        pair_bidirectional_spans(prompt, spans)


@pytest.mark.parametrize(
    ("request_type", "completion_cls"),
    [
        (ClassificationRequest, ClassificationCompletionRequest),
        (PoolingRequest, PoolingCompletionRequest),
        (EmbeddingRequest, EmbeddingCompletionRequest),
    ],
)
def test_pooling_routes_take_spans_on_completion_style_bodies(
    request_type, completion_cls
):
    """The routes validate a union of completion and chat models, so the chat
    refusal must not fire on completion bodies."""
    request = TypeAdapter(request_type).validate_python(
        {"model": "m", "input": IDS, "bidirectional_spans": [[0, 40]]}
    )
    assert isinstance(request, completion_cls)
    assert request.bidirectional_spans == [(0, 40)]


@pytest.mark.parametrize(
    ("request_type", "body"),
    [
        (ClassificationRequest, {"messages": CHAT}),
        (PoolingRequest, {"messages": CHAT}),
        (EmbeddingRequest, {"messages": CHAT}),
        (EmbeddingRequest, {"messages": [CHAT, CHAT]}),
        (EmbeddingRequest, {"input": CHAT}),
    ],
    ids=["classify", "pooling", "embed", "embed-batch", "embed-input"],
)
def test_pooling_routes_refuse_spans_on_chat_bodies(request_type, body):
    with pytest.raises(VLLMValidationError, match="not supported for chat"):
        TypeAdapter(request_type).validate_python(
            {"model": "m", "bidirectional_spans": [[0, 2]], **body}
        )


def test_chat_completions_refuse_spans():
    with pytest.raises(VLLMValidationError, match="not supported for chat"):
        ChatCompletionRequest(model="m", messages=CHAT, bidirectional_spans=[[0, 2]])


def test_null_spans_on_chat_completions_are_unset():
    ChatCompletionRequest(model="m", messages=CHAT, bidirectional_spans=None)


@pytest.fixture(scope="module")
def model_path():
    # The checks below need the config, tokenizer and chat template, not the
    # weights.
    return get_non_weight_snapshot_path(
        MODEL_NAME,
        cache_dir=os.path.join(tempfile.gettempdir(), "gemma4_e2b_config"),
    )


def server_args(*extra: str) -> list[str]:
    return [
        # vllm serve listens on every interface unless told otherwise.
        "--host",
        "127.0.0.1",
        "--load-format",
        "dummy",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "audio": 0}),
        "--hf-overrides",
        json.dumps(SPANS),
        *extra,
    ]


def post(server, route: str, body: dict, **kwargs) -> requests.Response:
    return requests.post(server.url_for(route), json=body, **kwargs)


def assert_refused(response: requests.Response, match: str) -> None:
    assert response.status_code == 400, response.text
    assert match in response.json()["error"]["message"], response.text


class TestCompletions:
    @pytest.fixture(scope="class")
    def server(self, model_path):
        with RemoteOpenAIServer(model_path, server_args()) as remote_server:
            yield remote_server

    def test_spans_are_served(self, server):
        response = post(
            server,
            "v1/completions",
            {
                "prompt": IDS,
                "bidirectional_spans": [[0, 20], [20, 40]],
                "max_tokens": 1,
            },
        )
        assert response.status_code == 200, response.text

    def test_each_prompt_of_a_list_gets_its_spans(self, server):
        response = post(
            server,
            "v1/completions",
            {
                "prompt": [IDS, IDS[:30]],
                "bidirectional_spans": [[[0, 40]], []],
                "max_tokens": 1,
            },
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["choices"]) == 2

    def test_streamed_spans_are_served(self, server):
        response = post(
            server,
            "v1/completions",
            {
                "prompt": IDS,
                "bidirectional_spans": [[0, 40]],
                "max_tokens": 2,
                "stream": True,
            },
            stream=True,
        )
        assert response.status_code == 200, response.text
        events = [line for line in response.iter_lines() if line]
        assert events[-1] == b"data: [DONE]"
        assert not any(b'"error"' in event for event in events)

    def test_streamed_span_errors_arrive_as_an_error_event(self, server):
        """The engine checks the spans; a stream has started by then."""
        response = post(
            server,
            "v1/completions",
            {
                "prompt": IDS,
                "bidirectional_spans": [[0, 41]],
                "max_tokens": 1,
                "stream": True,
            },
            stream=True,
        )
        assert response.status_code == 200, response.text
        events = [line for line in response.iter_lines() if line]
        assert any(b"prompt length" in event for event in events), events

    @pytest.mark.parametrize("stream", [False, True])
    def test_requests_without_spans_are_refused(self, server, stream):
        response = post(
            server,
            "v1/completions",
            {"prompt": IDS, "max_tokens": 1, "stream": stream},
        )
        assert_refused(response, "must set bidirectional_spans")

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({"prompt": "Some text.", "bidirectional_spans": [[0, 2]]}, "token-id"),
            ({"prompt": IDS, "bidirectional_spans": [[0, 41]]}, "prompt length"),
            ({"prompt": [IDS, IDS], "bidirectional_spans": [[0, 2]]}, "one list"),
            (
                {
                    "prompt": IDS,
                    "bidirectional_spans": [[0, 20]],
                    "truncate_prompt_tokens": 30,
                },
                "truncation",
            ),
            (
                {"prompt_embeds": "AAAA", "bidirectional_spans": [[0, 2]]},
                "not supported with prompt_embeds",
            ),
            (
                {
                    "prompt": IDS,
                    "bidirectional_spans": [[0, 20]],
                    "echo": True,
                    "logprobs": 1,
                },
                "prompt logprobs",
            ),
        ],
        ids=[
            "text",
            "out-of-range",
            "one-list-for-many",
            "truncation",
            "prompt-embeds",
            "echo-logprobs",
        ],
    )
    def test_invalid_spans_are_refused(self, server, body, match):
        response = post(server, "v1/completions", {**body, "max_tokens": 1})
        assert_refused(response, match)

    def test_chat_requests_refuse_spans(self, server):
        response = post(
            server,
            "v1/chat/completions",
            {"messages": CHAT, "bidirectional_spans": [[0, 2]]},
        )
        assert_refused(response, "not supported for chat")

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("route", "body"),
        [
            ("v1/chat/completions", {"messages": CHAT}),
            ("v1/responses", {"input": "hi"}),
            ("v1/messages", {"messages": CHAT, "max_tokens": 1}),
        ],
        ids=["chat", "responses", "messages"],
    )
    def test_requests_that_cannot_set_spans_are_refused(
        self, server, model_path, route, body, stream
    ):
        # The Messages API requires the model.
        body = {"model": model_path, **body, "stream": stream}
        response = post(server, route, body)
        assert response.status_code == 400, response.text
        assert "which this request cannot set" in response.text, response.text

    def test_batched_chat_requests_are_refused(self, server):
        response = post(server, "v1/chat/completions/batch", {"messages": [CHAT, CHAT]})
        assert response.status_code == 400, response.text
        assert "which this request cannot set" in response.text, response.text


class TestPoolingRoutes:
    @pytest.fixture(scope="class", params=["classify", "embed"])
    def pooling(self, request, model_path):
        args = server_args("--runner", "pooling", "--convert", request.param)
        routes = {"classify": "classify", "embed": "v1/embeddings"}
        with RemoteOpenAIServer(model_path, args) as remote_server:
            yield remote_server, routes[request.param]

    @pytest.mark.parametrize("route", ["task", "pooling"])
    def test_spans_are_served(self, pooling, route):
        server, task_route = pooling
        route = task_route if route == "task" else route
        response = post(
            server, route, {"input": IDS, "bidirectional_spans": [[0, 20], [20, 40]]}
        )
        assert response.status_code == 200, (route, response.text)

    def test_each_input_of_a_list_gets_its_spans(self, pooling):
        server, route = pooling
        response = post(
            server,
            route,
            {"input": [IDS, IDS[:30]], "bidirectional_spans": [[[0, 40]], []]},
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["data"]) == 2

    def test_requests_without_spans_are_refused(self, pooling):
        server, route = pooling
        assert_refused(
            post(server, route, {"input": IDS}), "must set bidirectional_spans"
        )

    def test_chat_requests_refuse_spans(self, pooling):
        server, route = pooling
        response = post(
            server, route, {"messages": CHAT, "bidirectional_spans": [[0, 2]]}
        )
        assert_refused(response, "not supported for chat")
