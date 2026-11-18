# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from argparse import Namespace
from typing import Final

import pytest
import schemathesis
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from jsonschema import Draft202012Validator
from schemathesis import GenerationMode
from schemathesis.checks import not_a_server_error
from schemathesis.config import (
    ChecksConfig,
    CoveragePhaseConfig,
    GenerationConfig,
    PhasesConfig,
    PositiveDataAcceptanceConfig,
    ProjectConfig,
    ProjectsConfig,
    SchemathesisConfig,
)

from vllm.platforms import current_platform

from ...utils import RemoteOpenAIServer

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAXIMUM_IMAGES = 2
_ROCM_TIMEOUT_MULTIPLIER = 3 if current_platform.is_rocm() else 1
DEFAULT_TIMEOUT_SECONDS: Final[int] = 10 * _ROCM_TIMEOUT_MULTIPLIER
LONG_TIMEOUT_SECONDS: Final[int] = 60 * _ROCM_TIMEOUT_MULTIPLIER


def _get_openapi_document() -> dict:
    from vllm.entrypoints.launchers.app import build_app

    args = Namespace(
        disable_fastapi_docs=False,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        enable_fault_tolerance=False,
        middleware=[],
        log_error_stack=False,
    )
    # No lifespan: schema generation does not need an inference engine.
    client = TestClient(build_app(args, supported_tasks=("generate",)))
    return client.get("/openapi.json").json()


def test_openapi_exposes_completion_response_schemas() -> None:
    """The published 200 contract of both completion APIs must reference the
    real response models, for the JSON body and for each SSE data chunk.

    Uses the production app: FastAPI renames models used as both input and
    output (-Input/-Output), which a router-only app cannot show.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionResponse,
        ChatCompletionStreamResponse,
    )
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionResponse,
        CompletionStreamResponse,
    )

    document = _get_openapi_document()
    message = {"role": "assistant", "content": "hi"}
    for path, unary, unary_choice, chunk, chunk_choice in (
        (
            "/v1/chat/completions",
            ChatCompletionResponse,
            {"index": 0, "message": message},
            ChatCompletionStreamResponse,
            {"index": 0, "delta": {"content": "hi"}},
        ),
        (
            "/v1/completions",
            CompletionResponse,
            {"index": 0, "text": "hi"},
            CompletionStreamResponse,
            {"index": 0, "text": "hi"},
        ),
    ):
        content = document["paths"][path]["post"]["responses"]["200"]["content"]
        data = content["text/event-stream"]["schema"]["properties"]["data"]
        assert data["anyOf"][0] == {"const": "[DONE]"}
        for schema, model, choice in (
            (content["application/json"]["schema"], unary, unary_choice),
            (data["anyOf"][1]["contentSchema"], chunk, chunk_choice),
        ):
            assert schema == {"$ref": f"#/components/schemas/{model.__name__}"}
            validator = Draft202012Validator(
                {**schema, "components": document["components"]}
            )
            payload = model(model="m", choices=[choice], usage={}).model_dump(
                mode="json"
            )
            validator.validate(payload)
            payload["choices"] = "not an array"
            assert not validator.is_valid(payload)


def test_openapi_exposes_responses_response_schemas() -> None:
    """The published 200 contract of /v1/responses* must expose ResponsesResponse
    for JSON and StreamingResponsesResponse without '[DONE]' for SSE."""
    from openai.types.responses import (
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseFunctionCallArgumentsDoneEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )

    from vllm.entrypoints.openai.responses.protocol import (
        ResponseCompletedEvent,
        ResponseCreatedEvent,
        ResponsesResponse,
    )

    document = _get_openapi_document()
    unary_ref = {"$ref": f"#/components/schemas/{ResponsesResponse.__name__}"}

    cancel_content = document["paths"]["/v1/responses/{response_id}/cancel"]["post"][
        "responses"
    ]["200"]["content"]
    assert cancel_content["application/json"]["schema"] == unary_ref

    sample_response = ResponsesResponse(
        model="m",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                content=[
                    ResponseOutputText(type="output_text", text="hi", annotations=[])
                ],
                status="completed",
            )
        ],
        parallel_tool_calls=True,
        temperature=0.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        max_output_tokens=16,
        service_tier="auto",
        status="completed",
        truncation="disabled",
    )

    for path, method in (
        ("/v1/responses", "post"),
        ("/v1/responses/{response_id}", "get"),
    ):
        content = document["paths"][path][method]["responses"]["200"]["content"]
        assert content["application/json"]["schema"] == unary_ref

        unary_validator = Draft202012Validator(
            {
                **content["application/json"]["schema"],
                "components": document["components"],
            }
        )
        unary_payload = sample_response.model_dump(mode="json", by_alias=True)
        unary_validator.validate(unary_payload)
        assert not unary_validator.is_valid({**unary_payload, "output": "bad"})

        data = content["text/event-stream"]["schema"]["properties"]["data"]
        assert all("const" not in item for item in data["anyOf"])
        assert data["anyOf"][1]["contentSchema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }

        chunk_validator = Draft202012Validator(
            {**data["anyOf"][0]["contentSchema"], "components": document["components"]}
        )
        for event in (
            ResponseCreatedEvent(
                type="response.created",
                sequence_number=0,
                response=sample_response,
            ),
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=1,
                content_index=0,
                output_index=0,
                item_id="msg_1",
                delta="hi",
                logprobs=[],
            ),
            ResponseTextDoneEvent(
                type="response.output_text.done",
                sequence_number=2,
                content_index=0,
                output_index=0,
                item_id="msg_1",
                text="hi",
                logprobs=[],
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=3,
                output_index=0,
                item_id="fc_1",
                delta="{}",
            ),
            ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                sequence_number=4,
                output_index=0,
                item_id="fc_1",
                arguments="{}",
                name="fn",
            ),
            ResponseCompletedEvent(
                type="response.completed",
                sequence_number=5,
                response=sample_response,
            ),
        ):
            chunk_validator.validate(event.model_dump(mode="json", by_alias=True))
        assert not chunk_validator.is_valid({"type": "response.unknown"})


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "generate",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": MAXIMUM_IMAGES}),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def get_schema(server):
    # avoid generating null (\x00) bytes in strings during test case generation
    return schemathesis.openapi.from_url(
        f"{server.url_root}/openapi.json",
        config=SchemathesisConfig(
            projects=ProjectsConfig(
                default=ProjectConfig(
                    generation=GenerationConfig(
                        allow_x00=False,
                        modes=[GenerationMode.POSITIVE],
                    ),
                    checks=ChecksConfig(
                        positive_data_acceptance=PositiveDataAcceptanceConfig(
                            enabled=False,
                        ),
                    ),
                    phases=PhasesConfig(
                        coverage=CoveragePhaseConfig(enabled=False),
                    ),
                ),
            ),
        ),
    )


schema = schemathesis.pytest.from_fixture("get_schema")


@schemathesis.hook
def before_generate_case(context: schemathesis.HookContext, strategy):
    op = context.operation
    assert op is not None

    def no_invalid_types(case: schemathesis.Case):
        """Skips tool_calls with `"type": "custom"` which schemathesis incorrectly
        generates instead of the valid `"type": "function"`.

        Example test case that is skipped:
        curl -X POST -H 'Content-Type: application/json' \
            -d '{"messages": [{"role": "assistant", "tool_calls": [{"custom": {"input": "", "name": ""}, "id": "", "type": "custom"}]}]}' \
            http://localhost:8000/v1/chat/completions
        """  # noqa: E501
        if (
            hasattr(case, "body")
            and isinstance(case.body, dict)
            and "messages" in case.body
            and isinstance(case.body["messages"], list)
            and len(case.body["messages"]) > 0
        ):
            for message in case.body["messages"]:
                if not isinstance(message, dict):
                    continue

                tool_calls = message.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            if tool_call.get("type") != "function":
                                return False
                            if "custom" in tool_call:
                                return False

        return True

    return strategy.filter(no_invalid_types)


@schema.parametrize()
@settings(
    deadline=LONG_TIMEOUT_SECONDS * 1000,
    max_examples=50,
    # Under CI's derandomized hypothesis seed, the schemathesis strategy
    # for /v1/chat/completions/batch's nested-message body, combined with
    # the no_invalid_types filter (notably the grammar=="" rule), exceeds
    # the default filtered-vs-good ratio. The filter is intentional, so
    # suppress the health check rather than drop the filter — dropping it
    # exposes pre-existing server bugs out of scope here.
    # The same nested schema can also trip Hypothesis' entropy budget while
    # generating large-but-valid request bodies before vLLM is called.
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
)
def test_openapi_stateless(case: schemathesis.Case):
    key = (
        case.operation.method.upper(),
        case.operation.path,
    )
    if case.operation.path.startswith("/v1/responses"):
        # Skip responses API as it is meant to be stateful.
        return

    # Skip weight transfer endpoints as they require special setup
    # (weight_transfer_config) and are meant to be stateful.
    if case.operation.path in (
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/start_draft_weight_update",
        "/update_weights",
        "/finish_weight_update",
        "/update_weight_version",
    ):
        return

    timeout = {
        # requires a longer timeout
        ("POST", "/v1/chat/completions"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/chat/completions/batch"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/completions"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/messages"): LONG_TIMEOUT_SECONDS,
        ("POST", "/inference/v1/generate"): LONG_TIMEOUT_SECONDS,
    }.get(key, DEFAULT_TIMEOUT_SECONDS)

    # No need to verify SSL certificate for localhost
    unsupported_decisions = case.operation.path in {
        "/v1/systemone",
        "/v1/decisions",
    }
    response = case.call_and_validate(
        verify=False,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
        excluded_checks=[not_a_server_error] if unsupported_decisions else None,
    )
    if unsupported_decisions:
        # SmolVLM does not support either decision endpoint.
        assert response.status_code < 500 or response.status_code == 501
