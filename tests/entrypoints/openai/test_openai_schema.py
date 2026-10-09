# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Final

import pytest
import schemathesis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from jsonschema import Draft202012Validator
from schemathesis import GenerationMode
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


@pytest.mark.parametrize("endpoint", ["chat_completion", "completion"])
def test_openapi_response_payloads(endpoint: str) -> None:
    """Each completion endpoint must export distinct JSON and SSE payload schemas.

    Check refs against serialized models and reject malformed choices so missing,
    swapped, or unconstrained schemas fail without starting an inference engine.
    """
    if endpoint == "chat_completion":
        from vllm.entrypoints.openai.chat_completion.api_router import router
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
            ChatCompletionStreamResponse,
        )

        path = "/v1/chat/completions"
        models = (ChatCompletionResponse, ChatCompletionStreamResponse)
        choices = (
            {"index": 0, "message": {"role": "assistant", "content": "hello"}},
            {"index": 0, "delta": {"content": "hello"}},
        )
    else:
        from vllm.entrypoints.openai.completion.api_router import router
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionResponse,
            CompletionStreamResponse,
        )

        path = "/v1/completions"
        models = (CompletionResponse, CompletionStreamResponse)
        choices = ({"index": 0, "text": "hello"},) * 2

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        document = client.get("/openapi.json").json()
    operation = document["paths"][path]["post"]
    content = operation["responses"]["200"]["content"]
    for media, model, choice in zip(
        ("application/json", "text/event-stream"), models, choices
    ):
        schema = content[media]["schema"]
        assert schema["$ref"] == f"#/components/schemas/{model.__name__}"
        validator = Draft202012Validator(
            {**schema, "components": document["components"]}
        )
        payload = model(model="test", choices=[choice], usage={}).model_dump(
            mode="json"
        )
        validator.validate(payload)
        payload["choices"] = "not an array"
        assert not validator.is_valid(payload)


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
        "--trust-remote-code",
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
    case.call_and_validate(
        verify=False,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
