# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Final

import pytest
import schemathesis
from hypothesis import HealthCheck, settings
from schemathesis import GenerationConfig
from schemathesis.models import Case

from vllm.platforms import current_platform

from ...utils import RemoteOpenAIServer

schemathesis.experimental.OPEN_API_3_1.enable()

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAXIMUM_IMAGES = 2
_ROCM_TIMEOUT_MULTIPLIER = 3 if current_platform.is_rocm() else 1
DEFAULT_TIMEOUT_SECONDS: Final[int] = 10 * _ROCM_TIMEOUT_MULTIPLIER
LONG_TIMEOUT_SECONDS: Final[int] = 60 * _ROCM_TIMEOUT_MULTIPLIER


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
    return schemathesis.openapi.from_uri(
        f"{server.url_root}/openapi.json",
        generation_config=GenerationConfig(allow_x00=False),
    )


schema = schemathesis.from_pytest_fixture("get_schema")


@schemathesis.hook
def before_generate_case(context: schemathesis.hooks.HookContext, strategy):
    op = context.operation
    assert op is not None

    def no_invalid_types(case: schemathesis.models.Case):
        """
        Skips tool_calls with `"type": "custom"` which schemathesis incorrectly
        generates instead of the valid `"type": "function"`.

        Example test case that is skipped:
        curl -X POST -H 'Content-Type: application/json' \
            -d '{"messages": [{"role": "assistant", "tool_calls": [{"custom": {"input": "", "name": ""}, "id": "", "type": "custom"}]}]}' \
            http://localhost:8000/v1/chat/completions
        """  # noqa: E501
        if hasattr(case, "body") and isinstance(case.body, dict):
            if (
                "messages" in case.body
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

            # Sometimes structured_outputs.grammar is generated to be empty
            # Causing a server error in EBNF grammar parsing
            # https://github.com/vllm-project/vllm/pull/22587#issuecomment-3195253421
            structured_outputs = case.body.get("structured_outputs", {})
            grammar = (
                structured_outputs.get("grammar")
                if isinstance(structured_outputs, dict)
                else None
            )

            if grammar == "":
                # Allow None (will be handled as no grammar)
                # But skip empty strings
                return False

        return True

    return strategy.filter(no_invalid_types)


@schema.parametrize()
@schema.override(headers={"Content-Type": "application/json"})
@settings(
    deadline=LONG_TIMEOUT_SECONDS * 1000,
    max_examples=50,
    # Under CI's derandomized hypothesis seed, the schemathesis strategy
    # for /v1/chat/completions/batch's nested-message body, combined with
    # the no_invalid_types filter (notably the grammar=="" rule), exceeds
    # the default filtered-vs-good ratio. The filter is intentional, so
    # suppress the health check rather than drop the filter — dropping it
    # exposes pre-existing server bugs out of scope here.
    suppress_health_check=[HealthCheck.filter_too_much],
)
def test_openapi_stateless(case: Case):
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
        "/update_weights",
        "/finish_weight_update",
    ):
        return

    timeout = {
        # requires a longer timeout
        ("POST", "/v1/chat/completions"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/chat/completions/batch"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/completions"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/messages"): LONG_TIMEOUT_SECONDS,
    }.get(key, DEFAULT_TIMEOUT_SECONDS)

    # No need to verify SSL certificate for localhost
    case.call_and_validate(verify=False, timeout=timeout)
