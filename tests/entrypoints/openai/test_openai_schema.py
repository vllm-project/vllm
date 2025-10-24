# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Final

import pytest
import schemathesis
from hypothesis import settings
from schemathesis import GenerationConfig

from ...utils import RemoteOpenAIServer

schemathesis.experimental.OPEN_API_3_1.enable()

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAXIMUM_IMAGES = 2
DEFAULT_TIMEOUT_SECONDS: Final[int] = 10
LONG_TIMEOUT_SECONDS: Final[int] = 60


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
        This filter skips test cases with invalid data that schemathesis
        incorrectly generates due to permissive schema configurations.
        
        1. Skips `POST /tokenize` endpoint cases with `"type": "file"` in 
           message content, which isn't implemented.
        
        2. Skips tool_calls with `"type": "custom"` which schemathesis 
           incorrectly generates instead of the valid `"type": "function"`.

        Example test cases that are skipped:
        curl -X POST -H 'Content-Type: application/json' \
            -d '{"messages": [{"content": [{"file": {}, "type": "file"}], "role": "user"}]}' \
            http://localhost:8000/tokenize

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

                    # Check for invalid file type in tokenize endpoint
                    if op.method.lower() == "post" and op.path == "/tokenize":
                        content = message.get("content", [])
                        if (
                            isinstance(content, list)
                            and len(content) > 0
                            and any(item.get("type") == "file" for item in content)
                        ):
                            return False

                    # Check for invalid tool_calls with non-function types
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
@settings(deadline=LONG_TIMEOUT_SECONDS * 1000)
def test_openapi_stateless(case: schemathesis.Case):
    key = (
        case.operation.method.upper(),
        case.operation.path,
    )
    if case.operation.path.startswith("/v1/responses"):
        # Skip responses API as it is meant to be stateful.
        return

    timeout = {
        # requires a longer timeout
        ("POST", "/v1/chat/completions"): LONG_TIMEOUT_SECONDS,
    }.get(key, DEFAULT_TIMEOUT_SECONDS)

    # No need to verify SSL certificate for localhost
    case.call_and_validate(verify=False, timeout=timeout)
