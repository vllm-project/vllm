# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
        "--task",
        "generate",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        f"image={MAXIMUM_IMAGES}",
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

    def no_file_type(case: schemathesis.models.Case):
        """
        This filter skips test cases for the `POST /tokenize` endpoint where the
        HTTP request body uses `"type": "file"` in any message's content.
        We expect these cases to fail because that type isn't implemented here
        https://github.com/vllm-project/vllm/blob/0b34593017953051b3225b1483ce0f4670e3eb0e/vllm/entrypoints/chat_utils.py#L1038-L1095

        Example test cases that are skipped:
        curl -X POST -H 'Content-Type: application/json' \
            -d '{"messages": [{"role": "assistant"}, {"content": [{"file": {}, "type": "file"}], "role": "user"}]}' \
            http://localhost:8000/tokenize

        curl -X POST -H 'Content-Type: application/json' \
            -d '{"messages": [{"content": [{"file": {}, "type": "file"}], "role": "user"}]}' \
            http://localhost:8000/tokenize
        """  # noqa: E501
        if (op.method.lower() == "post" and op.path == "/tokenize"
                and hasattr(case, "body") and isinstance(case.body, dict)
                and "messages" in case.body
                and isinstance(case.body["messages"], list)
                and len(case.body["messages"]) > 0):
            for message in case.body["messages"]:
                if not isinstance(message, dict):
                    continue
                content = message.get("content", [])
                if not isinstance(content, list) or len(content) == 0:
                    continue
                if any(item.get("type") == "file" for item in content):
                    return False
        return True

    return strategy.filter(no_file_type)


@schema.parametrize()
@schema.override(headers={"Content-Type": "application/json"})
@settings(deadline=LONG_TIMEOUT_SECONDS * 1000)
def test_openapi_stateless(case: schemathesis.Case):
    key = (
        case.operation.method.upper(),
        case.operation.path,
    )
    timeout = {
        # requires a longer timeout
        ("POST", "/v1/chat/completions"):
        LONG_TIMEOUT_SECONDS,
    }.get(key, DEFAULT_TIMEOUT_SECONDS)

    #No need to verify SSL certificate for localhost
    case.call_and_validate(verify=False, timeout=timeout)
