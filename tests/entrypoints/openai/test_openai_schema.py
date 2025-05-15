# SPDX-License-Identifier: Apache-2.0
import pytest
import schemathesis
from schemathesis import GenerationConfig

from ...utils import RemoteOpenAIServer

schemathesis.experimental.OPEN_API_3_1.enable()

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAXIMUM_IMAGES = 2


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
        if op.method.lower() != "post" or op.path != "/tokenize":
            return True

        if case.body and isinstance(case.body, dict):
            if "messages" not in case.body:
                return True

            messages = case.body.get("messages", [])
            if not isinstance(messages, list) or len(messages) == 0:
                return True

            for message in messages:
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
def test_openapi_stateless(case: schemathesis.Case):
    #No need to verify SSL certificate for localhost
    case.call_and_validate(verify=False)
