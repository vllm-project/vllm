# SPDX-License-Identifier: Apache-2.0
from typing import Final

import pytest
import schemathesis
from schemathesis.generation import GenerationConfig

from ...utils import RemoteOpenAIServer

schemathesis.experimental.OPEN_API_3_1.enable()

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAXIMUM_IMAGES = 2
DEFAULT_TIMEOUT: Final[int] = 10  # in seconds


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
    return schemathesis.specs.openapi.loaders.from_uri(
        f"{server.url_root}/openapi.json",
        generation_config=GenerationConfig(allow_x00=False),
    )


schema = schemathesis.from_pytest_fixture("get_schema")


@schema.parametrize()
@schema.override(headers={"Content-Type": "application/json"})
async def test_openapi_stateless(case):
    key = (
        case.operation.method.upper(),
        case.operation.path,
    )
    timeout = {
        ("POST", "/v1/chat/completions"): 30,
    }.get(key, DEFAULT_TIMEOUT)

    # No need to verify SSL certificate for localhost
    case.call_and_validate(verify=False, timeout=timeout)
