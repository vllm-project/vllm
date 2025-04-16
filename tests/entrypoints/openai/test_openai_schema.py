import pytest
import schemathesis
import pytest
from schemathesis import GenerationConfig
from ...utils import RemoteOpenAIServer
schemathesis.experimental.OPEN_API_3_1.enable()

MODEL_NAME = "ibm-granite/granite-vision-3.2-2b"
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
    return schemathesis.openapi.from_uri(f"{server.url_root}/openapi.json",generation_config=GenerationConfig(allow_x00=False),)

schema = schemathesis.from_pytest_fixture("get_schema")


@schema.parametrize()
async def test_openapi_stateless(case):
    case.headers = {
        "Content-Type": "application/json",
    }
    #disable SSL certificate verification for localhost it doesn't matter 
    await case.call_and_validate(verify=False)
    