import pytest
import pytest_asyncio
from huggingface_hub import snapshot_download

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from .utils import ARGS, CONFIGS, ServerConfig


# for each server config, download the model and return the config
@pytest.fixture(scope="session", params=CONFIGS.keys())
def server_config(request):
    config = CONFIGS[request.param]

    if current_platform.is_rocm() and not config.get("supports_rocm", True):
        pytest.skip("The {} model can't be tested on the ROCm platform".format(
            config["model"]))

    # download model and tokenizer using transformers
    snapshot_download(config["model"])
    yield CONFIGS[request.param]


# run this for each server config
@pytest.fixture(scope="session")
def server(request, server_config: ServerConfig):
    model = server_config["model"]
    args_for_model = server_config["arguments"]
    with RemoteOpenAIServer(model, ARGS + args_for_model,
                            max_wait_seconds=480) as server:
        yield server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    async with server.get_async_client() as async_client:
        yield async_client
