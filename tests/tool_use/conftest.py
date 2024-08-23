import pytest
from huggingface_hub import snapshot_download

from tests.utils import RemoteOpenAIServer

from .utils import ARGS, CONFIGS, ServerConfig


# for each server config, download the model and return the config
@pytest.fixture(scope="session", params=CONFIGS.keys())
def server_config(request):
    config = CONFIGS[request.param]
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


@pytest.fixture(scope="session")
def client(server: RemoteOpenAIServer):
    return server.get_async_client()
