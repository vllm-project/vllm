import pytest
from huggingface_hub import snapshot_download

from tests.utils import RemoteOpenAIServer
from .util import ARGS, CONFIGS, ServerConfig


# for each server config, download the model and return the config
@pytest.fixture(scope="module", params=CONFIGS.keys())
def server_config(request):
    config = CONFIGS[request.param]
    # download model and tokenizer using transformers
    snapshot_download(config["model"])
    yield CONFIGS[request.param]


# run this for each server config
@pytest.fixture(scope="module")
def server(request, server_config: ServerConfig):
    model = server_config["model"]
    args_for_model = server_config["arguments"]
    with RemoteOpenAIServer(model, ARGS + args_for_model,
                            max_start_wait_s=240) as server:
        yield server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_async_client()
