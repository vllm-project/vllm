# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import pytest_asyncio
from huggingface_hub import snapshot_download

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from .utils import ARGS, CONFIGS, ServerConfig


# select models to test based on command line arguments
def pytest_addoption(parser):
    parser.addoption("--models",
                     nargs="+",
                     help="Specify one or more models to test")
    parser.addoption("--extended",
                     action="store_true",
                     default=False,
                     help="invoke extended tests requiring large GPUs")


# for each server config, download the model and return the config
@pytest.fixture(scope="session", params=CONFIGS.keys())
def server_config(request):
    extended = request.config.getoption("--extended")
    models = request.config.getoption("--models")

    config_keys_to_test = [
        key for key in CONFIGS if (models is None or key in models) and (
            extended or not CONFIGS[key].get("extended", False))
    ]

    config_key = request.param
    if config_key not in config_keys_to_test:
        pytest.skip(f"Skipping config '{config_key}'")

    config = CONFIGS[config_key]

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
