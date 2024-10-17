"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
from vllm.entrypoints import api_server
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger("vllm.entrypoints.neuron_multi_node.api_server")


async def run_driver(args, engine):
    await api_server.run_server(args, engine)


async def initialize_worker():
    args = api_server.parse_args()

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    await api_server.init_app(args)
    assert api_server.engine is not None
    return args, api_server.engine
