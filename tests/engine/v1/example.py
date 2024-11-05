from multiprocessing import freeze_support

from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import EngineCoreClient

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

if __name__ == "__main__":
    freeze_support()

    engine_args = EngineArgs(model=MODEL_NAME)
    vllm_config = engine_args.create_engine_config()
    executor_class = AsyncLLM._get_executor_cls(vllm_config)
    client = EngineCoreClient.make_client(
        vllm_config,
        executor_class,
        UsageContext.UNKNOWN_CONTEXT,
        multiprocess_mode=True,
        asyncio_mode=False,
    )

    print(client)
    del client
