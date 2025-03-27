# SPDX-License-Identifier: Apache-2.0
import argparse
import os

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger("vllm.neuron.multi-node.worker")


def initialize_worker():
    parser = argparse.ArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)
    return args, engine


def start_worker():
    rank_id = int(os.getenv("NEURON_RANK_ID"))
    if rank_id == 0:
        logger.error("Worker must have rank > 0")
    args, engine = initialize_worker()
    worker = engine.engine.model_executor.driver_worker
    while True:
        worker.execute_model()


def main():
    try:
        start_worker()
    except Exception as e:
        logger.error("Failed starting worker %s", e)
        exit(1)


if __name__ == "__main__":
    main()
