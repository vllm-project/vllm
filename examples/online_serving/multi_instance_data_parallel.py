# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.loggers import AggregatedLoggingStatLogger

"""
To run this example, run the following commands simultaneously with
different CUDA_VISIBLE_DEVICES:
    python examples/online_serving/multi_instance_data_parallel.py

    vllm serve ibm-research/PowerMoE-3b -dp 2 -dpr 1 \
        --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 62300 \
        --data-parallel-size-local 1 --enforce-eager --headless

Once both instances have completed the handshake, this example will
send a request to the instance with DP rank 1.
"""


def _do_background_logging(engine, interval, stop_event):
    try:
        while not stop_event.is_set():
            asyncio.run(engine.do_log_stats())
            stop_event.wait(interval)
    except Exception as e:
        print(f"vLLM background logging shutdown: {e}")
        pass


async def main():
    engine_args = AsyncEngineArgs(
        model="ibm-research/PowerMoE-3b",
        data_parallel_size=2,
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        data_parallel_address="127.0.0.1",
        data_parallel_rpc_port=62300,
        data_parallel_size_local=1,
        enforce_eager=True,
        enable_log_requests=True,
        disable_custom_all_reduce=True,
    )

    engine_client = AsyncLLMEngine.from_engine_args(
        engine_args,
        # Example: Using aggregated logger
        stat_loggers=[AggregatedLoggingStatLogger],
    )
    stop_logging_event = threading.Event()
    logging_thread = threading.Thread(
        target=_do_background_logging,
        args=(engine_client, 5, stop_logging_event),
        daemon=True,
    )
    logging_thread.start()
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )
    num_prompts = 10
    for i in range(num_prompts):
        prompt = "Who won the 2004 World Series?"
        final_output: RequestOutput | None = None
        async for output in engine_client.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=f"abcdef-{i}",
            data_parallel_rank=1,
        ):
            final_output = output
        if final_output:
            print(final_output.outputs[0].text)

    stop_logging_event.set()
    logging_thread.join()


if __name__ == "__main__":
    asyncio.run(main())
