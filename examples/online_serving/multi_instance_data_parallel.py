# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Optional

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

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


async def main():
    engine_args = AsyncEngineArgs(
        model="ibm-research/PowerMoE-3b",
        data_parallel_size=2,
        dtype="auto",
        max_model_len=2048,
        data_parallel_address="127.0.0.1",
        data_parallel_rpc_port=62300,
        data_parallel_size_local=1,
        enforce_eager=True,
    )

    engine_client = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )

    prompt = "Who won the 2004 World Series?"
    final_output: Optional[RequestOutput] = None
    async for output in engine_client.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id="abcdef",
        data_parallel_rank=1,
    ):
        final_output = output
    if final_output:
        print(final_output.outputs[0].text)


if __name__ == "__main__":
    asyncio.run(main())
