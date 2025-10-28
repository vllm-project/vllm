import json
import random
import os
import time
import asyncio
import pandas as pd

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.core.sched.ewsjf_scheduler.scheduler_cls import SCHEDULER_CLS


os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


async def generate_async(engine, request_id, prompt, sampling_params):
    results_generator = engine.generate(prompt, sampling_params, request_id)
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def send_requests_with_rate_limit(engine, prompts, sampling_params, requests_per_second=15):
    """Send requests at a controlled rate (requests per second)"""
    tasks = []
    interval = 1.0 / requests_per_second  # Time interval between requests (in seconds)

    for i, prompt in enumerate(prompts):
        request_id = f"request_{i}"

        # Create task
        task = asyncio.create_task(
            generate_async(engine, request_id, prompt, sampling_params)
        )
        tasks.append(task)

        # Wait between requests (except for the last one)
        if i < len(prompts) - 1:
            await asyncio.sleep(interval)

    return tasks


async def main_ewsjf(queues_config):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100, min_tokens=1)

    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'data_30000_100_2000.csv')
    dataset = pd.read_csv(csv_path)
    prompts = dataset['input'].tolist()

    rates = [1000, 500, 100, 60, 40, 20, 10]

    external_parameters = {"queues_config": queues_config, "step_size": 1500}
    engine_args = AsyncEngineArgs(
        model="meta-llama/Meta-Llama-3-8B",
        scheduler_cls=SCHEDULER_CLS['ewsjf'],
        external_parameters=external_parameters,
        # tensor_parallel_size=2
    )

    for i in range(len(rates)):

        await run_engine(engine_args, prompts, sampling_params, rates[i])


async def main_fcfs():
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100, min_tokens=1)

    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'data_30000_100_2000.csv')
    dataset = pd.read_csv(csv_path)
    prompts = dataset['input'].tolist()

    rates = [1000, 500, 100, 60, 40, 20, 10]
    engine_args = AsyncEngineArgs(
        model="meta-llama/Meta-Llama-3-8B",
        # tensor_parallel_size=2
    )

    for i in range(len(rates)):
        await run_engine(engine_args, prompts, sampling_params, rates[i])


async def main(queues_config):
    await main_ewsjf(queues_config)

    await main_fcfs()


async def run_engine(engine_args, prompts, sampling_params, rate):
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    start = time.time()
    print(f"Sending {len(prompts)} requests at rate of {rate} requests per second...")
    print(f"Estimated time to send all requests: {len(prompts) / rate:.2f} seconds")
    # Send requests with rate limiting
    tasks = await send_requests_with_rate_limit(engine, prompts, sampling_params, requests_per_second=rate)
    # Wait for all requests to complete
    print("Waiting for all requests to complete...")
    outputs = await asyncio.gather(*tasks)
    end = time.time()

    duration = end - start

    await metrics(duration, prompts, outputs, rate)

    print("\nGenerated Outputs:\n" + "-" * 60)
    print("Num answers: " + str(len(outputs)))
    print(f"\nRequest sending time: {len(prompts) / rate:.2f} seconds")
    print(f"Total execution time (until all requests complete): {end - start:.2f} seconds")


async def metrics(total_runtime, prompts, outputs, rate):
    # --- 5. Manually calculate aggregate metrics ---
    # Since output.metrics is None, we rely on our external timer.
    num_requests = len(prompts)
    total_generated_tokens = 0
    total_prompt_tokens = 0

    print("\n--- Processing Outputs (without internal metrics) ---")
    for i, output in enumerate(outputs):
        prompt_len = len(output.prompt_token_ids)
        generated_len = len(output.outputs[0].token_ids)

        total_prompt_tokens += prompt_len
        total_generated_tokens += generated_len

        # print(f"Prompt {i + 1}: Generated {generated_len} tokens.")

    # --- 6. Calculate aggregate throughput ---
    # Overall throughput in requests per second
    requests_per_second = num_requests / total_runtime if total_runtime > 0 else 0

    # Overall throughput in output tokens per second
    output_tokens_per_second = total_generated_tokens / total_runtime if total_runtime > 0 else 0

    print("\n--- Manually Measured Aggregate Metrics ---")
    print(f"Total time taken: {total_runtime:.2f} seconds")
    print(f"Total prompts processed: {num_requests}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print("--------------------------------------------------")
    print(f"Requests per second: {requests_per_second:.2f} req/s")
    print(f"Output tokens per second: {output_tokens_per_second:.2f} tokens/s")
    print("\nNOTE: TTFT and per-request latencies are not available in this version.")


if __name__ == "__main__":
    queues_30_100_2000 = [{'boundaries': (100, 163)},
                 {'boundaries': (164, 226)},
                 {'boundaries': (227, 289)},
                 {'boundaries': (290, 352)},
                 {'boundaries': (353, 415)},
                 {'boundaries': (416, 478)},
                 {'boundaries': (479, 541)},
                 {'boundaries': (542, 604)},
                 {'boundaries': (605, 667)},
                 {'boundaries': (668, 730)},
                 {'boundaries': (731, 793)},
                 {'boundaries': (794, 856)},
                 {'boundaries': (857, 919)},
                 {'boundaries': (920, 982)},
                 {'boundaries': (983, 1045)},
                 {'boundaries': (1046, 1108)},
                 {'boundaries': (1109, 1171)},
                 {'boundaries': (1172, 1234)},
                 {'boundaries': (1235, 1297)},
                 {'boundaries': (1298, 1360)},
                 {'boundaries': (1361, 1423)},
                 {'boundaries': (1424, 1486)},
                 {'boundaries': (1487, 1549)},
                 {'boundaries': (1550, 1612)},
                 {'boundaries': (1613, 1675)},
                 {'boundaries': (1676, 1738)},
                 {'boundaries': (1739, 1801)},
                 {'boundaries': (1802, 1864)},
                 {'boundaries': (1865, 1927)},
                 {'boundaries': (1928, 2000)}]

    asyncio.run(main_ewsjf(queues_30_100_2000))
    # asyncio.run(main_fcfs())
    # asyncio.run(main(queues_30_1000_7500))