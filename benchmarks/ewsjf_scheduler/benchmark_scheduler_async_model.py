# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# import multiprocessing
import asyncio
import os
import time

from datasets import load_dataset

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# from vllm.v1.core.sched.ewsjf_scheduler.clustering import HybridQueueBuilder
from vllm.v1.core.sched.ewsjf_scheduler.scheduler_cls import SCHEDULER_CLS

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


async def generate_async(engine, request_id, prompt, sampling_params):
    results_generator = engine.generate(prompt, sampling_params, request_id)
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def send_requests_with_rate_limit(
    engine, prompts, sampling_params, requests_per_second=15
):
    """Send requests at a controlled rate (requests per second)"""
    tasks = []
    # Time interval between requests (in seconds)
    interval = 1.0 / requests_per_second

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


async def main_ewsjf(
    queues_config, prompts, model, rates, tensor_parallel_size, sampling_params
):
    # clustering:
    # data = np.array(dataset['train']['in_len_tokens_llama'])
    # clustering_config = get_queues_config(data)
    # configs = [queues_config, clustering_config]

    external_parameters = {"queues_config": queues_config, "step_size": 1500}
    engine_args = AsyncEngineArgs(
        model=model,
        scheduler_cls=SCHEDULER_CLS["ewsjf"],
        external_parameters=external_parameters,
        tensor_parallel_size=tensor_parallel_size,
        max_num_batched_tokens=7000,
        # Maximum number of tokens to be processed in a single iteration.
        # This config has no static default. If left unspecified by the user,
        # it will be set in `EngineArgs.create_engine_config` based on usage.
        # long_prefill_token_threshold=8000,
        # For chunked prefill, a request is considered long if the prompt is
        # longer than this number of tokens.
        enable_chunked_prefill=True,
    )

    for i in range(len(rates)):
        await run_engine(engine_args, prompts, sampling_params, rates[i], "EWSJF")


async def main_sjf(prompts, model, rates, tensor_parallel_size, sampling_params):
    engine_args = AsyncEngineArgs(
        model=model,
        scheduler_cls=SCHEDULER_CLS["sjf"],
        tensor_parallel_size=tensor_parallel_size,
        max_num_batched_tokens=7000,
        # long_prefill_token_threshold=8000,
        enable_chunked_prefill=True,
    )

    for i in range(len(rates)):
        await run_engine(engine_args, prompts, sampling_params, rates[i], "SJF")


async def main_fcfs(prompts, model, rates, tensor_parallel_size, sampling_params):
    engine_args = AsyncEngineArgs(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_num_batched_tokens=7000,
        # long_prefill_token_threshold=8000,
        enable_chunked_prefill=True,
    )

    for i in range(len(rates)):
        await run_engine(engine_args, prompts, sampling_params, rates[i], "FCFS")


async def main(
    queues_config, prompts, model, rates, tensor_parallel_size, sampling_params
):
    await main_ewsjf(
        queues_config,
        prompts,
        model,
        rates,
        tensor_parallel_size,
        sampling_params,
    )

    await main_sjf(prompts, model, rates, tensor_parallel_size, sampling_params)

    await main_fcfs(prompts, model, rates, tensor_parallel_size, sampling_params)


async def run_engine(engine_args, prompts, sampling_params, rate, scheduler_name):
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    start = time.time()
    print(f"Sending {len(prompts)} requests at rate of {rate} requests per second...")
    print(f"Estimated time to send all requests: {len(prompts) / rate:.2f} seconds")
    # Send requests with rate limiting
    tasks = await send_requests_with_rate_limit(
        engine, prompts, sampling_params, requests_per_second=rate
    )
    # Wait for all requests to complete
    end_sending = time.time()

    print(f"Waiting for all requests to complete... time: {end_sending - start}")
    outputs = await asyncio.gather(*tasks)
    end = time.time()

    duration = end - start

    await metrics(duration, prompts, outputs, scheduler_name)

    print("\nGenerated Outputs:\n" + "-" * 60)
    print("Num answers: " + str(len(outputs)))
    print(f"\nRequest sending time: {len(prompts) / rate:.2f} seconds")
    print(
        f"Total execution time (until all requests complete): {end - start:.2f} seconds"
    )


async def metrics(total_runtime, prompts, outputs, scheduler_name):
    # --- 5. Manually calculate aggregate metrics ---
    # Since output.metrics is None, we rely on our external timer.
    num_requests = len(prompts)
    total_generated_tokens = 0
    total_prompt_tokens = 0
    print(f"\n--- Scheduler: {scheduler_name} ---")

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
    output_tokens_per_second = (
        total_generated_tokens / total_runtime if total_runtime > 0 else 0
    )

    print("\n--- Manually Measured Aggregate Metrics ---")
    print(f"Total time taken: {total_runtime:.2f} seconds")
    print(f"Total prompts processed: {num_requests}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print("--------------------------------------------------")
    print(f"Requests per second: {requests_per_second:.2f} req/s")
    print(f"Output tokens per second: {output_tokens_per_second:.2f} tokens/s")
    print("\nNOTE: TTFT and per-request latencies are not available in this version.")


# def get_queues_config(data):
#      builder = HybridQueueBuilder(kmeans_k=10, max_queues=30, verbose=True)
#
#      queue = multiprocessing.Queue()
#      p = multiprocessing.Process(
#          target=builder.build_queues_multiprocessing,
#          args=(queue, data,)
#      )
#      p.start()
#      queues = queue.get()
#      p.terminate()
#
#      return queues

if __name__ == "__main__":
    queues_30_100_16000 = [
        {"boundaries": (100, 633)},
        {"boundaries": (634, 1167)},
        {"boundaries": (1168, 1701)},
        {"boundaries": (1702, 2235)},
        {"boundaries": (2236, 2769)},
        {"boundaries": (2770, 3293)},
        {"boundaries": (3294, 3817)},
        {"boundaries": (3818, 4341)},
        {"boundaries": (4342, 4865)},
        {"boundaries": (4866, 5389)},
        {"boundaries": (5390, 5913)},
        {"boundaries": (5914, 6437)},
        {"boundaries": (6438, 6961)},
        {"boundaries": (6962, 7485)},
        {"boundaries": (7486, 8009)},
        {"boundaries": (8010, 8533)},
        {"boundaries": (8534, 9057)},
        {"boundaries": (9058, 9581)},
        {"boundaries": (9582, 10105)},
        {"boundaries": (10106, 10629)},
        {"boundaries": (10630, 11153)},
        {"boundaries": (11154, 11677)},
        {"boundaries": (11678, 12101)},
        {"boundaries": (12102, 12625)},
        {"boundaries": (12626, 13149)},
        {"boundaries": (13150, 13673)},
        {"boundaries": (13674, 14197)},
        {"boundaries": (14198, 14721)},
        {"boundaries": (14722, 15245)},
        {"boundaries": (15246, 16000)},
    ]
    queues_30_100_2000 = [
        {"boundaries": (100, 163)},
        {"boundaries": (164, 226)},
        {"boundaries": (227, 289)},
        {"boundaries": (290, 352)},
        {"boundaries": (353, 415)},
        {"boundaries": (416, 478)},
        {"boundaries": (479, 541)},
        {"boundaries": (542, 604)},
        {"boundaries": (605, 667)},
        {"boundaries": (668, 730)},
        {"boundaries": (731, 793)},
        {"boundaries": (794, 856)},
        {"boundaries": (857, 919)},
        {"boundaries": (920, 982)},
        {"boundaries": (983, 1045)},
        {"boundaries": (1046, 1108)},
        {"boundaries": (1109, 1171)},
        {"boundaries": (1172, 1234)},
        {"boundaries": (1235, 1297)},
        {"boundaries": (1298, 1360)},
        {"boundaries": (1361, 1423)},
        {"boundaries": (1424, 1486)},
        {"boundaries": (1487, 1549)},
        {"boundaries": (1550, 1612)},
        {"boundaries": (1613, 1675)},
        {"boundaries": (1676, 1738)},
        {"boundaries": (1739, 1801)},
        {"boundaries": (1802, 1864)},
        {"boundaries": (1865, 1927)},
        {"boundaries": (1928, 2000)},
    ]
    queues_30_1000_7500 = [
        {"boundaries": (1000, 1216)},
        {"boundaries": (1217, 1433)},
        {"boundaries": (1434, 1650)},
        {"boundaries": (1651, 1867)},
        {"boundaries": (1868, 2084)},
        {"boundaries": (2085, 2301)},
        {"boundaries": (2302, 2518)},
        {"boundaries": (2519, 2735)},
        {"boundaries": (2736, 2952)},
        {"boundaries": (2953, 3169)},
        {"boundaries": (3170, 3386)},
        {"boundaries": (3387, 3603)},
        {"boundaries": (3604, 3820)},
        {"boundaries": (3821, 4037)},
        {"boundaries": (4038, 4254)},
        {"boundaries": (4255, 4471)},
        {"boundaries": (4472, 4688)},
        {"boundaries": (4689, 4905)},
        {"boundaries": (4906, 5122)},
        {"boundaries": (5123, 5339)},
        {"boundaries": (5340, 5556)},
        {"boundaries": (5557, 5773)},
        {"boundaries": (5774, 5990)},
        {"boundaries": (5991, 6207)},
        {"boundaries": (6208, 6424)},
        {"boundaries": (6425, 6641)},
        {"boundaries": (6642, 6858)},
        {"boundaries": (6859, 7075)},
        {"boundaries": (7076, 7292)},
        {"boundaries": (7293, 7500)},
    ]
    queues_30_1000_7500_2 = [
        {"boundaries": (500, 1000)},
        {"boundaries": (1001, 1500)},
        {"boundaries": (1501, 2000)},
        {"boundaries": (2001, 2500)},
        {"boundaries": (2501, 3000)},
        {"boundaries": (3001, 3500)},
        {"boundaries": (3501, 4000)},
        {"boundaries": (4001, 4500)},
        {"boundaries": (4501, 5000)},
        {"boundaries": (5001, 5500)},
        {"boundaries": (5501, 6000)},
        {"boundaries": (6001, 6500)},
        {"boundaries": (6501, 7000)},
        {"boundaries": (7001, 7500)},
        {"boundaries": (7501, 8000)},
    ]

    # dataset = load_dataset("ChayaLevi/data-100-2000")
    dataset = load_dataset("ChayaLevi/data-100-16000")
    # prompts = list(dataset['train']['input'])

    # dataset = pd.read_csv("/home/chaya/data_30000_100_2000_.csv")
    # dataset = pd.read_csv("/home/chaya/data_10000_1000_7500.csv")
    prompts = list(dataset["input"])

    # prompt = prompts[:1000]

    rates = [10, 20]  # [10, 20, 40, 60, 100, 200, 500]
    # "meta-llama/Meta-Llama-3-8B" #"Qwen/Qwen2.5-7B-Instruct"
    model = "meta-llama/Meta-Llama-3-8B"
    tensor_parallel_size = 1
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=100, min_tokens=1
    )

    asyncio.run(main_sjf(prompts, model, rates, tensor_parallel_size, sampling_params))
    # asyncio.run(main_ewsjf(queues_30_1000_7500_2, prompts,
    #                        model, rates, tensor_parallel_size,
    #                        sampling_params))
    # asyncio.run(main(queues_30_100_16000, prompts,
    #                  model, rates, tensor_parallel_size,
    #                  sampling_params))
