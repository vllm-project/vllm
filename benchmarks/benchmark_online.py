import argparse
import asyncio
from enum import Enum
import json
import multiprocessing
import os
import queue
import random
import requests
import subprocess
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import time
from typing import Iterable, List, Tuple

from benchmark_tools import OnlineBenchmark, Query, summarize_online_benchmarks

from prompt_generator import PromptsGenerator


MAX_SEQUENCE_LENGTH = 4096

class Framework(str, Enum):
    DEEPSPEED_MII = "mii"
    VLLM = "vllm"


def list_of_floats(arg):
    return list(map(float, arg.split(',')))


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def parse_args():
    parser = argparse.ArgumentParser(description="OnlineBenchmark inference")
    parser.add_argument("-k",
                        "--max_new_tokens",
                        type=int,
                        default=1024)
    parser.add_argument("-w",
                        "--warmup",
                        type=int,
                        help="number of queries for warming up",
                        default=128)
    parser.add_argument("-l",
                        "--prompt_length",
                        help="average number of tokens each prompt.",
                        type=list_of_ints,
                        default="512,1024,1536,2048,2560")
    parser.add_argument("-tp",
                        "--tensor_parallel",
                        type=int,
                        help="Tensor parallelism",
                        default='1')
    parser.add_argument("-c",
                        "--client_num",
                        type=int,
                        help="Number of clients",
                        default=64)
    parser.add_argument('--framework',
                        required=True,
                        type=str,
                        default='vllm')
    parser.add_argument("-qps",
                        "--queries_per_second",
                        type=list_of_floats,
                        help="List of queries per second",
                        default="0.5,1.0,1.5,2.0")
    parser.add_argument('--model', type=str, required=True, help="path to the model")

    args, _ = parser.parse_known_args()
    return args


args = parse_args()
if args.framework == Framework.DEEPSPEED_MII:
    import mii


class CallbackObject:
    def __init__(self):
        self.start_time = time.time()
        self.responses = []
        self.first = True
        self.first_token_time = 0.0


def benchmark_mii(
    client,
    prompts: List[str],
    max_new_tokens: int,
    start_time: float,
) -> List[OnlineBenchmark]:
    benchmarks = []
    callback_obj = CallbackObject()

    def callback(response):
        if callback_obj.first:
            callback_obj.first_token_time = time.time()
            callback_obj.first = False
        callback_obj.responses.append(response[0])

    client.generate(
        prompts=prompts,
        streaming_fn=callback,
        do_sample=False,
        top_p=1.0,
        max_new_tokens=max_new_tokens
    )
    end_time = time.time()
    time_to_first_token = callback_obj.first_token_time - start_time
    latency = end_time - start_time

    input_lengths = []
    output_lengths = []

    input_lengths.append(callback_obj.responses[-1].prompt_length)
    output_lengths.append(callback_obj.responses[-1].generated_length)

    benchmarks.append(
        OnlineBenchmark(
            framework=Framework.DEEPSPEED_MII,
            input_length=input_lengths,
            output_length=output_lengths,
            time_to_first_token=time_to_first_token,
            latency=latency,
            tensor_parallel=args.tensor_parallel,
        )
    )
    return benchmarks


def benchmark_vllm(
    model: str,
    prompts: List[str],
    max_new_tokens: int,
    query: Query,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> List[OnlineBenchmark]:
    api_url = "http://localhost:8000/generate"
    pload = {
        "prompt": prompts[0],
        "n": 1,
        "use_beam_search": False,
        "temperature": 0,
        "top_p": 0.9,
        "top_k": 1,
        "max_tokens": max_new_tokens,
        "ignore_eos": False,
        "stream": True,
    }

    def get_streaming_response(response: requests.Response, time_last_token) -> Iterable[Tuple[str, float]]:
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False,
                                        delimiter=b"\0"):
            if chunk:
                time_now = time.time()
                data = json.loads(chunk.decode("utf-8"))
                output: str = data["text"][0]
                yield output, time_now - time_last_token
                time_last_token = time_now

    response = requests.post(api_url, json=pload, stream=True)
    token_gen_time = []
    last_response = ""
    for h, t in get_streaming_response(response, query.start_time):
        last_response = h
        token_gen_time.append(t)

    time_to_first_token = token_gen_time[0]
    latency = time.time() - query.start_time

    input_length = [query.input_tokens]
    output_length = [len(tokenizer.encode(last_response)) - query.input_tokens]

    benchmarks = ([
        OnlineBenchmark(
            framework=Framework.VLLM,
            input_length=input_length,
            output_length=output_length,
            time_to_first_token=time_to_first_token,
            latency=latency,
            tensor_parallel=args.tensor_parallel,
        )
    ])

    return benchmarks


def _run_parallel(
    model: str,
    framework: str,
    barrier: multiprocessing.Barrier,
    query_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    max_new_tokens: int,
    client_num: int,
) -> None:
    pid = os.getpid()
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    if framework == Framework.DEEPSPEED_MII:
        client = mii.client(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    barrier.wait()

    # Warmup
    try:
        while True:
            query = query_queue.get(timeout=1)
            print(f"warmup queue size: {query_queue.qsize()} ({pid})", flush=True)
            if framework == Framework.DEEPSPEED_MII:
                benchmark_mii(client=client, prompts=[query.prompt], max_new_tokens=max_new_tokens, start_time=query.start_time)
            elif framework == Framework.VLLM:
                benchmark_vllm(model=model, prompts=[query.prompt], max_new_tokens=max_new_tokens, query=query, tokenizer=tokenizer)
    except queue.Empty:
        pass

    barrier.wait()

    time.sleep(random.uniform(0, client_num) * 0.01)
    while True:
        try:
            query = query_queue.get(timeout=300)
            if len(query.prompt) == 0:
                break
            if framework == Framework.DEEPSPEED_MII:
                benchmarks = benchmark_mii(client=client, prompts=[query.prompt], max_new_tokens=max_new_tokens, start_time=query.start_time)
            elif framework == Framework.VLLM:
                benchmarks = benchmark_vllm(model=model, prompts=[query.prompt], max_new_tokens=max_new_tokens, query=query, tokenizer=tokenizer)
            [result_queue.put(benchmark) for benchmark in benchmarks]
        except queue.Empty:
            pass


def run_benchmarks(
    client_num: int,
    framework: str,
    model: str,
    queries_per_second_list: List[float],
    prompt_length_list: List[int],
    max_new_tokens: int,
    warmup: int,
) -> List[OnlineBenchmark]:
    proc = None
    try:
        start_time = time.time()
        print(f"loading model: {model}")
        if framework == Framework.DEEPSPEED_MII:
            mii.serve(
                model_name_or_path=model,
                deployment_name=model,
                tensor_parallel=args.tensor_parallel,
                replica_num=1,
            )
        elif framework == Framework.VLLM:
            # proc = subprocess.Popen(f'python -m vllm.entrypoints.api_server --model={args.model} -tp={args.tensor_parallel}', shell=True, stdout=subprocess.DEVNULL)
            ready = False
            proc_start_time = time.time()
            timeout_secs = 1200
            # Model should be loaded within timeout_secs, abort if not the case.
            while time.time() - proc_start_time < timeout_secs:
                try:
                    if requests.head("http://localhost:8000/generate") is not None:
                        ready = True
                        break
                except:
                    pass
                time.sleep(1)
            if not ready:
                raise ValueError(f"Unable to load {args.model} in vllm within {timeout_secs} seconds.")
        print(f"{framework} loaded model {model} in {time.time() - start_time} seconds")
            

        barrier = multiprocessing.Barrier(client_num + 1)
        query_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        processes = []
        for _ in range(client_num):
            processes.append(
                multiprocessing.Process(
                    target=_run_parallel,
                    args=(model, framework, barrier, query_queue, result_queue, max_new_tokens, client_num)
                )
            )
        for p in processes:
            p.start()

        prompt_generator = PromptsGenerator(tokenizer_path=model)

        # Generate warmup prompts. This will generate n * len(prompt_lengths) warmup queries
        prompts = (
            prompt_generator.generate(
                average_token=max(prompt_length_list),
                variance=max(prompt_length_list)*0.3,
                max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                n=warmup,
                show_progress=True,
            )
        )
        [query_queue.put(Query(prompt)) for prompt in prompts]

        # Barrier to wait for all clients to initialized
        barrier.wait()
        # Barrier for all clients to finish warmup
        barrier.wait()

        time.sleep(5)

        summarization_results = []
        for prompt_length in prompt_length_list:
            for queries_per_second in queries_per_second_list:
                print(f"benchmarking {prompt_length} prompt length at {queries_per_second} qps")
                # Generate prompts to run benchmark on
                prompts = (
                    prompt_generator.generate(
                        average_token=prompt_length,
                        variance=prompt_length*0.3,
                        max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                        n=100,
                        show_progress=True,
                    )
                )

                # For 5 minutes, send a query every 1/qps
                i = 0
                total_queries_sent = 0
                time_start = time.time()
                while time.time() - time_start < 300:
                    if i >= len(prompts):
                        i = 0
                    query_queue.put(Query(prompts[i]))
                    i += 1
                    total_queries_sent += 1
                    time.sleep(1/queries_per_second)

                benchmarks = []
                while len(benchmarks) < total_queries_sent:
                    res = result_queue.get(block=True)
                    benchmarks.append(res)

                total_time = time.time() - time_start

                summarization_results.append(summarize_online_benchmarks(
                    framework=args.framework,
                    token_input=prompt_length,
                    queries_per_second=queries_per_second,
                    clients=args.client_num,
                    benchmarks=sorted(benchmarks),
                    total_time=total_time,
                ))

        for _ in range(client_num):
            query_queue.put(Query(("", 0)))

        for summarization_result in summarization_results:
            print(summarization_result)

    except Exception as e:
        raise e
    finally:
        if proc is not None:
            proc.terminate()


if __name__ ==  "__main__":
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    benchmarks = run_benchmarks(
        client_num=args.client_num,
        framework=args.framework,
        model=args.model,
        queries_per_second_list=args.queries_per_second,
        prompt_length_list=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
    )