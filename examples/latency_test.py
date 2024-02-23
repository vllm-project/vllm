import time, sys, os, json, argparse
from typing import List, Tuple
from tqdm import tqdm

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm import LLM

path = '/data/zyh/datasets/ShareGPT52K/sg_90k_part1.json'
max_length = 2048
max_test_num = 100
sampling_params = SamplingParams(max_tokens=256)

def create_test_prompts() -> List[str]:
    prompts = []
    count = 0
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for entry in data: 
        for topic in entry['conversations']:
            if topic['from'] == 'human':
                if len(topic['value']) < max_length and len(topic['value']) != 0:
                    prompts.append(topic['value'])
                    count += 1
                    if count >= max_test_num:
                        return prompts

# add one time and step one time.
def process_requests(engine: LLMEngine,
                     test_prompts: List[str]):
    """Continuously process a list of prompts and handle the outputs."""

    request_id = 0
    iter_count = 1
    start_time_list = {}
    start_iter_list = {}
    time_list = []
    while test_prompts or engine.has_unfinished_requests():

        if test_prompts:
            prompt = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            start_time_list[request_id] = time.perf_counter()
            start_iter_list[request_id] = iter_count
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        
        for request_output in request_outputs:
            if request_output.finished:
                ID = int(request_output.request_id)
                start_time = start_time_list[ID]
                end_time = time.perf_counter()
                # print(ID, start_iter_list[ID], iter_count, \
                #    request_output.prompt[:10].replace('\n', ' '), \
                #    end_time - start_time)
                time_list.append(end_time - start_time)

        iter_count += 1

    print("\nnum of requests processed:", len(time_list))
    print("avg time:", sum(time_list)/len(time_list))
    print("block_size = ", engine.cache_config.block_size)
    return sum(time_list)/len(time_list)

# add all requests and use step to process.
def my_process_requests(engine: LLMEngine, 
                         test_prompts: List[str]):
    request_id = 0
    while test_prompts:
        prompt = test_prompts.pop(0)
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1
    
    start_time = time.perf_counter()
    iter_count = 1
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                end_time = time.perf_counter()
                print(request_output.request_id, iter_count,\
                     request_output.finished, len(request_output.prompt), \
                     request_output.prompt[:10], end_time - start_time)
        iter_count += 1

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)

    print("warming up...")
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)
    print("warming up done.")
    num_iters = 3
    # benchmark
    # file_name = 'latency-recompute-' + str(engine.cache_config.block_size) + '.txt'
    # with open(file_name, 'w') as f:
    #    sys.stdout = f
    latencies = []
    for _ in tqdm(range(num_iters)):
        test_prompts = create_test_prompts()
        avg_latency = process_requests(engine, test_prompts)
        latencies.append(avg_latency)
    print(latencies)
    #sys.stdout = sys.__stdout__
    #print(latencies)
    return sum(latencies)/len(latencies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)