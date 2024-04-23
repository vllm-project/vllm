import argparse
import gc
import time
from functools import total_ordering
import tqdm

import torch

from prompt_generator import PromptsGenerator


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def list_of_strings(arg):
    return arg.split(',')


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument("-k",
                        "--max_new_tokens",
                        type=int,
                        default=1024)
    parser.add_argument("-n",
                        "--num_queries",
                        type=list_of_ints,
                        help="number of queries to run",
                        default='10')
    parser.add_argument("-w",
                        "--warmup",
                        type=int,
                        help="number of queries for warming up",
                        default=128)
    parser.add_argument("-l",
                        "--prompt_length",
                        help="average number of tokens each prompt.",
                        type=list_of_ints,
                        default='1024')
    parser.add_argument("-m",
                        "--max_prompt_length",
                        help="max number of tokens each prompt.",
                        type=int,
                        default=3072)
    parser.add_argument('--framework',
                        required=True,
                        type=list_of_strings,
                        default='vllm,mii')
    parser.add_argument("-tp",
                        "--tensor_parallel",
                        type=int,
                        help="Tensor parallelism",
                        default='1')
    parser.add_argument('--model', type=str, required=True, help="path to the model")
    parser.add_argument('--quantization', type=str, default="deepspeedfp", help="quantization method")

    args, _ = parser.parse_known_args()
    return args


@total_ordering
class Benchmark:
    def __init__(self, framework, num_queries, input_length, output_length, latency, tensor_parallel):
        self.num_queries = num_queries

        def _avg(lt):
            return sum(lt) // len(lt)

        self.avg_input = _avg(input_length)

        self.framework = framework

        self.max_input = max(input_length)
        self.min_input = min(input_length)

        self.avg_output = _avg(output_length)
        self.max_output = max(output_length)
        self.min_output = min(output_length)

        self.tensor_parallel = tensor_parallel
        self.throughput = (sum(input_length)+sum(output_length))/latency
        self.throughput_output = sum(output_length) / latency
        self.latency = latency

    def __str__(self):
        return f'{self.framework}' \
            f', {self.num_queries}' \
            f', {self.avg_input}, {self.min_input}, {self.max_input}' \
            f', {self.avg_output}, {self.min_output}, {self.max_output}' \
            f', {self.latency: .2f}' \
            f', {self.throughput: .2f}' \
            f', {self.throughput_output: .2f}' \
            f', {self.tensor_parallel}'

    def __lt__(self, other):
        if self.num_queries != other.num_queries:
            return self.num_queries < other.num_queries
        if self.avg_input != other.avg_input:
            return self.avg_input < other.avg_input
        if self.tensor_parallel != other.tensor_parallel:
            return self.tensor_parallel < other.tensor_parallel
        if self.framework != other.framework:
            return self.framework < other.framework


def benchmark_mii(model, tensor_parallel, num_queries, warmup, prompt_lengths, max_prompt_length, max_new_tokens):
    import mii
    from deepspeed.inference import (DeepSpeedTPConfig,
                                     RaggedInferenceEngineConfig)
    from deepspeed.inference.v2.ragged import DSStateManagerConfig

    tp_config = DeepSpeedTPConfig(tp_size=tensor_parallel)
    mgr_config = DSStateManagerConfig(max_ragged_batch_size=128*tensor_parallel,
                                      max_ragged_sequence_count=128*tensor_parallel)
    inference_config = RaggedInferenceEngineConfig(tensor_parallel=tp_config,
                                                   state_manager=mgr_config)
    start = time.time()
    llm = mii.serve(
        model,
        deployment_name='mii',
        tensor_parallel=tensor_parallel,
        inference_engine_config=inference_config,
        replica_num=1,
        task='text-generation'
    )
    print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start llm engine")

    prompt_generator = PromptsGenerator(tokenizer_path=model)
    if warmup > 0:
        print('warmming up...')
        warmup_prompts = prompt_generator.generate(1024, 1024*0.3, 2048, warmup)
        text_input_prompts = [prompt[0] for prompt in warmup_prompts]
        llm.generate(text_input_prompts, max_new_tokens=max_new_tokens)
        print('warm up finished')

    benchmarks = []
    for prompt_length in prompt_lengths:
        for num_query in num_queries:
            prompt_generator.reset()
            prompts = [prompt[0] for prompt in prompt_generator.generate(
                average_token=prompt_length,
                variance=prompt_length*0.3,
                max_token=max_prompt_length,
                n=num_query,
                show_progress=True)]
            text_input_prompts = [prompt[0] for prompt in prompts]
            print('generating...')
            start = time.time()
            outputs = llm.generate(text_input_prompts,
                                   do_sample=False,
                                   top_p=1.0,
                                   max_new_tokens=max_new_tokens)
            latency = time.time() - start
            print('generating finished')

            input_lengths = []
            output_lengths = []

            for output in outputs:
                input_lengths.append(output.prompt_length)
                output_lengths.append(output.generated_length)
            benchmarks.append(Benchmark(framework='mii',
                                        num_queries=num_query,
                                        input_length=input_lengths,
                                        output_length=output_lengths,
                                        latency=latency,
                                        tensor_parallel=tensor_parallel))
            for i in benchmarks:
                print(i)

    try:
        # Destroy
        llm.terminate_server()
    except Exception as e:
        print(f'failed to destroy mii: {e}')
    return benchmarks


def benchmark_vllm(model, tensor_parallel, num_queries, warmup, prompt_lengths, max_prompt_length, max_new_tokens, quantization):
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import \
        destroy_model_parallel

    # Create an LLM.
    start = time.time()
    llm = LLM(model=model, 
              tensor_parallel_size=tensor_parallel, 
              trust_remote_code=True, 
              quantization=quantization,
            #   gpu_memory_utilization=0.88,
              disable_custom_all_reduce=False)
    print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start llm engine")

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0,  # get rid of nondeterminism.
                                     top_p=1.0,
                                     top_k=-1,
                                     max_tokens=max_new_tokens)

    prompt_generator = PromptsGenerator(tokenizer_path=model)
    if warmup > 0:
        print('warmming up...')
        warmup_prompts = prompt_generator.generate(1024, 1024*0.3, 2048, warmup)
        text_input_prompts = [prompt[0] for prompt in warmup_prompts]
        llm.generate(text_input_prompts, sampling_params)
        print('warm up finished')

    benchmarks = []
    for prompt_length in prompt_lengths:
        for num_query in num_queries:
            prompt_generator.reset()
            prompts = prompt_generator.generate(average_token=prompt_length,
                                                variance=prompt_length*0.3,
                                                max_token=max_prompt_length,
                                                n=num_query,
                                                show_progress=True)
            text_input_prompts = [prompt[0] for prompt in prompts]
            start = time.time()
            outputs = llm.generate(text_input_prompts, sampling_params)
            latency = time.time() - start

            input_lengths = []
            output_lengths = []

            for output in outputs:
                input_lengths.append(len(output.prompt_token_ids))
                output_lengths.append(len(output.outputs[0].token_ids))

            benchmarks.append(Benchmark(framework='vllm',
                                        num_queries=num_query,
                                        input_length=input_lengths,
                                        output_length=output_lengths,
                                        latency=latency,
                                        tensor_parallel=tensor_parallel))
            for i in benchmarks:
                print(i)

    try:
        # Destroy
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
    except Exception as e:
        print(f'failed to destroy vllm: {e}')
    return benchmarks


if __name__ == "__main__":
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    benchmarks = []

    if 'mii' in args.framework:
        benchmarks += benchmark_mii(
                model=args.model,
                tensor_parallel=args.tensor_parallel,
                num_queries=args.num_queries,
                warmup=args.warmup,
                prompt_lengths=args.prompt_length,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens)

    if 'vllm' in args.framework:
        benchmarks += benchmark_vllm(
                model=args.model,
                tensor_parallel=args.tensor_parallel,
                num_queries=args.num_queries,
                warmup=args.warmup,
                prompt_lengths=args.prompt_length,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                quantization=args.quantization)

    benchmarks = sorted(benchmarks)

    print('!!!---Printing results---!!!')
    # Output results as a csv
    print('framework, num_prompts, avg_input, max_input, min_input, avg_output, max_output, min_output, latency(s), throughput, throughput_output, tensor_parallel')
    for i in benchmarks:
        print(i)
