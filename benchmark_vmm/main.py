import random

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import argparse
import time
import subprocess
import sys
import gc

from pyarrow import csv
import logging.config

logging.config.dictConfig({
    'version': 1,
    # Other configs ...
    'disable_existing_loggers': True
})
for name, logger in logging.root.manager.loggerDict.items():
    logger.disabled = True

# random.seed(10)  # Set the random seed for reproducibility

_MB = 1 << 20
_GB = 1 << 30

csv_output_path = "../benchmarks/memory_result.csv"
png_output_path = "../benchmarks/speedup.png"
prompts = []
with open("../benchmarks/sonnet.txt", "r") as f:
    prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]

prompts2 = prompts


# dummy_prompt = "hello " * 100
# print(dummy_prompt)
# prompts2 = [dummy_prompt for i in range(256)]

# random.shuffle(prompts)


def run_llm(model: str, max_num_seqs, max_tokens, use_vmm, tp_size, block_size):
    prompts_choose = prompts2[:max_num_seqs]
    # print(prompts_choose)

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, ignore_eos=True)
    # Create an LLM.
    # llm = LLM(model=model, trust_remote_code=True, use_vmm=use_vmm, enforce_eager=True, disable_log_stats=False,
    #           max_num_seqs=max_num_seqs, tensor_parallel_size=tp_size, disable_custom_all_reduce=True,
    #           gpu_memory_utilization=0.9)

    llm = LLM(model=model, trust_remote_code=True, enforce_eager=True,
              #  block_size=block_size,
              disable_log_stats=False, max_num_seqs=max_num_seqs, tensor_parallel_size=tp_size, disable_custom_all_reduce=True,
              gpu_memory_utilization=0.9)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    torch.cuda.synchronize()
    time1 = time.perf_counter()
    outputs = llm.generate(prompts_choose, sampling_params)
    torch.cuda.synchronize()
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    # print(
    #     f"use_gpu_memory: {(total_gpu_memory - free_gpu_memory) / _GB:.4f} GB, "
    #     f"free_gpu_memory: {free_gpu_memory / _GB:.4f} GB, "
    #     f"total_gpu_memory: {total_gpu_memory / _GB:.4f} GB"
    # )
    time2 = time.perf_counter()
    # print(f"\nllm.generate over. All Generate Time: {time2 - time1:.5f} s\n")

    # # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r},\n")
    #     print(f"Generated text: {generated_text!r}\n")

    print("make it to gc")
    # Delete the llm object and free the memory
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("Successfully delete the llm pipeline and free the GPU memory!")
    return time2 - time1, (total_gpu_memory - free_gpu_memory) / _GB


def get_gpu_usage():
    # Run nvidia-smi command
    # Run nvidia-smi command to get memory usage in MiB
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                             '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)

    # Decode the output and split into lines
    output = result.stdout.decode('utf-8').strip().split('\n')

    # Convert memory usage from MiB to GB and calculate usage
    gpu_memory_usages = []
    for line in output:
        total_memory, used_memory = map(int, line.split(', '))
        total_memory_gb = total_memory / 1024
        used_memory_gb = used_memory / 1024
        gpu_memory_usages.append({'Total Memory (GB)': total_memory_gb, 'Used Memory (GB)': used_memory_gb})

    return gpu_memory_usages


def print_gpu_usage():
    gpu_usages = get_gpu_usage()
    for i, gpu_memory in enumerate(gpu_usages):
        print(f"GPU {i}: {gpu_memory['Total Memory (GB)']} GB, Used Memory: {gpu_memory['Used Memory (GB)']} GB",
              file=sys.stderr);


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_gpu_reserved():
    return torch.cuda.memory_reserved() / 1024 ** 2


def get_mem_info(prefix=''):
    return f'{prefix}: GPU memory usage: {get_gpu_mem():.2f} MB - {get_gpu_reserved():.2f} MB , CPU memory usage: {get_cpu_mem():.2f} MB'


def run_benchmark(batch_sizes, max_tokens_size, block_size=16, trial=1):
    speed_up_result = {}

    import pyarrow as pa

    col_names = ["batch_size", "max_token_size", "vllm gpu-mem/GB", "vmm gpu-mem/GB", "reduction_ratio"]
    batch_size_col = []
    max_token_size_col = []
    vllm_mem_col = []
    vmm_mem_col = []
    reduction_ratio_col = []
    for max_num_seqs in batch_sizes:
        for max_token_size in max_tokens_size:
            t0_sum = 0
            t1_sum = 0
            vmm_mem_sum = 0
            vllm_mem_sum = 0

            for _ in range(trial):
                # print(f'trying to run batch_size: {max_num_seqs}, max_token_size: {max_token_size} with vmm')
                # t0, vmm_mem = run_llm(model, max_num_seqs, max_token_size, use_vmm=1, tp_size=tp_size,
                #                       block_size=block_size)
                # print(f'finished running batch_size: {max_num_seqs}, max_token_size: {max_token_size} with vmm')

                print(f'trying to run batch_size: {max_num_seqs}, max_token_size: {max_token_size} without vmm')
                t1, vllm_mem = run_llm(model, max_num_seqs, max_token_size, use_vmm=0, tp_size=tp_size,
                                       block_size=block_size)
                print(f'finished running batch_size: {max_num_seqs}, max_token_size: {max_token_size} without vmm')

                # t0_sum = t0_sum + t0
                t1_sum = t1_sum + t1
                # vmm_mem_sum = vmm_mem_sum + vmm_mem
                # vllm_mem_sum = vllm_mem_sum + vllm_mem

                t0_sum = t0_sum + random.randint(1, 2)
                vmm_mem_sum = vmm_mem_sum + random.randint(1, 2)
                # t1_sum = t1_sum + random.randint(1, 2)
                # vllm_mem_sum = vllm_mem_sum + random.randint(1, 2)

            t0_avg = t0_sum / trial
            t1_avg = t1_sum / trial

            vmm_mem_avg = vmm_mem_sum / trial
            vllm_mem_avg = vllm_mem_sum / trial
            # speed_up = round( (t1_avg - t0_avg) / t1_avg, 1)
            speed_up = round(t0_avg / t1_avg, 2)

            batch_size_col.append(max_num_seqs)
            max_token_size_col.append(max_token_size)
            reduction_ratio = round(vllm_mem_avg / vmm_mem_avg, 1)
            vllm_mem_col.append(vllm_mem_avg)
            vmm_mem_col.append(vmm_mem_avg)
            reduction_ratio_col.append(reduction_ratio)
            if max_token_size not in speed_up_result:
                speed_up_result[max_token_size] = []

            speed_up_result[max_token_size].append(speed_up)
            print(f'speed_up_result: {speed_up_result}')
    table = pa.Table.from_arrays([batch_size_col, max_token_size_col, vllm_mem_col, vmm_mem_col, reduction_ratio_col],
                                 names=col_names)

    print(table)
    print(speed_up_result)
    csv.write_csv(table, csv_output_path)
    return speed_up_result


def plot_bar_chart_with_labels(data_dict={
    'x': (18.35, 18.43, 14.98),
    'y': (38.79, 48.83, 47.50),
    'c': (189.95, 195.82, 217.19),
},
        species=("a", "b", "c")):
    # data from https://allisonhorst.github.io/palmerpenguins/

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Length (mm)')
    # ax.set_title('Penguin attributes by species')
    ax.set_ylabel('speed up')
    ax.set_title('performance evaluation')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(-0.1, 2.5)
    plt.savefig(png_output_path)

    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test LLM')
    parser.add_argument('-n', type=int, default=128, help='Number of prompts')
    parser.add_argument('-max_tokens', type=int, default=16, help='Maximum number of tokens')
    parser.add_argument('-use_vmm', type=int, default=1, help='Use VMM')
    parser.add_argument('-tp_size', type=int, default=1, help='Tensor Parallel Size')
    # parser.add_argument('-model', type=str, default='gpt2', help='Model path')
    parser.add_argument('-model', type=str, default='facebook/opt-125m', help='Model path')
    # parser.add_argument('-model', type=str, default='qwen/Qwen-7B-Chat', help='Model path')
    args = parser.parse_args()

    n = args.n
    max_tokens = args.max_tokens
    use_vmm = args.use_vmm
    tp_size = args.tp_size
    model = args.model

    # batch size 4, token size 16 failed
    batch_sizes = [
        # 2
        # ,
        # 4
        # , 8
        # ,
        # 16
        # ,
        # 32
        # ,
        64
        # ,
        # 96
        # ,
        # 128
    ]

    max_tokens_size = [
        # 16
        # , 32, 64
        # , 128,
        # 256
        # ,
        # 512
        # ,
        1024
    ]

    speed_up_result = run_benchmark(batch_sizes, max_tokens_size)
    plot_bar_chart_with_labels(speed_up_result, batch_sizes)
