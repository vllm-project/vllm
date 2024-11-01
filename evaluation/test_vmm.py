'''
 Copyright (c) ByteDance Inc.
 Authors:
  - Chak-Pong Chung (chakpong.chung@bytedance.com)
  - Tongping Liu (tongping.liu@bytedance.com)
'''

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import random
import random
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

random.seed(0)  # Set the random seed for reproducibility

_MB = 1 << 20
_GB = 1 << 30

#dummy_prompt = "hello " * 100
# print(dummy_prompt)

prompts = []
with open("./benchmarks/sonnet.txt", "r") as f:
    prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]

prompts2 = prompts


def run_llm(model: str, seqs_num, max_tokens, use_vmm, tp_size, block_size):
    prompts_choose = prompts2[:seqs_num]
    # print(prompts_choose)

    max_length = 0
    for i in range(0, seqs_num):
        length = len(prompts_choose[i])
        if length > max_length:
            max_length = length

    print(f"max_length out of {seqs_num} (from {len(prompts2)} prompts is {max_length}")
    #import pdb 
    #pdb.set_trace()
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, ignore_eos=True)

    # Create an LLM.
    llm = LLM(model=model, trust_remote_code=True, use_vmm=use_vmm, enforce_eager=True, block_size=block_size,
              disable_log_stats=False,
              max_num_seqs=64, gpu_memory_utilization=0.9)
              #max_num_seqs=64, tensor_parallel_size=tp_size, disable_custom_all_reduce=True, gpu_memory_utilization=0.9)

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


def plot_bar_chart_with_labels(penguin_means={
    'Bill Depth': (18.35, 18.43, 14.98),
    'Bill Length': (38.79, 48.83, 47.50),
    'Flipper Length': (189.95, 195.82, 217.19),
    },
    species=("Adelie", "Chinstrap", "Gentoo")):
    # data from https://allisonhorst.github.io/palmerpenguins/

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
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
    ax.set_ylim(-0.1, 1.5)
    plt.savefig('speedup.png')

    # plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test LLM')
    parser.add_argument('-n', type=int, default=128, help='Number of prompts')
    parser.add_argument('-max_tokens', type=int, default=16, help='Maximum number of tokens')
    parser.add_argument('-use_vmm', type=int, default=1, help='Use VMM')
    parser.add_argument('-tp_size', type=int, default=1, help='Tensor Parallel Size')
    parser.add_argument('-model', type=str, default='gpt2', help='Model path')
    #parser.add_argument('-model', type=str, default='qwen/Qwen-7B-Chat', help='Model path')
    args = parser.parse_args()

    n = args.n
    max_tokens = args.max_tokens
    use_vmm = args.use_vmm
    tp_size = args.tp_size
    model = args.model

    seqs_num = [
        #8,
        # ,
        # 16
        # ,
        #128,
        # ,
        # 64
        # ,
        # 96
        # ,
        256
    ]

    max_tokens_size = [
        # , 128, 256,
        #512,
        # ,
        4096
    ]

    speed_up_result = {}

    import pyarrow as pa

    col_names = ["seqs_num", "max_token_size", "vllm gpu-mem/GB", "vmm gpu-mem/GB", "reduction_ratio"]
    seqs_num_col = []
    max_token_size_col = []
    vllm_mem_col = []
    vmm_mem_col = []
    reduction_ratio_col = []
    import sys
    for seq_num in seqs_num:
        for max_token_size in max_tokens_size:
            print(f'trying to run seq_num: {seq_num}, max_token_size: {max_token_size} without vmm', file=sys.stderr)
            t1, vllm_mem = run_llm(model, seq_num, max_token_size, use_vmm=False, tp_size=tp_size, block_size=16)
            print(f'finished running seq_num: {seq_num}, max_token_size: {max_token_size} without vmm', file=sys.stderr)
