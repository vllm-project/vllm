import torch
from vllm import LLM, SamplingParams
# from vllm.global_vars import get_time
import random
import random
import argparse
import time

random.seed(0)  # Set the random seed for reproducibility

_MB = 1 << 20
_GB = 1 << 30

dummy_prompt = "hello " * 100
# print(dummy_prompt)

prompts = []
with open("../benchmarks/sonnet.txt", "r") as f:
    prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]

prompts2 = prompts


# prompts2 = [dummy_prompt for i in range(256)]

# random.shuffle(prompts)


def run_llm(model: str, max_num_seqs, max_tokens, use_vmm, tp_size):
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    prompts_choose = prompts2[:max_num_seqs]
    # print(prompts_choose)

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, ignore_eos=True)

    # Create an LLM.
    # llm = LLM(model=model, trust_remote_code=True, use_vmm=use_vmm, enforce_eager=True, disable_log_stats=False, max_num_seqs=16, tensor_parallel_size=tp_size, disable_custom_all_reduce=True)
    # llm = LLM(model=model, trust_remote_code=True, use_vmm=use_vmm, enforce_eager=True, disable_log_stats=False,
    llm = LLM(model=model, trust_remote_code=True, enforce_eager=True, disable_log_stats=False,
              max_num_seqs=max_num_seqs, tensor_parallel_size=tp_size, disable_custom_all_reduce=True,
              gpu_memory_utilization=0.9)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    torch.cuda.synchronize()
    time1 = time.perf_counter()
    outputs = llm.generate(prompts_choose, sampling_params)
    torch.cuda.synchronize()
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    print(
        f"use_gpu_memory: {(total_gpu_memory - free_gpu_memory) / _GB:.4f} GB, "
        f"free_gpu_memory: {free_gpu_memory / _GB:.4f} GB, "
        f"total_gpu_memory: {total_gpu_memory / _GB:.4f} GB"
    )
    time2 = time.perf_counter()

    print(f"\nllm.generate over. All Generate Time: {time2 - time1:.5f} s\n")

    # # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r},\n")
        print(f"Generated text: {generated_text!r}\n")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test LLM')
    parser.add_argument('-max_num_seqs', type=int, default=64, help='Number of prompts')
    parser.add_argument('-max_tokens', type=int, default=1024, help='Maximum number of tokens')
    parser.add_argument('-use_vmm', type=int, default=0, help='Use VMM')
    parser.add_argument('-tp_size', type=int, default=1, help='Tensor Parallel Size')
    parser.add_argument('-model', type=str, default='facebook/opt-125m', help='Model path')
    # parser.add_argument('-model', type=str, default='gpt2', help='Model path')
    # parser.add_argument('-model', type=str, default='qwen/Qwen-7B-Chat', help='Model path')

    args = parser.parse_args()

    max_num_seqs = args.max_num_seqs
    max_tokens = args.max_tokens
    use_vmm = args.use_vmm
    tp_size = args.tp_size
    model = args.model

    run_llm(model, max_num_seqs, max_tokens, use_vmm, tp_size)
