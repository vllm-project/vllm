# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import os
import subprocess
from dataclasses import dataclass

from vllm.v1.spec_decode.dynamic.profiling_server import (
    kill_server,
    setup_server,
    start_server,
)


NGRAM_FMT = "min-{min}-max-{max}-k-{k}"
EAGLE_FMT = "k-{k}"


def run_command(command):
    try:
        result = subprocess.run(
            f"bash -c '{command}'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        print("Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:")
        print(e.stderr)


def run_benchmarks(dry_run, 
                   model_dir, 
                   draft_dir, 
                   method,
                   prompt_lookup_max,
                   prompt_lookup_min,
                   num_speculative_tokens_list, 
                   batch_size_list,
                   max_vllm_batch_size, 
                   tp, 
                   temp, 
                   top_p, 
                   top_k, 
                   num_batches,
                   dataset_name,
                   dataset_path,
                   result_dir, 
                   extra_log_arg):

    assert method in ["vanilla", "ngram", "eagle", "eagle3"], (
        "invalid method specified"
    )

    assert max_vllm_batch_size == max(batch_size_list), (
        "max_vllm_batch_size must be equal to max of batch_size"
    )

    # setup server
    setup_server()

    port = 9001

    # ablation
    num_exp_run = 0

    # collate all spec configs to run for a given method
    all_spec_config = []
    if method == "ngram":
        for ngram_k in num_speculative_tokens_list:
            all_spec_config.append(
                {
                    "method": "ngram",
                    "num_speculative_tokens": ngram_k,
                    "prompt_lookup_max": prompt_lookup_max,
                    "prompt_lookup_min": prompt_lookup_min,
                }
            )
    elif method == "eagle":
        for eagle_k in num_speculative_tokens_list:
            all_spec_config.append(
                {
                    "method": "eagle",
                    "model": draft_dir,
                    "num_speculative_tokens": eagle_k,
                    "draft_tensor_parallel_size": tp,
                }
            )
    else:
        # vanilla case
        all_spec_config.append(None)

    for spec_config in all_spec_config:
        
        # start server
        server_process = start_server(
            port=port,
            target_model_dir=model_dir,
            spec_config=spec_config,
            tp=tp,
            max_vllm_bs=max_vllm_batch_size,
            dry_run=dry_run,
        )

        # start client
        for bench_concurrency in batch_size_list:
            spec_config_str = "vanilla"
            if method == "ngram":
                spec_config_str = NGRAM_FMT.format(
                    min=spec_config["prompt_lookup_min"],
                    max=spec_config["prompt_lookup_max"],
                    k=spec_config["num_speculative_tokens"],
                )
            elif method == "eagle":
                spec_config_str = EAGLE_FMT.format(
                    k=spec_config["num_speculative_tokens"]
                )

            # dataset specific config
            if "philschmid/mt-bench" in dataset_path:
                bench_config_str = "mt_bench"
            
            num_prompts = num_batches * bench_concurrency
            bench_vllm_config = f"--dataset-name {dataset_name} --dataset-path {dataset_path} --num-prompts {num_prompts}"

            print(
                f"Number of prompts in {dataset_path}: {num_prompts}"
            )

            # create dir if not exists
            # TODO: make the path shared with generate_config.py
            # result_dir = f"{result_dir}/tp-{tp}_temp-{temp}_top_p-{top_p}_top_k-{top_k}/{bench_dataset}/{method}/"  # noqa E501
            final_result_dir = f"{result_dir}/{method}/"  # noqa E501
            if not os.path.exists(final_result_dir):
                os.makedirs(final_result_dir)

            cmd = f'''time vllm bench serve --port {port} --save-result --save-detailed \
            --model {model_dir} \
            --backend openai-chat \
            --endpoint /v1/chat/completions \
            {bench_vllm_config} \
            --max-concurrency {bench_concurrency} \
            --temperature={temp} \
            --top-p={top_p} \
            --top-k={top_k} \
            --result-dir "{final_result_dir}" \
            --result-filename "{spec_config_str}_{bench_config_str}_bs-{bench_concurrency}_{extra_log_arg}.txt"'''

            print(cmd)
            num_exp_run += 1

            if not dry_run:
                run_command(cmd)

        # server teardown: kill server and any gpu processes
        kill_server(port, server_process)

        print(f"Total number of experiments run: {num_exp_run}")


"""
# eagle
time python3 vllm/v1/spec_decode/online_profiling_client.py \
    --batch-size-list 1 4 16 64 256 \
    --num-speculative-tokens-list 1 3 5 \
    --max-vllm-batch-size 256 \
    --method eagle \
    --model-dir meta-llama/Llama-3.1-8B-Instruct \
    --draft-dir yuhuili/EAGLE-LLaMA3.1-Instruct-8B

# vanilla
time python3 vllm/v1/spec_decode/online_profiling_client.py \
    --batch-size-list 1 4 16 64 256 \
    --max-vllm-batch-size 256 \
    --method vanilla
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry run mode. If set, commands will be printed but not executed.",
    )
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--draft-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="vanilla")
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument(
        "--num-speculative-tokens-list", nargs="*", type=int, default=[1, 3, 5]
    )
    parser.add_argument(
        "--batch-size-list", nargs="*", type=int, default=[1, 4, 16, 64, 256]
    )
    parser.add_argument(
        "--max-vllm-batch-size",
        type=int,
        help="Max vllm server batch size (max concurrency)",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--num-batches", type=int, default=20, help="Number of batches to run for each benchmark.")
    parser.add_argument("--dataset-name", type=str, default="hf")
    parser.add_argument("--dataset-path", type=str, default="philschmid/mt-bench")
    parser.add_argument("--result-dir", type=str, default="./log/dynamic_sd")
    parser.add_argument("--extra-log-arg", type=str, default="")
    args = parser.parse_args()

    run_benchmarks(
        dry_run = args.dry_run,
        model_dir = args.model_dir,
        draft_dir = args.draft_dir,
        method = args.method,
        prompt_lookup_max = args.prompt_lookup_max,
        prompt_lookup_min = args.prompt_lookup_min,
        num_speculative_tokens_list = args.num_speculative_tokens_list,
        batch_size_list = args.batch_size_list,
        max_vllm_batch_size = args.max_vllm_batch_size,
        tp = args.tp,
        temp = args.temp,
        top_p = args.top_p,
        top_k = args.top_k,
        num_batches = args.num_batches,
        dataset_name = args.dataset_name,
        dataset_path = args.dataset_path,
        result_dir = args.result_dir,
        extra_log_arg = args.extra_log_arg)