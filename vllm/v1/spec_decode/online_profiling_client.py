import os
import subprocess
import pandas as pd
from dataclasses import dataclass
import argparse
from vllm.vllm.v1.spec_decode.online_profiling_server import (
    start_server,
    kill_server, 
    setup_server)


@dataclass
class Dataset:
    name: str
    config: list

NGRAM_FMT = "min-{min}-max-{max}-k-{k}"
EAGLE_FMT = "k-{k}"

def run_command(command):
    try:
        result = subprocess.run(f"bash -c '{command}'", 
                                shell=True, 
                                check=True, 
                                capture_output=True, 
                                text=True)
        print("Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:")
        print(e.stderr)


def run_benchmarks(args):

    # setup server
    setup_server()

    port=9001
    all_sampling_profile=[
        {'temperature': 0, 'topp': 1}, # greedy
    ]

    MTBENCH_CONFIG = [{"num_samples_per_seq": 20}]

    all_bench_dataset = [
        Dataset(name = "philschmid/mt-bench", config = MTBENCH_CONFIG),
    ]

    assert (all(len(ds.config) > 0 for ds in all_bench_dataset)), "Each dataset must have at least one config"

    all_ngram_params = [{"min": 2, "max": 5, "k": k} for k in args.num_speculative_tokens]
    all_eagle_params = args.num_speculative_tokens

    # ablation
    num_exp_run = 0
    for tp in args.tp:
        for spec_method in args.method_list:
            # collate all spec configs to run for a given method
            all_spec_config = []
            if spec_method == "ngram":
                for ngram_params in all_ngram_params:
                    all_spec_config.append({
                        "method": "ngram",
                        "num_speculative_tokens": ngram_params['k'],
                        "prompt_lookup_max": ngram_params['max'],
                        "prompt_lookup_min": ngram_params['min'],
                    })
            elif spec_method == "eagle":
                for eagle_k in all_eagle_params:
                    all_spec_config.append({
                        "method": "eagle",
                        "model": args.draft_dir,
                        "num_speculative_tokens": eagle_k,
                        "draft_tensor_parallel_size": tp,
                    })
            else:
                # vanilla case
                all_spec_config.append(None)

            for spec_config in all_spec_config:
                # start server
                server_process = start_server(port=port, 
                                              target_model_dir=args.model_dir, 
                                              spec_config=spec_config, 
                                              tp=tp, 
                                              max_vllm_bs=args.max_vllm_batch_size, 
                                              dry_run=args.dry_run)

                # start client
                for bench_concurrency in args.batch_size_list:
                    for bench_dataset_object in all_bench_dataset:
                        bench_dataset = bench_dataset_object.name
                        for bench_config in bench_dataset_object.config:
                            for sampling_profile in all_sampling_profile:
                                bench_temperature = sampling_profile['temperature']
                                bench_topp = sampling_profile['topp']

                                spec_config_str = "vanilla"
                                if spec_method == "ngram":
                                    spec_config_str = NGRAM_FMT.format(
                                        min=spec_config['prompt_lookup_min'],
                                        max=spec_config['prompt_lookup_max'],
                                        k=spec_config['num_speculative_tokens']
                                    )
                                elif spec_method == "eagle":
                                    spec_config_str = EAGLE_FMT.format(
                                        k=spec_config['num_speculative_tokens']
                                    )

                                # dataset specific config
                                if "philschmid/mt-bench" in bench_dataset:
                                    bench_config_str = f"mt_bench"
                                    num_prompts = bench_config["num_prompts"] * bench_concurrency
                                    bench_vllm_serve_config = f'--dataset-name hf --dataset-path {bench_dataset} --num-prompts {num_prompts}'

                                print(f"Number of prompts in {bench_dataset}: {num_prompts}")

                                # create dir if not exists
                                result_dir = f"{args.result_dir}/tp-{tp}_temp-{bench_temperature}_top_p-{bench_topp}/{bench_dataset}/{spec_method}/online/"
                                if not os.path.exists(result_dir):
                                    os.makedirs(result_dir)

                                cmd = f'''time vllm bench serve --port {port} --save-result --save-detailed \
                                --model {args.model_dir} \
                                --backend openai-chat \
                                --endpoint /v1/chat/completions \
                                {bench_vllm_serve_config} \
                                --max-concurrency {bench_concurrency} \
                                --temperature={bench_temperature} \
                                --top-p={bench_topp} \
                                --result-dir "{result_dir}" \
                                --result-filename "{spec_config_str}_{bench_config_str}_bs-{bench_concurrency}_{args.extra_log_arg}.txt"'''

                                print(cmd)
                                num_exp_run += 1

                                if not args.dry_run:
                                    run_command(cmd)

                # server teardown: kill server and any gpu processes
                kill_server(port, server_process)

    print(f"Total number of experiments run: {num_exp_run}")


# time python3 vllm/cohere/utils/eagle/sweep_ngram_eagle_online_benchmark.py 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run in dry run mode. If set, commands will be printed but not executed.")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--draft-dir", type=str, default=None)
    parser.add_argument("--method-list", type=list[str], default=["vanilla", "eagle"])
    parser.add_argument("--num-speculative-tokens-list", type=list[int], default=[1, 3, 5])
    parser.add_argument("--batch-size-list", type=list[int], default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--max-vllm-batch-size", type=int, help="Max vllm server batch size (max concurrency)")
    parser.add_argument("--tp-list", type=list[int], default=[1])
    parser.add_argument("--result-dir", type=str, default="./log")
    parser.add_argument("--extra-log-arg", type=str, default="")
    args = parser.parse_args()

    assert all([method in ["vanilla", "ngram", "eagle", "eagle3"] for method in args.method_list]), \
        "invalid method in method_list"
    
    assert 1 in args.batch_size_list, "batch_size must contain 1"
    assert 1 in args.num_speculative_tokens_list, "num_speculative_tokens must contain 1"
    assert args.max_vllm_batch_size == max(args.batch_size_list), \
        "max_vllm_batch_size must be equal to max of batch_size"

    model_dir = args.model_dir
    args.model_dir = "meta-llama/Llama-3.1-8B-Instruct" if args.model_dir is None else args.model_dir
    
    if args.method == "eagle" or args.method == "eagle3":
        if args.method == "eagle" and args.eagle_dir is None:
            args.eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

        elif args.method == "eagle3" and args.eagle_dir is None:
            args.eagle_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        speculative_config = {
            "method": args.method,
            "model": args.eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
        }

    run_benchmarks(args)