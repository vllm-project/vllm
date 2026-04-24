# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import argparse
import subprocess

import pandas as pd


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


def main(args: argparse.Namespace):
    print(args)

    K = args.k
    BENCH_TP = args.tp

    BENCH_DTYPE = "fp8"
    ALL_BENCH_CONCURRENCY = [1, 4, 16]

    ALL_SAMPLING_PROFILE = [
        # {'temperature': 0, 'topp': 1}, # greedy
        {"temperature": 0.3, "topp": 0.75},  # default
    ]

    ALL_BENCH_DATASET = [
        # "mt_bench", # SKIP
        # "mtbench-cohere", # has cohere preamble
        # "code",
        # "reasoning",
        # "translation",
        # "creative_writing",
        # "sharegpt_sampled_300",
        # "sharegpt_1k_100_bench_data",
        "summarize_1_2k",
        "summarize_2_4k",
        "summarize_4_8k",
        "gov_report_16k",
        # "gov_report_16k",
        # "gov_report_32k",
        # "narrative_qa_64k_test_filtered_gt_15_output_tokens",
        # "narrative_qa_128k_test_filtered_gt_10_output_tokens"
    ]

    for BENCH_CONCURRENCY in ALL_BENCH_CONCURRENCY:
        for BENCH_DATASET in ALL_BENCH_DATASET:
            ONLINE_CUSTOM_DATASET = f"/host/vllm-cohere/data/{BENCH_DATASET}.jsonl"
            jsonl_data = pd.read_json(path_or_buf=ONLINE_CUSTOM_DATASET, lines=True)
            NUM_PROMPTS = len(jsonl_data)
            print(f"Number of prompts in {ONLINE_CUSTOM_DATASET}: {NUM_PROMPTS}")

            for SAMPLING_PROFILE in ALL_SAMPLING_PROFILE:
                BENCH_TEMPERATURE = SAMPLING_PROFILE["temperature"]
                BENCH_TOPP = SAMPLING_PROFILE["topp"]

                # NOTE: set the model name in the eagle.py
                # BENCH_MODEL_NAME = "vanilla"
                BENCH_MODEL_NAME = "eagle-v2"

                TARGET_MODEL_PATH = (
                    "/host/engines/vllm/Command3-111B_TP4_FP8_16k/poseidon"
                )

                # skip chat template but make sure the input data has them
                cmd = f'''time python3 benchmarks/benchmark_serving.py --port 9001 --save-result --save-detailed \
                --backend vllm \
                --model {TARGET_MODEL_PATH} \
                --endpoint /v1/completions \
                --dataset-name custom \
                --dataset-path {ONLINE_CUSTOM_DATASET} \
                --custom-skip-chat-template \
                --num-prompts {NUM_PROMPTS} \
                --max-concurrency {BENCH_CONCURRENCY} \
                --temperature={BENCH_TEMPERATURE} \
                --top-p={BENCH_TOPP} \
                --result-dir "./log/EAGLE-1/online/" \
                --result-filename "{BENCH_MODEL_NAME}_k{K}_bs{BENCH_CONCURRENCY}_tp{BENCH_TP}_temp_{BENCH_TEMPERATURE}_top_p_{BENCH_TOPP}_data_{BENCH_DATASET}_dtype_{BENCH_DTYPE}.txt"'''

                print(cmd)
                run_command(cmd)


# USAGE:
# Terminal 1: start vllm server in one terminal. It will need a fixed TP size and K (number of spec tokens)
# Terminal 2:
# time python3 vllm/cohere/utils/eagle/sweep_eagle_online_benchmark.py --tp 8 --k 3

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="K used to start vllm server",
    )
    parser.add_argument(
        "--tp",
        type=int,
        required=True,
        help="TP used to start vllm server",
    )

    args = parser.parse_args()
    main(args)
