# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import os
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


def main():
    BENCH_DTYPE = "fp8"
    ALL_BENCH_TP = [2]
    ALL_BENCH_CONCURRENCY = [64]
    ALL_K = [3]  # number of spec tokens

    ALL_SAMPLING_PROFILE = [
        # {'temperature': 0, 'topp': 1}, # greedy
        {"temperature": 0.3, "topp": 0.75},  # default
    ]

    ALL_BENCH_DATASET = [
        # "mt_bench", # SKIP
        "mtbench-cohere",  # has cohere preamble
        "code",
        "reasoning",
        "translation",
        "creative_writing",
        # "sharegpt_sampled_300",
        "sharegpt_1k_100_bench_data",
        # "summarize_1_2k",
        # "summarize_2_4k",
        # "summarize_4_8k",
        # "gov_report_16k",
        # "gov_report_32k",
        # "narrative_qa_64k_test_filtered_gt_15_output_tokens",
        # "narrative_qa_128k_test_filtered_gt_10_output_tokens"
    ]

    MODEL_DIR = "/host/engines/vllm/Command3-111B_TP4_FP8_16k/poseidon"
    EAGLE_DIR = "/host/engines/vllm/2j2j_latest_data_softlabel_lr0001_bs64_ln_gclip_inpord_ealn_wreg100/fp8/llm_compressor/poseidon"

    for BENCH_CONCURRENCY in ALL_BENCH_CONCURRENCY:
        for BENCH_DATASET in ALL_BENCH_DATASET:
            ONLINE_CUSTOM_DATASET = f"/host/vllm-cohere/data/{BENCH_DATASET}.jsonl"
            jsonl_data = pd.read_json(path_or_buf=ONLINE_CUSTOM_DATASET, lines=True)
            NUM_PROMPTS = len(jsonl_data)
            print(f"Number of prompts in {ONLINE_CUSTOM_DATASET}: {NUM_PROMPTS}")

            for BENCH_TP in ALL_BENCH_TP:
                CUDA_VISIBLE_DEVICES_STRING = ",".join(
                    str(device_id) for device_id in range(BENCH_TP)
                )

                for K in ALL_K:
                    for SAMPLING_PROFILE in ALL_SAMPLING_PROFILE:
                        BENCH_TEMPERATURE = SAMPLING_PROFILE["temperature"]
                        BENCH_TOPP = SAMPLING_PROFILE["topp"]

                        # NOTE: set the target and draft model name in the eagle.py
                        BENCH_MODEL_NAME = "eagle-v2"

                        # NOTE: skip chat template but make sure the input data has them
                        cmd = f"""time CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES_STRING} VLLM_USE_V1=1 python3 /host/vllm-cohere/examples/offline_inference/spec_decode.py \
                            --method eagle \
                            --model-dir {MODEL_DIR} \
                            --eagle-dir {EAGLE_DIR} \
                            --max-model-len 150000 \
                            --dataset-name "custom" \
                            --dataset-path {ONLINE_CUSTOM_DATASET} \
                            --custom-skip_chat_template \
                            --num-prompts {NUM_PROMPTS} \
                            --num_spec_tokens {K} \
                            --max_num_seqs {BENCH_CONCURRENCY} \
                            --tp {BENCH_TP} \
                            --draft_tp {BENCH_TP} \
                            --temp={BENCH_TEMPERATURE} \
                            --top_p={BENCH_TOPP} &> /host/vllm-cohere/data/bench_log/{BENCH_MODEL_NAME}_k{K}_bs{BENCH_CONCURRENCY}_tp{BENCH_TP}_temp_{BENCH_TEMPERATURE}_top_p_{BENCH_TOPP}_data_{BENCH_DATASET}_dtype_{BENCH_DTYPE}.txt"""

                        print(cmd)
                        run_command(cmd)


# USAGE: time python3 vllm/cohere/utils/eagle/sweep_eagle_benchmark.py

# DOWNLOAD DATASET
# 4 categories
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/code/code.jsonl .
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/reasoning/reasoning.jsonl .
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/translation/translation.jsonl .
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/creative_writing/creative_writing.jsonl .

# sharegpt
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/sharegpt/sharegpt_sampled_300.jsonl .
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/sharegpt/sharegpt_1k_100_bench_data.jsonl .

# Long context
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/long_context/gov_report_16k.jsonl .
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/long_context/gov_report_32k.jsonl .
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/long_context/narrative_qa_64k_test_filtered_gt_15_output_tokens.jsonl .
# gsutil -m cp gs://cohere-dev-central-2/kkt/tuna-data/medusa_eval_data/long_context/narrative_qa_128k_test_filtered_gt_10_output_tokens.jsonl .

# CONVERT DATASET: this will create folder structure similar to mt_bench with question.jsonl
# python3 vllm/cohere/utils/eagle/benchmark_prepare_dataset.py --data_dir="/host/vllm-cohere/data/code.jsonl" --output_dir="/host/vllm-cohere/data"
# python3 vllm/cohere/utils/eagle/benchmark_prepare_dataset.py --data_dir="/host/vllm-cohere/data/reasoning.jsonl" --output_dir="/host/vllm-cohere/data"
# python3 vllm/cohere/utils/eagle/benchmark_prepare_dataset.py --data_dir="/host/vllm-cohere/data/translation.jsonl" --output_dir="/host/vllm-cohere/data"
# python3 vllm/cohere/utils/eagle/benchmark_prepare_dataset.py --data_dir="/host/vllm-cohere/data/creative_writing.jsonl" --output_dir="/host/vllm-cohere/data"
# python3 vllm/cohere/utils/eagle/benchmark_prepare_dataset.py --data_dir="/host/vllm-cohere/data/sharegpt_sampled_300.jsonl" --output_dir="/host/vllm-cohere/data"
# python3 vllm/cohere/utils/eagle/benchmark_prepare_dataset.py --data_dir="/host/vllm-cohere/data/sharegpt_1k_100_bench_data.jsonl" --output_dir="/host/vllm-cohere/data"

if __name__ == "__main__":
    os.makedirs("/host/vllm-cohere/data/bench_log", exist_ok=True)
    main()
