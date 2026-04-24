# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import regex as re

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_DIR = SCRIPT_DIR.parent.parent.parent
BEE_DIR = Path(str(os.getenv("BEE_DIR")))
GPU_TYPE = os.getenv("GPU_TYPE")
TP_SIZE = os.getenv("TP_SIZE")
QUANTIZATION_TYPES = ["fp8", "w4a16"]


def extract_summary_metrics_name(file_path: str) -> list:
    with open(file_path) as f:
        task_text = f.read()

    # Regex pattern to match metric definitions: [options.summary_metrics.metric_name]
    metric_pattern = r"\[options\.summary_metrics\.([^\]]+)\]"

    # Find all matches in the config text
    extracted_metrics_name = re.findall(metric_pattern, task_text)

    return extracted_metrics_name


def get_quantization_type(model_name: str) -> str:
    for q_type in QUANTIZATION_TYPES:
        if q_type.lower() in model_name.lower():
            return q_type

    raise ValueError(
        f"No quantization type found in string: '{model_name}'. "
        f"Expected one of: {QUANTIZATION_TYPES}"
    )


def get_metrics(data: dict, metric_names: list) -> dict:
    metrics = {}
    for metric_name in metric_names:
        metrics[metric_name] = data.get(metric_name)
    return metrics


def main(summary_path, output_directory, co_bench_directory):
    with open(summary_path) as file:
        summary = json.load(file)
    with open(SCRIPT_DIR.parent / "configs" / "eval-config.json") as f:
        eval_config = json.load(f)
    with open(SCRIPT_DIR.parent / "configs" / "co_bench_map.json") as f:
        co_bench_mapping = json.load(f)
    model_map = co_bench_mapping["model_map"]
    sku_map = co_bench_mapping["sku_map"]
    task_map = co_bench_mapping["task_map"]
    gcp_script = ""

    for data in summary:
        # Only convert data points that co bench can display
        if data["model"] in model_map:
            q_type = get_quantization_type(data["model"])
            model = f"{model_map[data['model']]}-{GPU_TYPE}-tp{TP_SIZE}-{q_type}-eval"
            export = data["checkpoint_path"]
            image = data["commit"]
            sku = f"{sku_map[GPU_TYPE]}-TP{TP_SIZE}"
            formatted_time = datetime.strptime(
                data["timestamp"], "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y-%m-%d-%H-%M")
            for taskname in task_map:
                if taskname in eval_config[data["model"]]:
                    bee_eval_config_path = BEE_DIR / taskname
                    metrics = get_metrics(
                        data["eval_results"],
                        extract_summary_metrics_name(bee_eval_config_path),
                    )
                    co_bench_data = [
                        {
                            "model": model,
                            "export": export,
                            "image": image,
                            "sku": sku,
                            "quantization": q_type,
                            "estimator": "openaiapi",
                            "url": "localhost",
                            "metrics": json.dumps(metrics),
                        }
                    ]
                    # Write to CSV file
                    csv_file_name = f"{model}.{task_map[taskname]}.{formatted_time}.csv"
                    with open(
                        output_directory / csv_file_name, "w", newline=""
                    ) as csvfile:
                        fieldnames = co_bench_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(co_bench_data)

                    # Prepare GCP uploading script
                    if not gcp_script:
                        gcp_script = (
                            """#!/bin/bash\n
                            set -x\n
                            script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")\n
                            """
                            + gcp_script
                        )
                    dst_path = (
                        f"{co_bench_directory}/evals/openaiapi/"
                        f"{model_map[data['model']]}/{sku}/{q_type}/"
                    )
                    gcp_script = (
                        gcp_script
                        + f"gcloud storage cp $script_dir/{csv_file_name} {dst_path}\n"
                    )

    # Write GCP uploading script
    if gcp_script:
        with open(output_directory / "gcp_upload.sh", "w") as f:
            f.write(gcp_script)
        os.chmod(output_directory / "gcp_upload.sh", 0o755)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert summary JSON file to data format"
        " that can be displayed on co-bench"
    )
    parser.add_argument(
        "--summary_path",
        default=VLLM_DIR / "results" / "eval_results_summary.json",
        required=False,
        help="Path to the summary JSON file",
    )
    parser.add_argument(
        "--output_directory",
        default=VLLM_DIR / "results" / "co-bench",
        required=False,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--co_bench_directory",
        default="gs://cohere-bench-reports",
        required=False,
        help="Path to the gcp co-bench",
    )
    args = parser.parse_args()

    main(args.summary_path, args.output_directory, args.co_bench_directory)
