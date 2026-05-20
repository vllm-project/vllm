# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate json config for serving tests; sweeps values
optional model override for more convenient local testing
"""

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import regex as re
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_PATH = SCRIPT_DIR.parent / "configs" / "serving-cohere-tests.json"
EVAL_CONFIG_PATH = SCRIPT_DIR.parent / "configs" / "eval-config.json"
BENCHMARK_CONFIG_PATH = SCRIPT_DIR.parent / "configs" / "benchmark-config.yaml"

# TODO(czhu): if we want to benchmark multiple models/configs, we can
# define different templates in json files and use environment variables
# to control which ones to load.
SERVER_DEFAULTS: dict[str, Any] = {
    "disable_log_stats": "",
}

CLIENT_DEFAULTS: dict[str, Any] = {
    "backend": "vllm",
    "dataset_name": "random",
    "ignore_eos": "",
}


def generate_eval_configs(
    models: dict[str, str],
    eval_config: dict[str, Any],
    tp_size: int,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for model_name, model_path in models.items():
        server_params: dict[str, Any] = {
            **SERVER_DEFAULTS,
            "model": model_path,
            "tensor_parallel_size": tp_size,
            "served_model_name": model_name,
            "mm_processor_cache_type": "shm",
        }

        # Melody for C5 models
        if "c5" in model_name:
            server_params["tool_call_parser"] = "cohere_command4"
            server_params["reasoning_parser"] = "cohere_command4"
            server_params["enable_auto_tool_choice"] = ""

        # Add speculative decoding configuration if model path contains "eagle"
        if "eagle" in model_path.lower():
            eagle_path = model_path + "/eagle"
            speculative_config = json.dumps(
                {
                    "method": "eagle",
                    "model": eagle_path,
                    "num_speculative_tokens": 3,
                    "draft_tensor_parallel_size": tp_size,
                },
                separators=(",", ":"),
            )
            server_params["speculative-config"] = f"'{speculative_config}'"

        # mapping bee eval tasks with model_name without tp suffix
        key = re.sub(r"_tp\d+$", "", model_name)
        try:
            tests = eval_config[key]
        except KeyError as err:
            raise KeyError(
                f"Model '{model_name}' (key '{key}') is not configured for "
                "evaluation in eval_config. "
                "Please check your eval-config.json and ensure this model is listed."
            ) from err

        test_names = [
            "_".join(PurePosixPath(p).with_suffix("").parts[1:]) for p in tests
        ]
        configs.append(
            {
                "test_name": f"serving_{model_name}_bee_eval_{'_'.join(test_names)}",
                "qps_list": ["inf"],
                "server_parameters": server_params,
                "eval_parameters": {
                    "model": model_name,
                    "tests": tests,
                    # Only pass thinking_token_budget for C5 models
                    **(
                        {
                            "thinking_token_budget": 4096,
                            "reasoning_thinking_token_budget": 20480,
                        }
                        if "c5" in model_name
                        else {}
                    ),
                },
            }
        )
    return configs


def generate_benchmark_configs(
    models: dict[str, str],
    benchmarks: list[dict[str, Any]],
    tp_size: int,
    output_len: int,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for model_name, model_path in models.items():
        for benchmark in benchmarks:
            input_len = benchmark["input_len"]
            requests = benchmark["requests"]
            warmup = benchmark["warmup"]

            for mc in benchmark["users"]:
                configs.append(
                    {
                        "test_name": (
                            f"serving_{model_name}_random_{mc=}_{input_len}_{output_len}"
                        ),
                        "qps_list": ["inf"],
                        "server_parameters": {
                            **SERVER_DEFAULTS,
                            "model": model_path,
                            "tensor_parallel_size": tp_size,
                            # TODO: when benchmarking spec decode,
                            # need to load real model weights.
                            "load_format": "dummy",
                            # prefix caching affects benchmark results,
                            # even for random dataset
                            "no_enable_prefix_caching": "",
                        },
                        "client_parameters": {
                            **CLIENT_DEFAULTS,
                            "model": model_path,
                            "random_input_len": input_len,
                            "random_output_len": output_len,
                            "num_warmups": warmup,
                            "num_prompts": requests * mc,
                            "max_concurrency": mc,
                        },
                    }
                )
    return configs


# TODO: to optimize runtime we can instantiate the server once and run multiple
# test configs against it.
def main(mode: str):
    model_name = os.environ.get("MODEL_NAME")
    model_path = os.environ.get("MODEL_PATH")
    if not model_name or not model_path:
        raise ValueError("MODEL_NAME and MODEL_PATH environment variables are required")

    tp_size = int(os.environ.get("TP_SIZE", 1))
    models = {model_name: model_path}

    if mode == "eval":
        with open(EVAL_CONFIG_PATH) as fp:
            eval_config = json.load(fp)
        configs = generate_eval_configs(models, eval_config, tp_size)
    else:
        benchmark_output_len = os.environ.get("BENCHMARK_OUTPUT_LEN")
        if benchmark_output_len is None:
            raise ValueError(
                "BENCHMARK_OUTPUT_LEN environment variable is required "
                "for benchmark mode"
            )
        with open(BENCHMARK_CONFIG_PATH) as fp:
            benchmark_config = yaml.safe_load(fp)
        output_len = int(benchmark_output_len)
        if output_len <= 0:
            raise ValueError("BENCHMARK_OUTPUT_LEN must be a positive integer")
        configs = generate_benchmark_configs(
            models, benchmark_config["benchmarks"], tp_size, output_len
        )

    with open(CFG_PATH, "w") as fp:
        json.dump(configs, fp, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate evaluation or benchmark configurations."
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "eval"],
        default="benchmark",
        help="Configuration mode (default: benchmark)",
    )
    args = parser.parse_args()

    main(mode=args.mode)
