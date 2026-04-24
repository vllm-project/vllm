# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LM eval harness on model to compare vs HF baseline computed offline.
Configs are found in configs/$MODEL.yaml
"""

import json
import os
from pathlib import Path

import lm_eval
import numpy as np
import pytest
import yaml

RTOL = 0.08
TEST_DATA_FILE = os.environ.get("TEST_DATA_FILE")
if TEST_DATA_FILE is None:
    raise ValueError("TEST_DATA_FILE environment variable must be set")
DUMP_RESULTS_DIR = os.environ.get("LM_EVAL_DUMP_RESULTS_DIR", ".")
# Auto-generate dump path based on config filename
config_name = Path(TEST_DATA_FILE).stem  # e.g., "Command-R7B-FP8-mbpp"
DUMP_RESULTS_FILE = Path(DUMP_RESULTS_DIR) / f"{config_name}_results.json"

TP_SIZE = os.environ.get("LM_EVAL_TP_SIZE", 1)


def launch_lm_eval(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", "false")
    max_model_len = eval_config.get("max_model_len", 132_000)
    max_num_batched_tokens = eval_config.get("max_num_batched_tokens", 8192)
    batch_size = eval_config.get("batch_size", "auto")
    backend = eval_config.get("backend", "vllm")
    enforce_eager = eval_config.get("enforce_eager", "true")
    kv_cache_dtype = eval_config.get("kv_cache_dtype", "auto")
    enable_prefix_caching = eval_config.get("enable_prefix_caching", "true")
    mm_processor_cache_type = eval_config.get("mm_processor_cache_type", "shm")
    gpu_memory_utilization = eval_config.get("gpu_memory_utilization", 0.95)
    model_args = (
        f"pretrained={eval_config['model_name']},"
        f"tensor_parallel_size={tp_size},"
        f"enforce_eager={enforce_eager},"
        f"kv_cache_dtype={kv_cache_dtype},"
        f"add_bos_token=true,"
        f"trust_remote_code={trust_remote_code},"
        f"max_model_len={max_model_len},"
        f"max_num_batched_tokens={max_num_batched_tokens},"
        f"enable_prefix_caching={enable_prefix_caching},"
        f"mm_processor_cache_type={mm_processor_cache_type},"
        f"gpu_memory_utilization={gpu_memory_utilization},"
    )
    results = lm_eval.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config.get("num_fewshot"),
        limit=eval_config["limit"],
        # TODO(yeq): using chat template w/ fewshot_as_multiturn is supposed help
        # text models. however, this is regressing measured strict-match for
        # existing text models in CI, so only apply it for mm, or explicitly set
        apply_chat_template=eval_config.get(
            "apply_chat_template", backend == "vllm-vlm"
        ),
        fewshot_as_multiturn=eval_config.get("fewshot_as_multiturn", False),
        # Forward decoding and early-stop controls (e.g., max_gen_toks, until=...)
        gen_kwargs=eval_config.get("gen_kwargs"),
        batch_size=batch_size,
        metadata=eval_config.get("metadata", {}),
        confirm_run_unsafe_code=True,
        log_samples=True,
    )
    return results


def dump_results(results, output_path):
    """Dump lm_eval results to a JSON file for inspection.

    The results contain 'samples' with model generations when log_samples=True
    is passed to simple_evaluate, otherwise just aggregated metrics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON-serializable format
    serializable_results = {}
    for key, value in results.items():
        if key == "samples":
            # samples contains the actual model generations per task
            serializable_results[key] = value
        elif (
            key == "results"
            or key == "configs"
            or isinstance(value, (str, int, float, bool, list, dict, type(None)))
        ):
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"Results dumped to {output_path}")


def test_lm_eval_correctness(record_property):
    assert TEST_DATA_FILE is not None  # validated at module load time
    eval_config = yaml.safe_load(Path(TEST_DATA_FILE).read_text(encoding="utf-8"))

    if (
        eval_config["model_name"]
        == "nm-testing/Meta-Llama-3-70B-Instruct-FBGEMM-nonuniform"
    ):  # noqa: E501
        pytest.skip("FBGEMM is currently failing on main.")

    # Launch eval requests.
    results = launch_lm_eval(eval_config, TP_SIZE)

    # Dump results to file
    dump_results(results, DUMP_RESULTS_FILE)

    # Confirm scores match ground truth.
    success = True
    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            task_results = results["results"][task["name"]]
            print(
                f"Evaluating task {task['name']} metric {metric['name']}, "
                f"results: {task_results}"
            )
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(
                f"{task['name']} | {metric['name']}: "
                f"ground_truth={ground_truth} | measured={measured_value}"
            )
            success = success and np.isclose(ground_truth, measured_value, rtol=RTOL)
            record_property(
                f"{task['name']} | {metric['name']}",
                f"ground_truth={ground_truth} | measured={measured_value}",
            )
    # Assert at the end, print all scores even on failure for debugging.
    assert success
