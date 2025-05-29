# SPDX-License-Identifier: Apache-2.0
"""
LM eval harness on model to compare vs HF baseline computed offline.
Configs are found in configs/$MODEL.yaml

pytest -s -v test_lm_eval_correctness.py \
    --config-list-file=configs/models-small.txt \
    --tp-size=1
"""

import lm_eval
import numpy as np
import yaml

RTOL = 0.08


def launch_lm_eval(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", False)
    model_args = (
        f"pretrained={eval_config['model_name']},"
        f"tensor_parallel_size={tp_size},"
        f"enforce_eager=true,"
        f"add_bos_token=true,"
        f"trust_remote_code={trust_remote_code}"
    )
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
        batch_size="auto",
    )
    return results


def test_lm_eval_correctness_param(config_filename, tp_size):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    results = launch_lm_eval(eval_config, tp_size)

    success = True
    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(
                f"{task['name']} | {metric['name']}: "
                f"ground_truth={ground_truth} | measured={measured_value}"
            )
            success = success and np.isclose(ground_truth, measured_value, rtol=RTOL)

    assert success
