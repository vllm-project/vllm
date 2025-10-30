# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

DEFAULT_RTOL = 0.08


def launch_lm_eval(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", False)
    max_model_len = eval_config.get("max_model_len", 4096)
    batch_size = eval_config.get("batch_size", "auto")
    backend = eval_config.get("backend", "vllm")

    model_args_list = [
        f"pretrained={eval_config['model_name']}",
        f"tensor_parallel_size={tp_size}",
        "enforce_eager=true",
        "add_bos_token=true",
        f"trust_remote_code={trust_remote_code}",
        f"max_model_len={max_model_len}",
    ]

    if "vllm_args" in eval_config:
        for key, value in eval_config["vllm_args"].items():
            if isinstance(value, bool):
                value = str(value).lower()
            model_args_list.append(f"{key}={value}")

    model_args = ",".join(model_args_list)

    results = lm_eval.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
        # TODO(yeq): using chat template w/ fewshot_as_multiturn is supposed help
        # text models. however, this is regressing measured strict-match for
        # existing text models in CI, so only apply it for mm.
        apply_chat_template=backend == "vllm-vlm",
        batch_size=batch_size,
    )
    return results


def test_lm_eval_correctness_param(config_filename, tp_size):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    results = launch_lm_eval(eval_config, tp_size)

    rtol = eval_config.get("rtol", DEFAULT_RTOL)

    success = True
    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(
                f"{task['name']} | {metric['name']}: "
                f"ground_truth={ground_truth:.3f} | "
                f"measured={measured_value:.3f} | rtol={rtol}"
            )
            success = success and np.isclose(ground_truth, measured_value, rtol=rtol)

    assert success
