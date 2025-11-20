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

RTOL = 0.08


def launch_lm_eval(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", False)
    max_model_len = eval_config.get("max_model_len", 4096)
    batch_size = eval_config.get("batch_size", "auto")
    backend = eval_config.get("backend", "vllm")
    enforce_eager = eval_config.get("enforce_eager", "true")
    kv_cache_dtype = eval_config.get("kv_cache_dtype", "auto")
    model_args = (
        f"pretrained={eval_config['model_name']},"
        f"tensor_parallel_size={tp_size},"
        f"enforce_eager={enforce_eager},"
        f"kv_cache_dtype={kv_cache_dtype},"
        f"add_bos_token=true,"
        f"trust_remote_code={trust_remote_code},"
        f"max_model_len={max_model_len},"
    )
    results = lm_eval.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config["num_fewshot"],
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
