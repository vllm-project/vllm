# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LM eval harness on model to compare vs HF baseline computed offline.
Configs are found in configs/$MODEL.yaml

pytest -s -v test_lm_eval_correctness.py \
    --config-list-file=configs/models-small.txt \
    --tp-size=1
"""

import os
from contextlib import contextmanager

import lm_eval
import numpy as np
import yaml

DEFAULT_RTOL = 0.08


@contextmanager
def scoped_env_vars(new_env: dict[str, str]):
    if not new_env:
        # Fast path: nothing to do
        yield
        return

    old_values = {}
    new_keys = []

    try:
        for key, value in new_env.items():
            if key in os.environ:
                old_values[key] = os.environ[key]
            else:
                new_keys.append(key)
            os.environ[key] = str(value)
        yield
    finally:
        # Restore / clean up
        for key, value in old_values.items():
            os.environ[key] = value
        for key in new_keys:
            os.environ.pop(key, None)


def launch_lm_eval(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", False)
    max_model_len = eval_config.get("max_model_len", 4096)
    batch_size = eval_config.get("batch_size", "auto")
    backend = eval_config.get("backend", "vllm")
    enforce_eager = eval_config.get("enforce_eager", "true")
    kv_cache_dtype = eval_config.get("kv_cache_dtype", "auto")
    attention_backend = eval_config.get("attention_backend")
    model_args = (
        f"pretrained={eval_config['model_name']},"
        f"tensor_parallel_size={tp_size},"
        f"enforce_eager={enforce_eager},"
        f"kv_cache_dtype={kv_cache_dtype},"
        f"add_bos_token=true,"
        f"trust_remote_code={trust_remote_code},"
        f"max_model_len={max_model_len},"
        "allow_deprecated_quantization=True,"
    )
    if attention_backend:
        model_args += f"attention_backend={attention_backend},"

    env_vars = eval_config.get("env_vars", None)
    with scoped_env_vars(env_vars):
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
