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
import pytest
import yaml

from vllm.platforms import current_platform

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

    if current_platform.is_rocm() and "Nemotron-3" in eval_config["model_name"]:
        model_args += "attention_backend=TRITON_ATTN"

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


def _check_rocm_gpu_arch_requirement(eval_config):
    """Skip the test if the model requires a ROCm GPU arch not present.

    Model YAML configs can specify::

        required_gpu_arch:
          - gfx942
          - gfx950

    The check only applies on ROCm.  On other platforms (e.g. CUDA) the
    field is ignored so that shared config files work for both NVIDIA and
    AMD CI pipelines.
    """
    required_archs = eval_config.get("required_gpu_arch")
    if not required_archs:
        return

    if not current_platform.is_rocm():
        return

    from vllm.platforms.rocm import _GCN_ARCH  # noqa: E402

    if not any(arch in _GCN_ARCH for arch in required_archs):
        pytest.skip(
            f"Model requires GPU arch {required_archs}, "
            f"but detected arch is '{_GCN_ARCH}'"
        )


def test_lm_eval_correctness_param(config_filename, tp_size):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    _check_rocm_gpu_arch_requirement(eval_config)

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

            min_acceptable = ground_truth * (1 - rtol)
            success = success and measured_value >= min_acceptable

    assert success
