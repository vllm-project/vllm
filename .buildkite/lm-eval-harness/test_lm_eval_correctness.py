# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LM eval harness on model to compare vs HF baseline computed offline.
Configs are found in configs/$MODEL.yaml

pytest -s -v test_lm_eval_correctness.py \
    --config-list-file=configs/models-small.txt \
    --tp-size=1
"""

import gc
import os
import time
from contextlib import contextmanager

import lm_eval
import pytest
import torch
import yaml
from lm_eval.api.registry import get_model
from lm_eval.utils import simple_parse_args_string

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

DEFAULT_RTOL = 0.08
ROCM_ENGINE_SHUTDOWN_TIMEOUT_S = 60.0
MEMORY_RELEASE_TIMEOUT_S = 240.0


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


def _wait_for_memory_release(gpu_memory_utilization: float) -> None:
    devices = range(torch.accelerator.device_count())
    deadline = time.monotonic() + MEMORY_RELEASE_TIMEOUT_S
    while True:
        memory = [torch.accelerator.get_memory_info(d) for d in devices]
        if all(free >= total * gpu_memory_utilization for free, total in memory):
            return
        if time.monotonic() >= deadline:
            usage = ", ".join(
                f"{d}: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free"
                for d, (free, total) in zip(devices, memory)
            )
            logger.warning(
                "GPU memory not released after %ss (%s); the next model needs %.2f",
                MEMORY_RELEASE_TIMEOUT_S,
                usage,
                gpu_memory_utilization,
            )
            return
        time.sleep(1)


def _shutdown_lm(lm) -> None:
    if not current_platform.is_rocm():
        return
    llm = getattr(lm, "model", None)
    if not isinstance(llm, LLM):
        return
    gpu_memory_utilization = (
        llm.llm_engine.vllm_config.cache_config.gpu_memory_utilization
    )
    try:
        llm.llm_engine.engine_core.shutdown(timeout=ROCM_ENGINE_SHUTDOWN_TIMEOUT_S)
    except Exception:
        logger.exception("Engine core shutdown raised; GPU memory may leak")
    del llm
    lm.model = None
    gc.collect()
    cleanup_dist_env_and_memory()
    _wait_for_memory_release(gpu_memory_utilization)


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

    moe_backend = eval_config.get("moe_backend", None)
    if moe_backend is not None:
        model_args += f"moe_backend={moe_backend},"

    if current_platform.is_rocm():
        rocm_load_strategy = eval_config.get("rocm_safetensors_load_strategy")
        if rocm_load_strategy is not None:
            model_args += f"safetensors_load_strategy={rocm_load_strategy},"

    tokenizer_mode = eval_config.get("tokenizer_mode", None)
    if tokenizer_mode is not None:
        model_args += f"tokenizer_mode={tokenizer_mode},"

    env_vars = eval_config.get("env_vars", None)
    with scoped_env_vars(env_vars):
        if current_platform.is_rocm():
            model = get_model(backend).create_from_arg_string(
                model_args,
                {"batch_size": batch_size, "max_batch_size": None, "device": None},
            )
            model_kwargs = {"metadata": simple_parse_args_string(model_args)}
        else:
            model = backend
            model_kwargs = {"model_args": model_args}
        results = lm_eval.simple_evaluate(
            model=model,
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
            **model_kwargs,
        )
    _shutdown_lm(model)
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
