# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import (
    compare_two_settings,
    create_new_process_for_each_test,
)
from vllm.config import (
    CompilationMode,
)


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_id",
    ["meta-llama/Llama-3.2-1B-Instruct", "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"],
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("async_tp_enabled", [True])
@pytest.mark.parametrize("distributed_backend", ["mp"])
@pytest.mark.parametrize("eager_mode", [False, True])
def test_async_tp_pass_correctness(
    model_id: str,
    tp_size: int,
    async_tp_enabled: bool,
    distributed_backend: str,
    eager_mode: bool,
    num_gpus_available: int,
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")
    model_info.check_available_online(on_fail="skip")

    pp_size = 1
    if num_gpus_available < tp_size:
        pytest.skip(f"Need at least {tp_size} x {pp_size} GPUs")

    common_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
    ]
    if eager_mode:
        common_args.append("--enforce-eager")

    compilation_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "compile_sizes": [2, 4, 8],
        "splitting_ops": [],
        "pass_config": {"fuse_gemm_comms": async_tp_enabled},
    }

    async_tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        distributed_backend,
        "--compilation_config",
        json.dumps(compilation_config),
    ]

    tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        "mp",
    ]

    compare_two_settings(model_id, async_tp_args, tp_args, method="generate")
