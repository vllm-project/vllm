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
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

NVFP4_MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
NVFP4_HF_OVERRIDES = {
    "num_hidden_layers": 4,
    "hidden_size": 512,
    "intermediate_size": 800,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
}


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
    monkeypatch,
):
    # Disable FlashInfer FP8 scaled_mm kernel as it is incompatible with
    # async TP patterns. No-op on H100 (kernel requires CC >= 100).
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")

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


@create_new_process_for_each_test()
def test_async_tp_pass_nvfp4_correctness(num_gpus_available: int, monkeypatch):
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
    ):
        pytest.skip("NVFP4 requires Blackwell")
    if not has_flashinfer():
        pytest.skip("FlashInfer is required for the NVFP4 AsyncTP path")

    monkeypatch.setenv("VLLM_NVFP4_GEMM_BACKEND", "flashinfer-cutlass")

    tp_size = 2
    if num_gpus_available < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs")

    common_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps(NVFP4_HF_OVERRIDES),
    ]

    compilation_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "compile_sizes": [2, 4, 8],
        "splitting_ops": [],
        "pass_config": {
            "enable_sp": True,
            "fuse_gemm_comms": True,
            "fuse_allreduce_rms": False,
            "sp_min_token_num": 1,
        },
    }

    async_tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        "mp",
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

    compare_two_settings(NVFP4_MODEL_ID, async_tp_args, tp_args, method="generate")
