# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging

import pytest

from tests.compile.fusions_e2e.common import FUSION_LOG_PATTERNS
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import (
    compare_two_settings,
    create_new_process_for_each_test,
    multi_gpu_test,
)
from vllm.config import (
    CompilationMode,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.flashinfer import has_flashinfer

NVFP4_MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
NVFP4_HF_OVERRIDES = {
    "num_hidden_layers": 4,
    "hidden_size": 512,
    "intermediate_size": 800,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
}


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm AsyncTP contract")
def test_rocm_async_tp_bf16_output_correctness(caplog_mp_spawn):
    """Trained-model outputs must agree with unfused TP, including padded decode."""
    # A snapshot path pins both servers and the comparison helper's tokenizer.
    model = hf_api().snapshot_download(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    common_args = [
        "--dtype",
        "bfloat16",
        "--load-format",
        "auto",
        "--tensor-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "8",
        "--kv-cache-memory-bytes",
        str(64 * 1024**2),
        "--gpu-memory-utilization",
        "0.01",
        "--attention-backend",
        "TRITON_ATTN",
        "--disable-custom-all-reduce",
    ]

    def settings(enabled):
        return [
            *common_args,
            "--compilation-config",
            json.dumps(
                {
                    "mode": CompilationMode.VLLM_COMPILE,
                    "cudagraph_mode": "NONE",
                    "splitting_ops": [],
                    "inductor_compile_config": {"force_disable_caches": True},
                    "pass_config": {
                        "enable_sp": enabled,
                        "fuse_gemm_comms": enabled,
                        "fuse_allreduce_rms": False,
                        "sp_min_token_num": 1,
                    },
                }
            ),
        ]

    env = {"VLLM_ROCM_USE_AITER": "0", "VLLM_DISABLE_COMPILE_CACHE": "1"}
    with caplog_mp_spawn(logging.DEBUG) as logs:
        compare_two_settings(
            model,
            settings(False),
            settings(True),
            env1=env,
            env2=env,
            method="generate",
            force_v1_runner=True,
        )
    matches = [
        int(count) for count in FUSION_LOG_PATTERNS["async_tp"].findall(logs.text)
    ]
    print(f"AsyncTP rewrite counts: {matches}")
    assert len(matches) >= 2 and all(count > 0 for count in matches), (
        f"Expected positive AsyncTP rewrites on both TP workers, found {matches}"
    )


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model_id",
    ["meta-llama/Llama-3.2-1B-Instruct", "RedHatAI/Llama-3.2-1B-Instruct-FP8"],
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("async_tp_enabled", [True])
@pytest.mark.parametrize("distributed_backend", ["mp"])
def test_async_tp_pass_correctness(
    model_id: str,
    tp_size: int,
    async_tp_enabled: bool,
    distributed_backend: str,
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

    compare_two_settings(
        model_id,
        async_tp_args,
        tp_args,
        method="generate",
        force_v1_runner=True,
    )


@create_new_process_for_each_test()
def test_async_tp_pass_nvfp4_correctness(num_gpus_available: int):
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
    ):
        pytest.skip("NVFP4 requires Blackwell")
    if not has_flashinfer():
        pytest.skip("FlashInfer is required for the NVFP4 AsyncTP path")

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
        "--linear-backend",
        "flashinfer_cutlass",
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

    compare_two_settings(
        NVFP4_MODEL_ID,
        async_tp_args,
        tp_args,
        method="generate",
        force_v1_runner=True,
    )
