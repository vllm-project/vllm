# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from vllm.config.cache import CacheConfig
from vllm.platforms import current_platform

from ..utils import (
    compare_two_settings,
    create_new_process_for_each_test,
    wait_for_memory_to_settle,
)


@pytest.mark.parametrize(
    "PP_SIZE, MODEL_NAME",
    [
        (2, "vllm-project/llama-160m"),
    ],
)
@pytest.mark.parametrize(
    "ATTN_BACKEND",
    [None] if current_platform.is_rocm() else ["FLASH_ATTN"],
)
@create_new_process_for_each_test()
def test_pp_cudagraph(
    PP_SIZE: int,
    MODEL_NAME: str,
    ATTN_BACKEND: str | None,
):
    if current_platform.is_rocm():
        # ROCm CI can hand this test a device that is still lazily
        # releasing VRAM from a previous process; wait before starting.
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - CacheConfig.DEFAULT_GPU_MEMORY_UTILIZATION
        )

    cudagraph_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--distributed-executor-backend",
        "mp",
    ]
    # On ROCm, defer to the platform attention selector instead of forcing a backend.
    if ATTN_BACKEND is not None:
        cudagraph_args.append(f"--attention-backend={ATTN_BACKEND}")

    eager_args = cudagraph_args + ["--enforce-eager"]

    compare_two_settings(MODEL_NAME, eager_args, cudagraph_args)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="Below SM90, FlashInfer captures spec-decode batches without TRT-LLM",
)
@create_new_process_for_each_test()
def test_pp_cudagraph_mode_differs_across_stages():
    """Each stage resolves its CUDA graph mode from its own attention layers.

    Only the last stage holds the draft. Its FlashInfer attention cannot capture
    spec-decode batches without TRT-LLM, so that stage falls back to PIECEWISE
    while the first stage keeps FULL decode graphs. The last stage then pads
    decode batches beyond the rows the first stage sends.
    """
    speculative_config = {
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 2,
        "attention_backend": "FLASHINFER",
    }
    common_args = [
        "--pipeline-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--max-model-len",
        "2048",
        "--attention-backend",
        "TRITON_ATTN",
        "--attention-config",
        json.dumps({"use_trtllm_attention": False}),
        "--speculative-config",
        json.dumps(speculative_config),
        "--load-format",
        "auto",
    ]
    eager_args = common_args + ["--enforce-eager"]

    compare_two_settings(
        "Qwen/Qwen3-8B", eager_args, common_args, include_seeded_sampling=False
    )
