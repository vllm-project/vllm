# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.cache import CacheConfig
from vllm.platforms import current_platform

from ..models.utils import check_logprobs_close
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
    reason="Below SM90, FlashInfer supports full CUDA graphs for spec-decode batches",
)
@create_new_process_for_each_test()
def test_pp_cudagraph_mode_differs_across_stages(
    vllm_runner, example_prompts, monkeypatch
):
    """Pipeline stages with different CUDA graph modes match eager outputs.

    The draft model's attention layers are on the last stage only. FlashInfer
    without TRT-LLM does not support full CUDA graphs for spec-decode batches, so
    the last stage uses PIECEWISE and the first stage uses FULL decode graphs,
    which pad decode batches to different sizes. NCCL is forced because the
    copy-engine transport receives directly into the persistent tensors.
    """
    monkeypatch.setenv("VLLM_PP_TRANSPORT", "nccl")
    outputs = {}
    for enforce_eager in (True, False):
        with vllm_runner(
            "Qwen/Qwen3-8B",
            trust_remote_code=False,
            max_model_len=2048,
            enforce_eager=enforce_eager,
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
            compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
            attention_config={"backend": "TRITON_ATTN", "use_trtllm_attention": False},
            speculative_config={
                "method": "eagle3",
                "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
                "num_speculative_tokens": 2,
                "attention_backend": "FLASHINFER",
            },
        ) as llm:
            outputs[enforce_eager] = llm.generate_greedy_logprobs(
                example_prompts[:4], max_tokens=32, num_logprobs=5
            )

    check_logprobs_close(
        outputs_0_lst=outputs[True],
        outputs_1_lst=outputs[False],
        name_0="eager",
        name_1="cudagraph",
    )
