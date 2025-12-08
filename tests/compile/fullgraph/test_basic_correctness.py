# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses

import pytest
import torch

from vllm.config import CompilationMode
from vllm.platforms import current_platform

from ...utils import compare_all_settings

ATTN_BACKEND = "FLASH_ATTN" if not current_platform.is_rocm() else "ROCM_ATTN"


@dataclasses.dataclass
class TestSetting:
    model: str
    model_args: list[str]
    pp_size: int
    tp_size: int
    attn_backend: str
    method: str


# we cannot afford testing the full Cartesian product
# of all models and all modes
@pytest.mark.parametrize(
    "test_setting",
    [
        # basic llama model
        TestSetting(
            model="meta-llama/Llama-3.2-1B-Instruct",
            model_args=["--max-model-len", "2048"],
            pp_size=2,
            tp_size=2,
            attn_backend=ATTN_BACKEND,
            method="generate",
        ),
        # llama model with quantization
        TestSetting(
            model="TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
            model_args=["--quantization", "gptq", "--max-model-len", "2048"],
            pp_size=1,
            tp_size=1,
            attn_backend=ATTN_BACKEND,
            method="generate",
        ),
        # MoE model
        TestSetting(
            model="ibm/PowerMoE-3b",
            model_args=["--max-model-len", "2048"],
            pp_size=1,
            tp_size=2,
            attn_backend=ATTN_BACKEND,
            method="generate",
        ),
        # embedding model
        TestSetting(
            model="BAAI/bge-multilingual-gemma2",
            model_args=[
                "--runner",
                "pooling",
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "2048",
            ],
            pp_size=1,
            tp_size=1,
            attn_backend=ATTN_BACKEND,
            method="encode",
        ),
        pytest.param(
            TestSetting(
                model="BAAI/bge-base-en-v1.5",
                model_args=["--runner", "pooling"],
                pp_size=1,
                tp_size=1,
                attn_backend="FLASH_ATTN",
                method="encode",
            ),
            marks=pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="Encoder self-attention is not implemented for ROCm",
            ),
        ),
        # vision language model
        # See https://github.com/vllm-project/vllm/issues/26716.
        # TestSetting(
        #     model="microsoft/Phi-3.5-vision-instruct",
        #     model_args=["--trust-remote-code", "--max-model-len", "2048"],
        #     pp_size=2,
        #     tp_size=1,
        #     attn_backend="FLASH_ATTN",
        #     method="generate_with_image",
        # ),
    ],
)
def test_compile_correctness(
    test_setting: TestSetting,
):
    # this test is run under multiple suits, with different GPUs.
    # make sure we only run the test with correct CUDA devices.
    # don't use "<", as it will duplicate the tests.
    model = test_setting.model
    model_args = test_setting.model_args
    pp_size = test_setting.pp_size
    tp_size = test_setting.tp_size
    attn_backend = test_setting.attn_backend
    method = test_setting.method
    gpu_nums = current_platform.device_count()
    if gpu_nums < pp_size * tp_size:
        pytest.skip(
            f"Need at least {pp_size}*{tp_size} gpus but got "
            f"{gpu_nums}"
        )

    final_args = [
        *model_args,
        "-pp",
        str(pp_size),
        "-tp",
        str(tp_size),
        "-cc.cudagraph_mode=none",
        f"--attention-backend={attn_backend}",
    ]

    all_args: list[list[str]] = []
    all_envs: list[dict[str, str] | None] = []

    for comp_mode in [
        CompilationMode.STOCK_TORCH_COMPILE,
        CompilationMode.DYNAMO_TRACE_ONCE,
        CompilationMode.VLLM_COMPILE,
    ]:
        for mode in [CompilationMode.NONE, comp_mode]:
            all_args.append(
                final_args + [f"-cc.mode={mode.name}", "-cc.backend=inductor"]
            )

        # inductor will change the output, so we only compare if the output
        # is close, not exactly the same.
        compare_all_settings(
            model,
            all_args,
            all_envs,
            method=method if method != "generate" else "generate_close",
        )
        all_envs.clear()
        all_args.clear()

    for mode in [
        CompilationMode.NONE,
        CompilationMode.STOCK_TORCH_COMPILE,
        CompilationMode.DYNAMO_TRACE_ONCE,
        CompilationMode.VLLM_COMPILE,
    ]:
        all_args.append(final_args + [f"-cc.mode={mode.name}", "-cc.backend=eager"])
        all_envs.append({})
        all_envs.append({})

    compare_all_settings(model, all_args * 3, all_envs, method=method)
