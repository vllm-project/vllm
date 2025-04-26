# SPDX-License-Identifier: Apache-2.0
"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

from ....utils import large_gpu_mark
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

# These have unsupported head_dim for FA. We do not
# not have a clean way to fall back, so we fail with
# a clear msg when it happens.
# https://github.com/vllm-project/vllm/issues/14524
REQUIRES_V0 = ["microsoft/phi-2", "stabilityai/stablelm-3b-4e1t"]

# This list contains the model that are using AITER kernel.
# Skip model that are not using AITER tests.
# When more AITER kernels are added, this list will not be
# needed as all the models will be calling AITER kernels
# in parts of the operators
AITER_MODEL_LIST = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "openbmb/MiniCPM3-4B",
    "Qwen/Qwen-7B-Chat",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "ehristoforu/Falcon3-MoE-2x7B-Insruct",
]


# @maybe_test_rocm_aiter
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "bigscience/bloom-560m",  # bloom - testing alibi slopes
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openai-community/gpt2",  # gpt2
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param("Milos/slovak-gpt-j-405M"),  # gptj
        pytest.param("bigcode/tiny_starcoder_py"),  # gpt_bigcode
        pytest.param("EleutherAI/pythia-70m"),  # gpt_neox
        pytest.param(
            "google/gemma-1.1-2b-it",  # gemma
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "THUDM/chatglm3-6b",  # chatglm (text-only)
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",  # llama
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openbmb/MiniCPM3-4B",
            # fused_moe not supported on CPU
            marks=[pytest.mark.core_model,
                   large_gpu_mark(min_gb=32)],
        ),
        pytest.param(
            "facebook/opt-125m",  # opt
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "microsoft/phi-2",  # phi
            marks=[pytest.mark.core_model],
        ),
        pytest.param(
            "Qwen/Qwen-7B-Chat",  # qwen (text-only)
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",  # qwen2
            marks=[pytest.mark.core_model],
        ),
        pytest.param("stabilityai/stablelm-3b-4e1t"),  # stablelm
        pytest.param("bigcode/starcoder2-3b"),  # starcoder2
        pytest.param(
            "ehristoforu/Falcon3-MoE-2x7B-Insruct",  # mixtral
            marks=[pytest.mark.cpu_model,
                   large_gpu_mark(min_gb=48)],
        )
    ])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False])
def test_models(hf_runner, vllm_runner, example_prompts, model: str,
                max_tokens: int, num_logprobs: int, use_rocm_aiter: bool,
                monkeypatch) -> None:

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    if model in REQUIRES_V0:
        monkeypatch.setenv("VLLM_USE_V1", "0")

    if use_rocm_aiter and (model in AITER_MODEL_LIST):
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    elif use_rocm_aiter and model not in AITER_MODEL_LIST:
        # Skip model that are not using AITER tests.
        # When more AITER kernels are added, this list will not be
        # needed as all the models will be calling AITER kernels
        # in parts of the operators
        pytest.skip(f"Skipping '{model}' model test with AITER kernel.")

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(
            model,
            tokenizer_name=model_info.tokenizer or model,
            tokenizer_mode=model_info.tokenizer_mode,
            trust_remote_code=model_info.trust_remote_code,
            max_num_seqs=2,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
    if use_rocm_aiter:
        # this is to ensure that vllm engine
        # has deallocated the memory before running the next
        # unit tests. On ROCm, when using AITER
        # the memory might not be deallocated completely
        # before running the next test case
        torch.cuda.synchronize()
