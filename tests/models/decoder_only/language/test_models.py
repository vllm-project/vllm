# SPDX-License-Identifier: Apache-2.0
"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

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
    "Qwen/Qwen-7B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "ehristoforu/Falcon3-MoE-2x7B-Insruct",
]


# @maybe_test_rocm_aiter
@pytest.mark.parametrize(
    "model_arch",
    [
        pytest.param(
            "BloomForCausalLM",  # testing alibi slopes
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "GPT2LMHeadModel",  # gpt2
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param("GPTJForCausalLM"),
        pytest.param("GPTBigCodeForCausalLM"),
        pytest.param("GPTNeoXForCausalLM"),
        pytest.param(
            "GemmaForCausalLM",  # gemma
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param("GlmForCausalLM"),
        pytest.param(
            "LlamaForCausalLM",
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "MiniCPM3ForCausalLM",
            # fused_moe not supported on CPU
            marks=[pytest.mark.core_model],
        ),
        pytest.param(
            "OPTForCausalLM",
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "PhiForCausalLM",
            marks=[pytest.mark.core_model],
        ),
        pytest.param("QWenLMHeadModel", ),
        pytest.param(
            "Qwen2ForCausalLM",
            marks=[pytest.mark.core_model],
        ),
        pytest.param("StableLmForCausalLM"),
        pytest.param("Starcoder2ForCausalLM"),
        pytest.param(
            "MixtralForCausalLM",
            marks=[pytest.mark.cpu_model],
        )
    ])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False])
def test_models(hf_runner, vllm_runner, example_prompts, model_arch: str,
                dtype: str, max_tokens: int, num_logprobs: int,
                use_rocm_aiter: bool, monkeypatch) -> None:

    model = HF_EXAMPLE_MODELS.get_hf_info(model_arch).default

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

    with hf_runner(model, dtype=dtype) as hf_model:
        if model.startswith("THUDM/chatglm3"):
            hf_model.model.get_output_embeddings = lambda: \
                hf_model.model.transformer.output_layer

        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model, dtype=dtype) as vllm_model:
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
