# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.platforms import current_platform

from ....utils import large_gpu_mark
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

# Models that require embedding scaling for prompt_embeds test
EMBED_SCALING_MODELS = {
    "openbmb/MiniCPM4.1-8B",
}

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
    "TitanML/tiny-mixtral",
    "Qwen/Qwen3-8B",
]


# @maybe_test_rocm_aiter
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "bigscience/bloom-560m",  # bloom - testing alibi slopes
            marks=[
                pytest.mark.core_model,
                pytest.mark.slow_test,
                pytest.mark.cpu_model,
                pytest.mark.skipif(
                    current_platform.is_zen_cpu(),
                    reason="bloom-560m ALiBi is currently not supported on\
                        AMD Zen CPUs due to lack of support for float16 compute.",
                ),
            ],
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
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
        pytest.param(
            "google/gemma-2-2b-it",  # test hybrid attention
            marks=[pytest.mark.cpu_model],
        ),
        pytest.param(
            "zai-org/chatglm3-6b",  # chatglm (text-only)
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",  # llama
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openbmb/MiniCPM4.1-8B",  # minicpm
            marks=[pytest.mark.core_model, large_gpu_mark(min_gb=48)],
        ),
        pytest.param(
            "facebook/opt-125m",  # opt
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "microsoft/phi-2",  # phi
            marks=[pytest.mark.core_model, pytest.mark.slow_test],
        ),
        pytest.param(
            "Qwen/Qwen-7B-Chat",  # qwen (text-only)
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",  # qwen2
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
        pytest.param(
            "Qwen/Qwen3-8B",  # qwen (text-only)
        ),
        pytest.param("stabilityai/stablelm-3b-4e1t"),  # stablelm
        pytest.param("bigcode/starcoder2-3b"),  # starcoder2
        pytest.param(
            "TitanML/tiny-mixtral",  # mixtral
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param("swiss-ai/Apertus-8B-Instruct-2509"),  # apertus
        pytest.param(
            "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",  # hyperclovax
            marks=[large_gpu_mark(min_gb=32)],
        ),
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
@pytest.mark.parametrize("use_prompt_embeds", [True, False])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    use_rocm_aiter: bool,
    use_prompt_embeds: bool,
    monkeypatch,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    if use_rocm_aiter and (model in AITER_MODEL_LIST):
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        if model == "TitanML/tiny-mixtral":
            # Untrained model: near-uniform logits make argmax sensitive to
            # AITER's bfloat16 rounding error in plain rms_norm.
            monkeypatch.setenv("VLLM_ROCM_USE_AITER_RMSNORM", "0")
    elif use_rocm_aiter and model not in AITER_MODEL_LIST:
        # Skip model that are not using AITER tests.
        # When more AITER kernels are added, this list will not be
        # needed as all the models will be calling AITER kernels
        # in parts of the operators
        pytest.skip(f"Skipping '{model}' model test with AITER kernel.")

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

        prompt_embeds: list[torch.Tensor] | None = [] if use_prompt_embeds else None

        for prompt in example_prompts:
            token_ids = hf_model.tokenizer(prompt, return_tensors="pt").input_ids.to(
                hf_model.model.device
            )
            if prompt_embeds is not None:
                embed = hf_model.model.get_input_embeddings()(token_ids)

                if "gemma" in model.lower() and (
                    Version(TRANSFORMERS_VERSION) < Version("5.3.0.dev0")
                ):
                    # For Gemma 1/2 models with Transformers 5.4.0+, the prompt
                    # embeddings are normalised in `get_prompt_embeddings`,
                    # like Gemma 3. For older versions, we need to manually normalise.
                    embed_scale = hf_model.config.hidden_size**0.5
                    normalizer = torch.tensor(embed_scale, dtype=embed.dtype)
                    embed *= normalizer

                # MiniCPM models apply scale_emb to embeddings internally.
                # vLLM expects pre-scaled embeddings when using inputs_embeds.
                if model in EMBED_SCALING_MODELS:
                    config = hf_model.model.config
                    embed = embed * config.scale_emb

                prompt_embeds.append(embed.squeeze(0))

    with vllm_runner(
        model,
        tokenizer_name=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        # Remove the effects of batch variance on ROCm since batch invariance
        # is not yet supported.
        # See: https://github.com/vllm-project/vllm/issues/27433
        max_num_seqs=1 if current_platform.is_rocm() else 2,
        enable_prompt_embeds=use_prompt_embeds,
        compilation_config={"cudagraph_capture_sizes": [1, 2]},
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )
        if prompt_embeds is not None:
            vllm_outputs_from_embeds = vllm_model.generate_greedy_logprobs(
                prompt_embeds, max_tokens, num_logprobs
            )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
    if prompt_embeds is not None:
        check_logprobs_close(
            outputs_0_lst=vllm_outputs,
            outputs_1_lst=vllm_outputs_from_embeds,
            name_0="vllm",
            name_1="vllm_from_embeds",
        )

    if use_rocm_aiter:
        # this is to ensure that vllm engine
        # has deallocated the memory before running the next
        # unit tests. On ROCm, when using AITER
        # the memory might not be deallocated completely
        # before running the next test case
        torch.accelerator.synchronize()
