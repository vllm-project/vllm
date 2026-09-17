# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.platforms import current_platform

from ..conftest import HfRunner, VllmRunner
from ..models.utils import check_outputs_equal

ATTN_BACKEND = ["ROCM_ATTN"] if current_platform.is_rocm() else ["FLASH_ATTN"]


def _fix_prompt_embed_outputs(
    vllm_outputs: list[tuple[list[int], str]],
    hf_model: HfRunner,
    example_prompts: list[str],
) -> list[tuple[list[int], str]]:
    fixed_vllm_outputs = []
    for vllm_output, hf_input, prompt in zip(
        vllm_outputs, hf_model.get_inputs(example_prompts), example_prompts
    ):
        hf_input_ids = hf_input["input_ids"].tolist()[0]
        fixed_vllm_outputs.append(
            (
                hf_input_ids + vllm_output[0][len(hf_input_ids) :],
                prompt + vllm_output[1],
            )
        )
    return fixed_vllm_outputs


def run_models(
    hf_runner,
    model: str,
    backend: str,
    max_tokens: int,
    enforce_eager: bool,
    async_scheduling: bool,
    model_executor: str,
    enable_prompt_embeds: bool,
) -> None:
    # 5042 tokens for gemma2
    # gemma2 has alternating sliding window size of 4096
    # we need a prompt with more than 4096 tokens to test the sliding window
    prompt = (
        "The following numbers of the sequence "
        + ", ".join(str(i) for i in range(1024))
        + " are:"
    )
    example_prompts = [prompt]

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
        if enable_prompt_embeds:
            with torch.no_grad():
                prompt_embeds = hf_model.get_prompt_embeddings(example_prompts)
            if model == "hmellor/tiny-random-Gemma2ForCausalLM" and (
                Version(TRANSFORMERS_VERSION) < Version("5.3.0.dev0")
            ):
                # For Gemma 1/2 models with Transformers 5.4.0+, the prompt embeddings
                # are normalised in `get_prompt_embeddings`, like Gemma 3.
                # For older versions, we need to manually normalise.
                embed_scale = hf_model.config.hidden_size**0.5
                normalizer = torch.tensor(embed_scale, dtype=prompt_embeds[0].dtype)
                prompt_embeds = [p_e * normalizer for p_e in prompt_embeds]

    with VllmRunner(
        model,
        max_model_len=8192,
        enforce_eager=enforce_eager,
        enable_prompt_embeds=enable_prompt_embeds,
        gpu_memory_utilization=0.7,
        async_scheduling=async_scheduling,
        distributed_executor_backend=model_executor,
        attention_config={"backend": backend},
    ) as vllm_model:
        if enable_prompt_embeds:
            vllm_outputs = vllm_model.generate_greedy(prompt_embeds, max_tokens)
            vllm_outputs = _fix_prompt_embed_outputs(
                vllm_outputs, hf_model, example_prompts
            )
        else:
            vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
