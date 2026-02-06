# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.utils import length_from_prompt_token_ids_or_embeds

from .data import EmbedsPrompt, ProcessorInputs, SingletonInputs, TokensPrompt


def split_enc_dec_inputs(
    inputs: ProcessorInputs,
) -> tuple[SingletonInputs | None, SingletonInputs]:
    if "encoder" in inputs and "decoder" in inputs:
        # NOTE: This passes pyright but not mypy
        return (
            inputs["encoder"],  # type: ignore[typeddict-item]
            inputs["decoder"],  # type: ignore[typeddict-item]
        )

    return None, inputs


def get_prompt_len(prompt: TokensPrompt | EmbedsPrompt):
    return length_from_prompt_token_ids_or_embeds(
        prompt.get("prompt_token_ids"),  # type: ignore[arg-type]
        prompt.get("prompt_embeds"),  # type: ignore[arg-type]
    )
