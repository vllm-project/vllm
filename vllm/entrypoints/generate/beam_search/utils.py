# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

from vllm.inputs import (
    DecoderOnlyEngineInput,
    EncoderDecoderInput,
    MultiModalInput,
    TokensInput,
    mm_input,
    tokens_input,
)
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    orig_prompt: TokensInput | MultiModalInput | EncoderDecoderInput

    # NOTE: Tokens represents decoder tokens in the encoder / decoder case
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    def get_prompt(self):
        prompt = self.orig_prompt

        if prompt["type"] == "enc_dec":
            return self._build_encoder_decoder_inputs(prompt)

        # Handle decoder-only inputs
        prompt_text = prompt.get("prompt")
        cache_salt = prompt.get("cache_salt")

        if prompt["type"] == "token":
            return tokens_input(
                self.tokens,
                prompt=prompt_text,
                cache_salt=cache_salt,
            )

        return mm_input(
            prompt_token_ids=self.tokens,
            mm_kwargs=prompt["mm_kwargs"],
            mm_hashes=prompt["mm_hashes"],
            mm_placeholders=prompt["mm_placeholders"],
            prompt=prompt_text,
            cache_salt=cache_salt,
        )

    def _build_encoder_decoder_inputs(
        self, prompt: EncoderDecoderInput
    ) -> EncoderDecoderInput:
        """Rebuild the encoder-decoder inputs with the current beam search
        sequence's tokens.

        FIXME (alex) - the encoder multimodal cache is not properly wired up
        yet, which means that currently we are running the encoder on every
        new beam because num_computed_tokens is 0 on each new request. This
        will be fixed once the cache is correctly implemented.
        """
        dec_prompt = prompt["decoder_prompt"]

        # Rebuild decoder prompt with updated tokens,
        # but keep everything else the same.
        new_dec_prompt: DecoderOnlyEngineInput
        if dec_prompt["type"] == "multimodal":
            new_dec_prompt = mm_input(
                self.tokens,
                mm_kwargs=dec_prompt["mm_kwargs"],
                mm_hashes=dec_prompt["mm_hashes"],
                mm_placeholders=dec_prompt["mm_placeholders"],
                prompt=dec_prompt.get("prompt"),
                cache_salt=dec_prompt.get("cache_salt"),
            )
        else:
            new_dec_prompt = tokens_input(
                self.tokens,
                prompt=dec_prompt.get("prompt"),
                cache_salt=dec_prompt.get("cache_salt"),
            )

        return EncoderDecoderInput(
            type="enc_dec",
            encoder_prompt=prompt["encoder_prompt"],
            decoder_prompt=new_dec_prompt,
        )


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """

    sequences: list[BeamSearchSequence]


class BeamSearchInstance:
    def __init__(
        self,
        prompt: TokensInput | MultiModalInput | EncoderDecoderInput,
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        priority: int = 0,
        **kwargs,
    ):
        decoder_prompt = (
            prompt if prompt["type"] != "enc_dec" else prompt["decoder_prompt"]
        )
        initial_tokens = decoder_prompt["prompt_token_ids"]

        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                orig_prompt=prompt,
                tokens=initial_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []
        self.priority = priority


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_ids: set[int],
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] in eos_token_ids:
        seq_len -= 1

    # An aborted beam may contain only an EOS prompt token.
    return cumulative_logprob / (max(seq_len, 1) ** length_penalty)


def create_sort_beams_key_function(eos_token_ids: set[int], length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_ids, length_penalty
        )

    return sort_beams_key


def get_beam_search_eos_token_ids(
    generation_config: dict[str, Any], eos_token_id: int | None
) -> set[int]:
    """Return the token ids that end generation: the tokenizer EOS plus the
    generation config's ``eos_token_id``.

    This uses a separate SamplingParams so the params sent to the engine keep
    their own ``stop_token_ids`` and are validated as before.
    """
    params = SamplingParams()
    params.update_from_generation_config(generation_config, eos_token_id)
    return params.all_stop_token_ids
