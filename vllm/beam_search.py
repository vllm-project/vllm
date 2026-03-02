# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.inputs import TokenInputs, token_inputs, EncoderDecoderInputs
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalInputs, mm_inputs, mm_enc_dec_inputs
from typing import TYPE_CHECKING, Any, cast, get_args, get_type_hints


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    orig_prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs

    # The tokens include the prompt.
    encoder_tokens: list[int] | None
    decoder_tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    def get_prompt(self):
        prompt = self.orig_prompt

        prompt_text = prompt.get("prompt")
        cache_salt = prompt.get("cache_salt")

        if prompt["type"] == "enc_dec":
            cast(EncoderDecoderInputs, prompt)
            decoder_prompt = prompt["decoder_prompt"]
            return mm_enc_dec_inputs(
                encoder_inputs=decoder_prompt, # FIXME - this doesn't look quite right?
                decoder_prompt_token_ids=self.decoder_tokens,
                decoder_prompt=decoder_prompt.get("prompt", None),
            )

        elif prompt["type"] == "token": # TODO - when we have mm inputs, this is taken on decode, right?
            return token_inputs(
                self.decoder_tokens,
                prompt=prompt_text,
                cache_salt=cache_salt,
            )

        return mm_inputs(
            prompt_token_ids=self.decoder_tokens,
            mm_kwargs=prompt["mm_kwargs"],
            mm_hashes=prompt["mm_hashes"],
            mm_placeholders=prompt["mm_placeholders"],
            prompt=prompt_text,
            cache_salt=cache_salt,
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
        prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs,
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    ):
        if prompt["type"] == "enc_dec":
            encoder_tokens = prompt["encoder_prompt"]["prompt_token_ids"]
            decoder_tokens = prompt["decoder_prompt"]["prompt_token_ids"]
        else:
            encoder_tokens = None
            decoder_tokens = prompt["prompt_token_ids"]

        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                orig_prompt=prompt,
                encoder_tokens=encoder_tokens,
                decoder_tokens=decoder_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_id, length_penalty
        )

    return sort_beams_key
