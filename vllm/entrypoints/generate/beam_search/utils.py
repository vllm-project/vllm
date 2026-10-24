# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.inputs import (
    DecoderOnlyEngineInput,
    EncoderDecoderInput,
    MultiModalInput,
    TokensInput,
    mm_input,
    tokens_input,
)
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def validate_beam_width(
    beam_width: int, sampling_params: SamplingParams, config: "VllmConfig"
) -> None:
    """Reject a `beam_width` the engine cannot serve at all.

    Beam search asks the engine for ``2 * beam_width`` logprobs per step to expand
    the beams. That request is exempt from ``max_logprobs``, which caps what a
    client may be *returned* rather than how widely the search may look; the
    per-step logprobs stored on each returned beam are capped separately by
    `trim_step_logprobs`.

    What remains is the one hard limit: a top-k over the vocabulary cannot ask for
    more entries than the vocabulary holds, so ``2 * beam_width <= vocab_size``.

    One soft limit is also reported. Every live beam materializes its own
    candidate mapping before trimming, so a step holds
    ``beam_width * 2 * beam_width`` `Logprob` objects -- quadratic in the width.
    Dropping the `max_logprobs` bound widens the reachable range from about 10 at
    the default config to ``VLLM_MAX_N_SEQUENCES`` (16384), where that product
    reaches ~5e8 objects. This logs a warning rather than raising: `beam_width` is
    already capped by ``VLLM_MAX_N_SEQUENCES``, which is the operator's knob for
    exactly this, and raising would refuse widths that work today.

    Call this *after* `BeamSearchParams.configure_logprobs`, which clears
    ``logprobs`` on the exact-ID path where no top-k is taken.

    Args:
        beam_width: The requested beam width.
        sampling_params: The configured per-step params, already passed through
            `configure_logprobs`.
        config: The engine config supplying the vocabulary size.

    Raises:
        ValueError: If ``2 * beam_width`` exceeds the vocabulary size.

    """
    required = sampling_params.logprobs
    if required is None:
        return

    # Resolve the vocabulary size defensively. Callers legitimately pass a
    # stubbed or mocked model config -- several existing beam-search tests do,
    # since they do not care about the vocabulary -- in which case
    # `get_vocab_size` may be absent entirely or return a non-integer. Neither
    # can bound the search, so validate nothing rather than raising
    # AttributeError or TypeError from a config the caller never meant to be
    # real.
    getter = getattr(config.model_config, "get_vocab_size", None)
    vocab_size = getter() if callable(getter) else None
    if not isinstance(vocab_size, int):
        return

    if required > vocab_size:
        raise ValueError(
            f"beam_width={beam_width} needs {required} logprobs per step "
            f"(2 x beam_width), which exceeds the vocabulary size {vocab_size}. "
            f"Lower beam_width to at most {vocab_size // 2}."
        )

    # Every live beam builds its own candidate mapping before trimming, so one
    # step holds `beam_width * 2 * beam_width` entries. Warn rather than reject:
    # `beam_width` is already capped by VLLM_MAX_N_SEQUENCES, which is the
    # operator's knob for this, and rejecting here would refuse widths that work
    # today. Surfacing the quadratic at request time is what was missing.
    aggregate = beam_width * required
    if aggregate > vocab_size:
        logger.warning_once(
            "beam_width=%d materializes about %d logprob entries per step "
            "(beam_width x 2 x beam_width), more than the %d-token vocabulary. "
            "Peak host memory grows with the square of beam_width; lower it or "
            "lower VLLM_MAX_N_SEQUENCES if the engine runs out of memory.",
            beam_width,
            aggregate,
            vocab_size,
        )


def resolve_returned_logprobs_cap(vllm_config: "VllmConfig") -> int | None:
    """How many per-step logprobs a returned beam may carry, or None for no cap.

    `max_logprobs` of -1 means uncapped. Every attribute on the way to it is read
    defensively: callers are not required to supply a fully-formed config, and
    several existing beam-search tests pass a stub that omits `model_config` or
    `max_logprobs` entirely. A stub cannot express a cap, so it yields None
    rather than an AttributeError from a config the caller never meant to be real.
    """
    model_config = getattr(vllm_config, "model_config", None)
    max_logprobs = getattr(model_config, "max_logprobs", None)
    if not isinstance(max_logprobs, int) or max_logprobs == -1:
        return None
    return max_logprobs


def trim_step_logprobs(
    step_logprobs: dict[int, "Logprob"], chosen_token_id: int, cap: int | None
) -> dict[int, "Logprob"]:
    """Cap one step's logprobs for storage on a single beam.

    Beam search requests ``2 * beam_width`` candidates per step, but every beam
    carries its step logprobs out to the caller, so what is stored stays bounded by
    ``max_logprobs`` -- the search may look wider than a client may be shown.

    Two things this must not get wrong:

    * The beam's own ``chosen_token_id`` is kept whenever the candidate set contains
      it, which at every call site it does. A naive "top ``cap``" would drop it for
      any beam whose token ranked below the cap, leaving that beam describing a step
      it did not take.
    * The result is a fresh dict. The candidate mapping is shared by every sibling
      expanded from one parent, so trimming in place would corrupt the siblings.

    Returns the chosen token plus the next highest-scoring candidates, ordered by
    descending logprob, for a total of exactly ``cap`` entries.

    ``cap`` entries, not ``cap + 1``. ``SamplingParams.logprobs`` says a response
    may hold "up to `logprobs+1` elements", which reads like the chosen token
    should be additional to the cap -- but ``max_logprobs`` is a hard ceiling on
    what is returned, and measuring ordinary generation confirms it: at
    ``max_logprobs=4`` with ``logprobs=4`` a step carries 4 entries, not 5. Beam
    search matches that rather than returning one more than the engine would.

    ``cap`` of ``None`` means no cap and the mapping is copied unchanged; ``0`` is a
    legal ``max_logprobs`` meaning nothing may be returned, and yields an empty
    mapping.
    """
    if cap is None or cap < 0:
        return dict(step_logprobs)
    if cap == 0:
        return {}
    if len(step_logprobs) <= cap:
        return dict(step_logprobs)

    ordered = sorted(
        step_logprobs.items(), key=lambda item: item[1].logprob, reverse=True
    )
    kept = [item for item in ordered[:cap] if item[0] != chosen_token_id]
    chosen = step_logprobs.get(chosen_token_id)
    if chosen is None:
        # The chosen token is absent from its own candidate set; keep the top `cap`
        # rather than inventing an entry.
        return dict(ordered[:cap])
    return dict(
        sorted(
            [(chosen_token_id, chosen), *kept[: cap - 1]],
            key=lambda item: item[1].logprob,
            reverse=True,
        )
    )


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
    num_generated_tokens: int,
    cumulative_logprob: float,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Like Transformers, the length counts the generated tokens, including a
    terminating EOS token. Adapted from

    https://github.com/huggingface/transformers/blob/09a71e5b32a747dbc7805599457e67c956ef212f/src/transformers/generation/utils.py#L3531
    """
    # An aborted beam may not have generated any tokens.
    return cumulative_logprob / (max(num_generated_tokens, 1) ** length_penalty)


def create_sort_beams_key_function(length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        # `logprobs` has one entry per generated token, including an EOS token
        # that `tokens` omits unless include_stop_str_in_output is set.
        return get_beam_search_score(len(x.logprobs), x.cum_logprob, length_penalty)

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
