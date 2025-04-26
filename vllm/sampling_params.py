# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for text generation."""
import copy
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cached_property
from typing import Annotated, Any, Optional, Union

import msgspec
from pydantic import BaseModel

from vllm.logger import init_logger
from vllm.logits_process import LogitsProcessor
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2


class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2


# maybe make msgspec?
@dataclass
class GuidedDecodingParams:
    """One of these fields will be used to build a logit processor."""
    json: Optional[Union[str, dict]] = None
    regex: Optional[str] = None
    choice: Optional[list[str]] = None
    grammar: Optional[str] = None
    json_object: Optional[bool] = None
    """These are other options that can be set"""
    backend: Optional[str] = None
    whitespace_pattern: Optional[str] = None
    structural_tag: Optional[str] = None

    @staticmethod
    def from_optional(
        json: Optional[Union[dict, BaseModel, str]] = None,
        regex: Optional[str] = None,
        choice: Optional[list[str]] = None,
        grammar: Optional[str] = None,
        json_object: Optional[bool] = None,
        backend: Optional[str] = None,
        whitespace_pattern: Optional[str] = None,
        structural_tag: Optional[str] = None,
    ) -> Optional["GuidedDecodingParams"]:
        if all(arg is None for arg in (json, regex, choice, grammar,
                                       json_object, structural_tag)):
            return None
        # Extract json schemas from pydantic models
        if isinstance(json, (BaseModel, type(BaseModel))):
            json = json.model_json_schema()
        return GuidedDecodingParams(
            json=json,
            regex=regex,
            choice=choice,
            grammar=grammar,
            json_object=json_object,
            backend=backend,
            whitespace_pattern=whitespace_pattern,
            structural_tag=structural_tag,
        )

    @property
    def backend_name(self) -> str:
        """Return the backend name without any options.
        
        For example if the backend is "xgrammar:no-fallback", returns "xgrammar"
        """
        return (self.backend or "").split(":")[0]

    def backend_options(self) -> list[str]:
        """Return the backend options as a list of strings."""
        if not self.backend or ":" not in self.backend:
            return []
        return self.backend.split(":")[1].split(",")

    def add_option(self, opt_name: str) -> None:
        """Adds an option to the backend options."""
        if not self.backend:
            self.backend = f":{opt_name}"
        elif ":" not in self.backend:
            self.backend += f":{opt_name}"
        else:
            options = set(self.backend_options())
            options.add(opt_name)
            self.backend = f"{self.backend_name}:{','.join(sorted(options))}"

    def no_fallback(self) -> bool:
        """Returns True if the "no-fallback" option is supplied for the guided
        decoding backend"""
        return "no-fallback" in self.backend_options()

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum([
            self.json is not None, self.regex is not None, self.choice
            is not None, self.grammar is not None, self.json_object is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple are "
                f"specified: {self.__dict__}")


class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


class SamplingParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. By default,
            `best_of` is set to `n`. Warning, this is only supported in V0.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
        seed: Random seed to use for the generation.
        stop: list of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        stop_token_ids: list of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        bad_words: list of words that are not allowed to be generated.
            More precisely, only the last token of a corresponding
            token sequence is not allowed when the next generated token
            can complete the sequence.
        include_stop_str_in_output: Whether to include the stop strings in
            output text. Defaults to False.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        min_tokens: Minimum number of tokens to generate per output sequence
            before EOS or stop_token_ids can be generated
        logprobs: Number of log probabilities to return per output token.
            When set to None, no probability is returned. If set to a non-None
            value, the result includes the log probabilities of the specified
            number of most likely tokens, as well as the chosen tokens.
            Note that the implementation follows the OpenAI API: The API will
            always return the log probability of the sampled token, so there
            may be up to `logprobs+1` elements in the response.
        prompt_logprobs: Number of log probabilities to return per prompt token.
        detokenize: Whether to detokenize the output. Defaults to True.
        skip_special_tokens: Whether to skip special tokens in the output.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens in the output.  Defaults to True.
        logits_processors: list of functions that modify logits based on
            previously generated tokens, and optionally prompt tokens as
            a first argument.
        truncate_prompt_tokens: If set to an integer k, will use only the last k
            tokens from the prompt (i.e., left truncation). Defaults to None
            (i.e., no truncation).
        guided_decoding: If provided, the engine will construct a guided
            decoding logits processor from these parameters. Defaults to None.
        logit_bias: If provided, the engine will construct a logits processor
            that applies these logit biases. Defaults to None.
        allowed_token_ids: If provided, the engine will construct a logits
            processor which only retains scores for the given token ids.
            Defaults to None.
        extra_args: Arbitrary additional args, that can be used by custom
            sampling implementations. Not used by any in-tree sampling
            implementations.
    """

    n: int = 1
    best_of: Optional[int] = None
    _real_n: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stop_token_ids: Optional[list[int]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    # NOTE: This parameter is only exposed at the engine level for now.
    # It is not exposed in the OpenAI API server, as the OpenAI API does
    # not support returning only a list of token IDs.
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    # Optional[list[LogitsProcessor]] type. We use Any here because
    # Optional[list[LogitsProcessor]] type is not supported by msgspec.
    logits_processors: Optional[Any] = None
    include_stop_str_in_output: bool = False
    truncate_prompt_tokens: Optional[Annotated[int, msgspec.Meta(ge=1)]] = None
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    output_text_buffer_length: int = 0
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)

    # Fields used to construct logits processors
    guided_decoding: Optional[GuidedDecodingParams] = None
    logit_bias: Optional[dict[int, float]] = None
    allowed_token_ids: Optional[list[int]] = None
    extra_args: Optional[dict[str, Any]] = None

    # Fields used for bad words
    bad_words: Optional[list[str]] = None
    _bad_words_token_ids: Optional[list[list[int]]] = None

    @staticmethod
    def from_optional(
        n: Optional[int] = 1,
        best_of: Optional[int] = None,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        repetition_penalty: Optional[float] = 1.0,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        seed: Optional[int] = None,
        stop: Optional[Union[str, list[str]]] = None,
        stop_token_ids: Optional[list[int]] = None,
        bad_words: Optional[list[str]] = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: Optional[int] = 16,
        min_tokens: int = 0,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        logits_processors: Optional[list[LogitsProcessor]] = None,
        truncate_prompt_tokens: Optional[Annotated[int,
                                                   msgspec.Meta(ge=1)]] = None,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
        guided_decoding: Optional[GuidedDecodingParams] = None,
        logit_bias: Optional[Union[dict[int, float], dict[str, float]]] = None,
        allowed_token_ids: Optional[list[int]] = None,
        extra_args: Optional[dict[str, Any]] = None,
    ) -> "SamplingParams":
        if logit_bias is not None:
            # Convert token_id to integer
            # Clamp the bias between -100 and 100 per OpenAI API spec
            logit_bias = {
                int(token): min(100.0, max(-100.0, bias))
                for token, bias in logit_bias.items()
            }

        return SamplingParams(
            n=1 if n is None else n,
            best_of=best_of,
            presence_penalty=0.0
            if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0
            if frequency_penalty is None else frequency_penalty,
            repetition_penalty=1.0
            if repetition_penalty is None else repetition_penalty,
            temperature=1.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            bad_words=bad_words,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            detokenize=detokenize,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            logits_processors=logits_processors,
            truncate_prompt_tokens=truncate_prompt_tokens,
            output_kind=output_kind,
            guided_decoding=guided_decoding,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
            extra_args=extra_args,
        )

    def __post_init__(self) -> None:
        # how we deal with `best_of``:
        # if `best_of`` is not set, we default to `n`;
        # if `best_of`` is set, we set `n`` to `best_of`,
        # and set `_real_n`` to the original `n`.
        # when we return the result, we will check
        # if we need to return `n` or `_real_n` results
        if self.best_of:
            if self.best_of < self.n:
                raise ValueError(
                    f"best_of must be greater than or equal to n, "
                    f"got n={self.n} and best_of={self.best_of}.")
            if not self._real_n:
                self._real_n = self.n
                self.n = self.best_of

        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature, _MAX_TEMP, _MAX_TEMP)
            self.temperature = max(self.temperature, _MAX_TEMP)

        if self.seed == -1:
            self.seed = None

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]

        if self.stop_token_ids is None:
            self.stop_token_ids = []

        if self.bad_words is None:
            self.bad_words = []

        if self.logprobs is True:
            self.logprobs = 1

        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = -1
            self.min_p = 0.0
            self._verify_greedy_sampling()

        # eos_token_id is added to this by the engine
        self._all_stop_token_ids.update(self.stop_token_ids)

    def _verify_args(self) -> None:
        if not isinstance(self.n, int):
            raise ValueError(f"n must be an int, but is of "
                             f"type {type(self.n)}")
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                "repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                             f"got {self.top_k}.")
        if not isinstance(self.top_k, int):
            raise TypeError(
                f"top_k must be an integer, got {type(self.top_k).__name__}")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1], got "
                             f"{self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be greater than or equal to 0, "
                             f"got {self.min_tokens}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {self.logprobs}.")
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError(f"prompt_logprobs must be non-negative, got "
                             f"{self.prompt_logprobs}.")
        if (self.truncate_prompt_tokens is not None
                and self.truncate_prompt_tokens < 1):
            raise ValueError(f"truncate_prompt_tokens must be >= 1, "
                             f"got {self.truncate_prompt_tokens}")
        assert isinstance(self.stop, list)
        if any(not stop_str for stop_str in self.stop):
            raise ValueError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise ValueError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop.")
        if self.best_of != self._real_n and self.output_kind == (
                RequestOutputKind.DELTA):
            raise ValueError("best_of must equal n to use output_kind=DELTA")

    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise ValueError("n must be 1 when using greedy sampling, "
                             f"got {self.n}.")

    def update_from_generation_config(
            self,
            generation_config: dict[str, Any],
            model_eos_token_id: Optional[int] = None) -> None:
        """Update if there are non-default values from generation_config"""

        if model_eos_token_id is not None:
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(model_eos_token_id)

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if model_eos_token_id is not None:
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(model_eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    def update_from_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        for bad_word in self.bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()

                if isinstance(tokenizer, MistralTokenizer):
                    # Mistral tokenizers should not add special tokens
                    prompt_token_ids = tokenizer.encode(text=prompt)
                else:
                    prompt_token_ids = tokenizer.encode(
                        text=prompt, add_special_tokens=False)

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (
                        add_prefix_space and prompt_token_ids[0]
                        != self._bad_words_token_ids[-1][0]
                        and len(prompt_token_ids) == len(
                            self._bad_words_token_ids[-1])):
                    self._bad_words_token_ids.append(prompt_token_ids)

        invalid_token_ids = [
            token_id for bad_words_token_ids in self._bad_words_token_ids
            for token_id in bad_words_token_ids
            if token_id < 0 or token_id > tokenizer.max_token_id
        ]
        if len(invalid_token_ids) > 0:
            raise ValueError(
                f"The model vocabulary size is {tokenizer.max_token_id+1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}.")

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    @property
    def all_stop_token_ids(self) -> set[int]:
        return self._all_stop_token_ids

    @property
    def bad_words_token_ids(self) -> Optional[list[list[int]]]:
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids

    def clone(self) -> "SamplingParams":
        """Deep copy, but maybe not the LogitsProcessor objects.

        LogitsProcessor objects may contain an arbitrary, nontrivial amount of
        data that is expensive to copy. However, if not copied, the processor
        needs to support parallel decoding for multiple sequences
        See https://github.com/vllm-project/vllm/issues/3087
        """

        logit_processor_refs = None if self.logits_processors is None else {
            id(lp): lp.clone() if hasattr(lp, 'clone') else lp
            for lp in self.logits_processors
        }
        return copy.deepcopy(self, memo=logit_processor_refs)

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"seed={self.seed}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"bad_words={self.bad_words}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"truncate_prompt_tokens={self.truncate_prompt_tokens}, "
            f"guided_decoding={self.guided_decoding}, "
            f"extra_args={self.extra_args})")


class BeamSearchParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""
    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
