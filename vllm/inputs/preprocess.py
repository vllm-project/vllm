# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import Any

from typing_extensions import assert_never

from vllm.config import ModelConfig, ObservabilityConfig
from vllm.inputs.parse import split_enc_dec_prompt
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.cache import BaseMultiModalProcessorCache
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalEncDecInputs,
    MultiModalInputs,
    MultiModalUUIDDict,
)
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.renderers import renderer_from_config
from vllm.tokenizers import TokenizerLike
from vllm.utils.jsontree import json_iter_leaves
from vllm.v1.metrics.stats import MultiModalCacheStats

from .data import (
    DecoderOnlyInputs,
    EmbedsInputs,
    EmbedsPrompt,
    EncoderDecoderInputs,
    ProcessorInputs,
    PromptType,
    SingletonInputs,
    SingletonPrompt,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
    embeds_inputs,
    token_inputs,
)
from .parse import is_explicit_encoder_decoder_prompt, parse_singleton_prompt

logger = init_logger(__name__)


class InputPreprocessor:
    def __init__(
        self,
        model_config: ModelConfig,
        observability_config: ObservabilityConfig | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        mm_processor_cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.observability_config = observability_config
        self.renderer = renderer_from_config(model_config)
        self.mm_registry = mm_registry
        self.mm_processor_cache = mm_processor_cache

        self.mm_cache_stats = MultiModalCacheStats() if mm_processor_cache else None

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def get_bos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for BOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.bos_token_id

    def get_eos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for EOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.eos_token_id

    def get_decoder_start_token_id(self) -> int:
        """
        Obtain the decoder start token id employed by an encoder/decoder
        model. Raises an error if it is not available.
        """
        dec_start_token_id = getattr(
            self.model_config.hf_config, "decoder_start_token_id", None
        )

        if dec_start_token_id is None:
            logger.warning_once(
                "Falling back on <BOS> for decoder start token "
                "id because decoder start token id is not "
                "available."
            )
            dec_start_token_id = self.get_bos_token_id()

        if dec_start_token_id is None:
            raise RuntimeError("Cannot find decoder start token id")

        return dec_start_token_id

    def _prepare_decoder_input_ids(self, decoder_input_ids: list[int]) -> list[int]:
        """
        Prepares `decoder_input_ids` for generation with encoder-decoder models.

        Based on:
        https://github.com/huggingface/transformers/blob/4037a2b5b1278736e566aec12e169100275545ea/src/transformers/generation/utils.py
        specifically,
        `GenerationMixin._prepare_decoder_input_ids_for_generation()`.

        Arguments:

        * decoder_input_ids: input token ids to preprocess

        Returns:

        * Processed token list
        """
        decoder_start_token_id = self.get_decoder_start_token_id()

        if (
            len(decoder_input_ids) == 0
            or decoder_input_ids[0] != decoder_start_token_id
        ):
            decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

        return decoder_input_ids

    def _get_tokenization_kw(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs = dict[str, Any]()

        if self.model_config.is_encoder_decoder:
            # For Whisper, special tokens should be provided by the user based
            # on the task and language of their request. Also needed to avoid
            # appending an EOS token to the prompt which disrupts generation.
            kwargs["add_special_tokens"] = False

        if overrides:
            kwargs.update(overrides)

        return kwargs

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        tokenizer = self.get_tokenizer()
        tokenization_kwargs = self._get_tokenization_kw(tokenization_kwargs)

        encoder_config = self.model_config.encoder_config

        if encoder_config and encoder_config.get("do_lower_case", False):
            prompt = prompt.lower()

        return tokenizer.encode(prompt, **tokenization_kwargs)

    def _get_mm_processor(self) -> BaseMultiModalProcessor:
        if not hasattr(self, "_mm_processor"):
            self._mm_processor = self.mm_registry.create_processor(
                self.model_config,
                self.observability_config,
                tokenizer=self.tokenizer,
                cache=self.mm_processor_cache,
            )

        return self._mm_processor

    def _process_multimodal(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object] | None,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        """
        Apply the model's multi-modal processor to a multi-modal prompt,
        returning the corresponding token IDs and metadata.
        """
        mm_processor = self._get_mm_processor()

        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        mm_items = mm_processor.info.parse_mm_data(mm_data)
        mm_input = mm_processor.apply(
            prompt,
            mm_items,
            hf_processor_mm_kwargs=mm_processor_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
        mm_hashes = mm_input["mm_hashes"]

        # Validate that all mm items have a string as their hash
        contains_only_strings = all(
            isinstance(leaf, str) for leaf in json_iter_leaves(mm_hashes)
        )
        if not contains_only_strings:
            raise ValueError(
                f"mm_hashes must contain only strings, got: {mm_hashes}. "
                "This is likely due to an incorrect custom implementation of "
                "MultiModalProcessor.apply method."
            )

        return mm_input

    def _process_embeds(
        self,
        parsed_content: EmbedsPrompt,
    ) -> EmbedsInputs:
        if not self.model_config.enable_prompt_embeds:
            raise ValueError(
                "You must set `--enable-prompt-embeds` to input `prompt_embeds`."
            )

        prompt_embeds = parsed_content["prompt_embeds"]

        # prompt_embeds must be (seq_len, hidden_size), but if the user
        # passes in a batch of size 1, i.e. (1, seq_len, hidden_size),
        # we can unambiguously process the intent by squeezing the batch
        # dimension.
        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.squeeze(dim=0)

        if prompt_embeds.ndim != 2:
            raise ValueError("prompt_embeds must be of shape (seq_len, hidden_size).")

        # Tensors must be on CPU for serialization between processes
        # in the MsgpackEncoder. Casting to CPU here ensures that there is no
        # hidden device transfer in the critical path of generation.
        prompt_embeds = prompt_embeds.cpu()

        return embeds_inputs(
            prompt_embeds=prompt_embeds, cache_salt=parsed_content.get("cache_salt")
        )

    def _truncate_inputs(
        self, inputs: list[int], tokenization_kwargs: dict[str, Any] | None = None
    ) -> list[int]:
        if (
            not tokenization_kwargs
            or "truncation" not in tokenization_kwargs
            or self.tokenizer is None
        ):
            return inputs

        max_length = tokenization_kwargs["max_length"]

        if self.tokenizer.truncation_side == "left":
            return inputs[-max_length:]
        else:
            return inputs[:max_length]

    def _process_tokens(
        self,
        parsed_content: TokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> TokenInputs | MultiModalInputs:
        prompt_token_ids = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs
        )

        inputs: TokenInputs | MultiModalInputs
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
        else:
            inputs = token_inputs(prompt_token_ids)

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_text(
        self,
        parsed_content: TextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> TokenInputs | MultiModalInputs:
        prompt_text = parsed_content["prompt"]

        inputs: TokenInputs | MultiModalInputs
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            inputs = token_inputs(prompt_token_ids)

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> SingletonInputs:
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt

        Returns:

        * [`SingletonInputs`][vllm.inputs.data.SingletonInputs] instance
        """
        parsed = parse_singleton_prompt(prompt)

        if parsed["type"] == "embeds":
            return self._process_embeds(parsed["content"])
        if parsed["type"] == "tokens":
            return self._process_tokens(
                parsed["content"],
                mm_uuids=mm_uuids,
            )
        if parsed["type"] == "text":
            return self._process_text(
                parsed["content"],
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
        if parsed["type"] == "str":
            return self._process_text(
                TextPrompt(prompt=parsed["content"]),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(parsed)

    def _validate_enc_inputs(
        self,
        inputs: SingletonInputs,
    ) -> TokenInputs | MultiModalEncDecInputs:
        if inputs["type"] == "embeds":
            raise ValueError(
                "Embedding inputs are not supported for encoder-decoder models"
            )

        if inputs["type"] == "multimodal" and "encoder_prompt_token_ids" not in inputs:
            raise RuntimeError(
                "You should register an encoder-decoder "
                "multi-modal processor for encoder-decoder models."
            )

        return inputs  # type: ignore[return-value]

    def _validate_dec_inputs(
        self,
        inputs: SingletonInputs,
    ) -> TokenInputs | MultiModalInputs:
        if inputs["type"] == "embeds":
            raise ValueError(
                "Embedding inputs are not supported for encoder-decoder models"
            )

        return inputs

    def _build_enc_dec_inputs(
        self,
        encoder_inputs: SingletonInputs,
        decoder_inputs: SingletonInputs | None = None,
    ) -> EncoderDecoderInputs:
        if decoder_inputs is None:
            decoder_inputs = encoder_inputs

        enc_inputs = self._validate_enc_inputs(encoder_inputs)
        dec_inputs = self._validate_dec_inputs(decoder_inputs)

        enc_inputs_new: TokenInputs | MultiModalEncDecInputs
        dec_inputs_new: TokenInputs | MultiModalInputs

        if enc_inputs["type"] == "multimodal":
            enc_inputs_new = token_inputs(enc_inputs["encoder_prompt_token_ids"])
            dec_inputs_new = MultiModalInputs(
                type="multimodal",
                prompt_token_ids=dec_inputs["prompt_token_ids"],
                mm_kwargs=enc_inputs["mm_kwargs"],
                mm_hashes=enc_inputs["mm_hashes"],
                mm_placeholders=enc_inputs["mm_placeholders"],
            )
        elif enc_inputs["type"] == "token":
            enc_inputs_new = token_inputs(prompt_token_ids=[])
            dec_inputs_new = dec_inputs
        else:
            assert_never(enc_inputs)

        dec_inputs_new["prompt_token_ids"] = self._prepare_decoder_input_ids(
            dec_inputs_new["prompt_token_ids"]
        )
        if cache_salt := enc_inputs.get("cache_salt"):
            dec_inputs_new["cache_salt"] = cache_salt

        return EncoderDecoderInputs(encoder=enc_inputs_new, decoder=dec_inputs_new)

    def _process_encoder_decoder_prompt(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> EncoderDecoderInputs:
        """
        For encoder/decoder models only:
        Process an input prompt into an
        [`EncoderDecoderInputs`][vllm.inputs.data.EncoderDecoderInputs]
        instance.

        There are two types of input prompts:
        singleton prompts which carry only the
        encoder prompt, and explicit encoder/decoder
        prompts which carry both the encoder and the
        decoder prompts as member variables.

        This function handles the following scenarios:
        * Singleton encoder prompt: extract encoder prompt
          token ids & infer default decoder prompt token ids
        * Explicit encoder/decoder prompt: extract encoder
          and decoder prompt token ids

        Note that for Explicit encoder/decoder prompts,
        each sub-prompt (encoder or decoder prompt) can
        have any possible singleton type; thus this
        method relies on helper functions to obtain
        token ids for the sub-prompts.

        Arguments:

        * prompt: an input prompt

        Returns:

        * [`EncoderDecoderInputs`][vllm.inputs.data.EncoderDecoderInputs]
          instance
        """
        encoder_prompt, decoder_prompt = split_enc_dec_prompt(prompt)

        return self._build_enc_dec_inputs(
            encoder_inputs=self._prompt_to_llm_inputs(
                encoder_prompt,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            ),
            decoder_inputs=(
                None
                if decoder_prompt is None
                else self._prompt_to_llm_inputs(
                    decoder_prompt,
                    tokenization_kwargs=tokenization_kwargs,
                )
            ),
        )

    def _process_decoder_only_prompt(
        self,
        prompt: SingletonPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> DecoderOnlyInputs:
        """
        For decoder-only models:
        Process an input prompt into a
        [`DecoderOnlyInputs`][vllm.inputs.data.DecoderOnlyInputs] instance.

        Arguments:

        * prompt: input prompt

        Returns:

        * [`DecoderOnlyInputs`][vllm.inputs.data.DecoderOnlyInputs] instance
        """
        return self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def _preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> ProcessorInputs:
        if self.model_config.is_encoder_decoder:
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder.
            return self._process_encoder_decoder_prompt(
                prompt,
                tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        if is_explicit_encoder_decoder_prompt(prompt):
            raise ValueError(
                "Cannot pass encoder-decoder prompt to decoder-only models"
            )

        return self._process_decoder_only_prompt(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> ProcessorInputs:
        """Preprocess the input prompt."""
        res = self._preprocess(prompt, tokenization_kwargs, mm_uuids=mm_uuids)

        if self.mm_processor_cache and self.mm_cache_stats is not None:
            delta = self.mm_processor_cache.make_stats(delta=True)
            self.mm_cache_stats.requests += 1
            self.mm_cache_stats.queries += delta.total
            self.mm_cache_stats.hits += delta.hits

        return res

    def stat_mm_cache(self) -> MultiModalCacheStats | None:
        mm_cache_stats = self.mm_cache_stats
        if mm_cache_stats is None:
            return None

        self.mm_cache_stats = MultiModalCacheStats()

        return mm_cache_stats

    def clear_mm_cache(self) -> None:
        if self.mm_processor_cache is not None:
            self.mm_processor_cache.clear_cache()

        if self.mm_cache_stats is not None:
            self.mm_cache_stats.reset = True
