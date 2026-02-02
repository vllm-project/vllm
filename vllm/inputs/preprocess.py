# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import Any, cast

from typing_extensions import assert_never

from vllm.config import ModelConfig, ObservabilityConfig
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
from vllm.multimodal.video_sparse import SimilarFrameDetector, is_multimodal_efs_enabled
from vllm.renderers import renderer_from_config
from vllm.tokenizers import TokenizerLike
from vllm.utils.jsontree import json_iter_leaves
from vllm.v1.metrics.stats import MultiModalCacheStats

from .data import (
    DecoderOnlyInputs,
    EmbedsInputs,
    EmbedsPrompt,
    EncoderDecoderInputs,
    ExplicitEncoderDecoderPrompt,
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

    def get_decoder_start_token_id(self) -> int | None:
        """
        Obtain the decoder start token id employed by an encoder/decoder
        model. Returns None for non-encoder/decoder models or if the
        model config is unavailable.
        """

        if not self.model_config.is_encoder_decoder:
            logger.warning_once(
                "Using None for decoder start token id because "
                "this is not an encoder/decoder model."
            )
            return None

        if self.model_config is None or self.model_config.hf_config is None:
            logger.warning_once(
                "Using None for decoder start token id because "
                "model config is not available."
            )
            return None

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

        return dec_start_token_id

    def _get_default_enc_dec_decoder_prompt(self) -> list[int]:
        """
        Specifically for encoder/decoder models:
        generate a default decoder prompt for when
        the user specifies only the encoder prompt.

        Encoder/decoder models utilize the decoder
        prompt in different ways; as new models are
        added, it is intended that this function
        will be extended to produce differing
        default decoder prompts, depending on the
        model variety.

        Absent a special case, the default behavior
        of this method is to mirror the behavior of
        the HuggingFace (HF) GenerationMixin for a None
        decoder prompt, which is to employ a logit processor
        setting to force the first decoded token to be <BOS>.
        Here, this behavior is approximated by having the
        "default" decoder prompt be <BOS>.

        However, it is possible that in the future
        other models may have different or more
        complex logic for the default decoder prompt.
        This motivates having a special helper method
        for default decoder prompts.

        Returns:

        * prompt_token_ids
        """

        bos_token_id = self.get_bos_token_id()
        assert bos_token_id is not None
        return [bos_token_id]

    def _prepare_decoder_input_ids_for_generation(
        self,
        decoder_input_ids: list[int] | None,
    ) -> list[int]:
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
        assert decoder_start_token_id is not None

        if decoder_input_ids is None:
            # no decoder prompt input ->
            # use decoder_start_token_id as decoder_input_ids
            decoder_input_ids = self._get_default_enc_dec_decoder_prompt()

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

    def _build_enc_dec_llm_inputs(
        self,
        encoder_inputs: SingletonInputs,
        decoder_inputs: SingletonInputs | None,
    ) -> EncoderDecoderInputs:
        if (
            encoder_inputs["type"] == "embeds"
            or decoder_inputs
            and decoder_inputs["type"] == "embeds"
        ):
            raise ValueError(
                "Embedding inputs are not supported for encoder-decoder models"
            )

        # Needed for mypy
        encoder_inputs = cast(TokenInputs | MultiModalInputs, encoder_inputs)
        decoder_inputs = cast(TokenInputs | MultiModalInputs | None, decoder_inputs)

        if decoder_inputs is None:
            if self.model_config.hf_config.model_type == "whisper":
                # For Whisper models, the text prompt should go to the decoder.
                # If no explicit encoder/decoder inputs, then copy the prompt
                # from the encoder to the decoder. The encoder tokens are later
                # overridden by the audio features.
                dec_token_ids = encoder_inputs["prompt_token_ids"].copy()
            else:
                dec_token_ids = self._prepare_decoder_input_ids_for_generation(None)
            decoder_inputs = token_inputs(dec_token_ids)
        else:
            if "multi_modal_data" in decoder_inputs:
                raise ValueError(
                    "Multi-modal decoder inputs of encoder-"
                    "decoder models are not supported yet"
                )

            dec_token_ids = self._prepare_decoder_input_ids_for_generation(
                decoder_inputs["prompt_token_ids"]
            )
            decoder_inputs["prompt_token_ids"] = dec_token_ids

        return EncoderDecoderInputs(
            encoder=encoder_inputs,
            decoder=decoder_inputs,
        )

    def _split_enc_dec_mm_inputs(
        self,
        inputs: SingletonInputs | MultiModalEncDecInputs,
        decoder_inputs_to_override: SingletonInputs | None = None,
    ) -> tuple[SingletonInputs, SingletonInputs]:
        """
        For encoder/decoder models only:
        Separate Encoder/Decoder inputs from a MultiModalEncDecInputs
        """
        if (
            inputs["type"] == "embeds"
            or decoder_inputs_to_override
            and decoder_inputs_to_override["type"] == "embeds"
        ):
            raise ValueError(
                "Embedding inputs are not supported for encoder-decoder models"
            )

        # Needed for mypy
        inputs = cast(
            TokenInputs | MultiModalInputs | MultiModalEncDecInputs,
            inputs,
        )
        decoder_inputs_to_override = cast(
            TokenInputs | MultiModalInputs | None,
            decoder_inputs_to_override,
        )

        encoder_inputs: SingletonInputs
        decoder_inputs: SingletonInputs

        if inputs["type"] == "multimodal":  # Multimodal data inputs
            if "encoder_prompt_token_ids" not in inputs:
                raise RuntimeError(
                    "You should register an encoder-decoder "
                    "multi-modal processor for encoder-decoder "
                    "models."
                )
            inputs = cast(MultiModalEncDecInputs, inputs)

            encoder_inputs = token_inputs(inputs["encoder_prompt_token_ids"])

            decoder_prompt_inputs = decoder_inputs_to_override or inputs
            decoder_inputs = MultiModalInputs(
                type="multimodal",
                prompt_token_ids=decoder_prompt_inputs["prompt_token_ids"],
                mm_kwargs=inputs["mm_kwargs"],
                mm_hashes=inputs["mm_hashes"],
                mm_placeholders=inputs["mm_placeholders"],
            )
            if cache_salt := inputs.get("cache_salt"):
                decoder_inputs["cache_salt"] = cache_salt

        elif inputs["type"] == "token":  # Text-only inputs
            encoder_inputs = token_inputs(prompt_token_ids=[])
            decoder_inputs = decoder_inputs_to_override or inputs
        else:
            assert_never(inputs)  # type: ignore[arg-type]

        return encoder_inputs, decoder_inputs

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
        encoder_inputs: SingletonInputs
        decoder_inputs: SingletonInputs | None
        if is_explicit_encoder_decoder_prompt(prompt):
            # `cast` is needed for mypy, but not pyright
            prompt_ = cast(ExplicitEncoderDecoderPrompt, prompt)
            encoder_inputs = self._prompt_to_llm_inputs(
                prompt_["encoder_prompt"],
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            if (decoder_input := prompt_["decoder_prompt"]) is None:
                decoder_inputs = None
            else:
                decoder_inputs = self._prompt_to_llm_inputs(
                    decoder_input, tokenization_kwargs=tokenization_kwargs
                )
            # For multimodal model, override decoder prompt from processor
            # with explicit decoder prompt.
            if self.model_config.is_multimodal_model:
                encoder_inputs, decoder_inputs = self._split_enc_dec_mm_inputs(
                    encoder_inputs, decoder_inputs
                )
        else:
            # `cast` is needed for mypy, but not pyright
            inputs = self._prompt_to_llm_inputs(
                cast(SingletonPrompt, prompt),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            if self.model_config.is_multimodal_model:
                # Encoder-Decoder Multimodal model
                encoder_inputs, decoder_inputs = self._split_enc_dec_mm_inputs(inputs)
            else:
                encoder_inputs = inputs
                decoder_inputs = None

        return self._build_enc_dec_llm_inputs(encoder_inputs, decoder_inputs)

    def _build_decoder_only_llm_inputs(
        self,
        prompt_inputs: DecoderOnlyInputs,
    ) -> DecoderOnlyInputs:
        if "prompt_token_ids" in prompt_inputs:
            prompt_inputs = cast(
                TokenInputs | MultiModalInputs, prompt_inputs
            )  # Needed for mypy

        return prompt_inputs

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

        prompt_comps = self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        return self._build_decoder_only_llm_inputs(prompt_comps)

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

        # Decoder-only operation
        # `cast` is needed for mypy, but not pyright
        return self._process_decoder_only_prompt(
            cast(SingletonPrompt, prompt),
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
        """Efficient Frame Selection For Videos."""
        if (
            self.model_config
            and hasattr(self.model_config, "multimodal_config")
            and self.model_config.multimodal_config
        ):
            efs_sparse_rate = self.model_config.multimodal_config.video_sparse_rate
        else:
            efs_sparse_rate = 0.0
        efs_sparse_enabled = is_multimodal_efs_enabled(efs_sparse_rate)
        if (
            efs_sparse_enabled
            and isinstance(prompt, dict)
            and "multi_modal_data" in prompt
            and isinstance(prompt["multi_modal_data"], dict)
            and "video" in prompt["multi_modal_data"]
        ):
            videos = prompt["multi_modal_data"]["video"]
            sparse_ratio = 1 - (efs_sparse_rate if efs_sparse_rate is not None else 0.0)
            detector = SimilarFrameDetector(sparse_ratio=sparse_ratio)
            videos = detector.process_video_frames(videos)
            prompt["multi_modal_data"]["video"] = videos
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
