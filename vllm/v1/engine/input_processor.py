# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Mapping
from typing import Any, Literal, cast

from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs, PromptType, SingletonInputs
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.cache import processor_cache_from_config
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalUUIDDict
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import EncDecMultiModalProcessor
from vllm.multimodal.utils import argsort_mm_positions
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import MistralTokenizer, TokenizerLike
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.metrics.stats import MultiModalCacheStats
from vllm.v1.structured_output.backend_guidance import validate_guidance_grammar
from vllm.v1.structured_output.backend_lm_format_enforcer import (
    validate_structured_output_request_lm_format_enforcer,
)
from vllm.v1.structured_output.backend_outlines import (
    validate_structured_output_request_outlines,
)
from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

logger = init_logger(__name__)


class InputProcessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        tokenizer: TokenizerLike | None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.structured_outputs_config = vllm_config.structured_outputs_config

        self.generation_config_fields = self.model_config.try_get_generation_config()

        self.mm_registry = mm_registry
        self.mm_processor_cache = processor_cache_from_config(vllm_config, mm_registry)

        self.input_preprocessor = InputPreprocessor(
            self.model_config,
            tokenizer,
            mm_registry,
            mm_processor_cache=self.mm_processor_cache,
        )

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.input_preprocessor.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerLike | None) -> None:
        self.input_preprocessor.tokenizer = tokenizer

    def _validate_logprobs(
        self,
        params: SamplingParams,
    ) -> None:
        max_logprobs = self.model_config.max_logprobs
        if max_logprobs == -1:
            max_logprobs = self.model_config.get_vocab_size()

        # Validate sample logprobs.
        if params.logprobs:
            num_logprobs = params.logprobs
            if num_logprobs == -1:
                num_logprobs = self.model_config.get_vocab_size()
            if num_logprobs > max_logprobs:
                raise ValueError(
                    f"Requested sample logprobs of {num_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}"
                )

        # Validate prompt logprobs.
        if params.prompt_logprobs:
            num_prompt_logprobs = params.prompt_logprobs
            if num_prompt_logprobs == -1:
                num_prompt_logprobs = self.model_config.get_vocab_size()
            if num_prompt_logprobs > max_logprobs:
                raise ValueError(
                    f"Requested prompt logprobs of {num_prompt_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}"
                )

    def _validate_sampling_params(
        self,
        params: SamplingParams,
    ) -> None:
        self._validate_structured_output(params)
        self._validate_logit_bias(params)

        if params.allowed_token_ids is None:
            return
        if not params.allowed_token_ids:
            raise ValueError("allowed_token_ids is not None and empty!")
        if self.tokenizer is None:
            # When skip_tokenizer_init=True, we can't validate token IDs
            # Skip validation and let the model handle invalid tokens
            return
        vocab_size = len(self.tokenizer)
        if not all(0 <= tid < vocab_size for tid in params.allowed_token_ids):
            raise ValueError("allowed_token_ids contains out-of-vocab token id!")

    def _validate_logit_bias(
        self,
        params: SamplingParams,
    ) -> None:
        """Validate logit_bias token IDs are within vocabulary range."""
        if not params.logit_bias:
            return

        vocab_size = self.model_config.get_vocab_size()
        invalid_token_ids = []

        for token_id in params.logit_bias:
            if token_id < 0 or token_id >= vocab_size:
                invalid_token_ids.append(token_id)

        if invalid_token_ids:
            raise ValueError(
                f"token_id(s) {invalid_token_ids} in logit_bias contain "
                f"out-of-vocab token ids. Vocabulary size: {vocab_size}"
            )

    def _validate_supported_sampling_params(
        self,
        params: SamplingParams,
    ) -> None:
        # Logits processors not supported.
        if params.logits_processors:
            raise ValueError(
                "vLLM V1 does not support per request user provided logits processors."
            )
        # Async scheduling + spec decode currently incompatible with some
        # sampling parameters.
        if (
            self.vllm_config.speculative_config is not None
            and self.vllm_config.scheduler_config.async_scheduling
            and (
                params.frequency_penalty != 0.0
                or params.presence_penalty != 0.0
                or params.repetition_penalty != 1.0
                or params.bad_words_token_ids
                or params.structured_outputs
            )
        ):
            raise ValueError(
                "async scheduling with spec decoding doesn't yet support "
                "penalties, bad words or structured outputs in sampling parameters."
            )

    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,
    ):
        """
        Validate supported SamplingParam.
        Should raise ValueError if unsupported for API Server.
        """

        if isinstance(params, PoolingParams):
            return

        self._validate_logprobs(params)
        self._validate_sampling_params(params)
        self._validate_supported_sampling_params(params)

    def _validate_multi_modal_uuids(self, prompt: PromptType) -> None:
        """
        Validate that user-provided multi_modal_uuids align with
        multi_modal_data in the incoming request prompt(s).
        Only checks lengths; `None` entries are allowed and will be
        auto-hashed downstream.
        """

        def _validate_single_prompt(single_prompt: dict | str) -> None:
            if not isinstance(single_prompt, dict):
                return
            mm_data = single_prompt.get("multi_modal_data")
            mm_uuids = single_prompt.get("multi_modal_uuids")
            if not mm_data or not mm_uuids:
                return

            for modality, items in mm_data.items():
                if modality in mm_uuids:
                    data_len = len(items) if isinstance(items, list) else 1
                    uuid_len = (
                        len(mm_uuids[modality])
                        if isinstance(mm_uuids[modality], list)
                        else 1
                    )
                    if uuid_len != data_len:
                        raise ValueError(
                            f"multi_modal_uuids for modality '{modality}' "
                            "must have same length as data: got "
                            f"{uuid_len} uuids vs "
                            f"{data_len} items."
                        )
                else:
                    raise ValueError(
                        f"multi_modal_uuids for modality '{modality}' must "
                        "be provided if multi_modal_data is provided."
                    )

        # Handle explicit encoder/decoder prompts or singleton prompt
        if isinstance(prompt, dict) and "encoder_prompt" in prompt:
            enc = prompt.get("encoder_prompt")
            dec = prompt.get("decoder_prompt")
            if enc is not None:
                _validate_single_prompt(cast(dict | str, enc))
            if dec is not None:
                _validate_single_prompt(cast(dict | str, dec))
        else:
            _validate_single_prompt(prompt)  # type: ignore[arg-type]

    def _validate_lora(self, lora_request: LoRARequest | None) -> None:
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise ValueError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            logger.warning_once(
                "vLLM has deprecated support for supporting different "
                "tokenizers for different LoRAs. By default, vLLM uses base "
                "model's tokenizer. If you are using a LoRA "
                "with its own tokenizer, consider specifying `--tokenizer "
                "[lora_path]` to use the LoRA tokenizer."
            )

    def _validate_structured_output(self, params: SamplingParams) -> None:
        if not params.structured_outputs or not self.structured_outputs_config:
            return

        if self.model_config.skip_tokenizer_init and params.structured_outputs:
            raise ValueError(
                "Structured outputs requires a tokenizer so it can't be used with 'skip_tokenizer_init'"  # noqa: E501
            )

        backend = self.structured_outputs_config.backend
        if _backend := params.structured_outputs._backend:
            # Request-level backend selection is not supported.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_backend_was_auto` field set in the params.
            if backend != _backend and not (
                backend == "auto" and params.structured_outputs._backend_was_auto
            ):
                raise ValueError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            params.structured_outputs._backend = backend

        # Request content validation
        if (
            isinstance(params.structured_outputs.choice, list)
            and not params.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise ValueError(
                f"Choice '{params.structured_outputs.choice}' cannot be an empty list"  # noqa: E501
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (
            isinstance(params.structured_outputs.grammar, str)
            and params.structured_outputs.grammar.strip() == ""
        ):
            raise ValueError("structured_outputs.grammar cannot be an empty string")

        if backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(params)
        elif backend.startswith("guidance"):
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            if isinstance(self.tokenizer, MistralTokenizer):
                raise ValueError(
                    "Mistral tokenizer is not supported for the 'guidance' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_guidance_grammar(params, tokenizer=None)
        elif backend == "outlines":
            # outlines backend
            validate_structured_output_request_outlines(params)
        elif backend == "lm-format-enforcer":
            # lm format enforcer backend
            if isinstance(self.tokenizer, MistralTokenizer):
                raise ValueError(
                    "Mistral tokenizer is not supported for the 'lm-format-enforcer' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_structured_output_request_lm_format_enforcer(params)
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # In this mode, we set opinionated defaults based on what we think
            # will satisfy the most use cases without having to worry about
            # this setting. We include fallback behavior here, but not with any
            # other setting where a specific backend was specified.
            try:
                validate_xgrammar_grammar(params)
                params.structured_outputs._backend = "xgrammar"
            except ValueError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.
                if isinstance(self.tokenizer, MistralTokenizer):
                    # Fall back to outlines if the tokenizer is Mistral
                    validate_structured_output_request_outlines(params)
                    params.structured_outputs._backend = "outlines"
                else:
                    # Fall back to guidance by default.
                    validate_guidance_grammar(params, tokenizer=None)
                    params.structured_outputs._backend = "guidance"
            # Remember that this backend was set automatically
            params.structured_outputs._backend_was_auto = True

    def _maybe_build_mm_uuids(
        self,
        request_id: str,
        prompt: PromptType,
    ) -> MultiModalUUIDDict | None:
        """Build per-item multimodal hash overrides when enabled. In this case,
        multimodal data items are identified by their request id, modality and
        index rather than their content.

        Returns a dictionary of modality -> list[str] of overrides, or None if
        disabled or no multimodal data is present.
        """

        def _extract_mm_data(p: PromptType):
            if isinstance(p, dict) and "encoder_prompt" in p:
                enc = p.get("encoder_prompt")
                if isinstance(enc, dict):
                    return enc.get("multi_modal_data")
                return None
            if isinstance(p, dict):
                return p.get("multi_modal_data")
            return None

        mm_data = _extract_mm_data(prompt)
        if not mm_data:
            return None

        mm_uuids: dict[str, list[str | None] | str] = {}
        for modality, data in mm_data.items():
            # Hash each item for embedding inputs.
            n = (
                len(data)
                if isinstance(data, list) or MultiModalDataParser.is_embeddings(data)
                else 1
            )
            mm_uuids[modality] = [f"{request_id}-{modality}-{i}" for i in range(n)]
        return mm_uuids

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> EngineCoreRequest:
        self._validate_lora(lora_request)
        self._validate_params(params)

        data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if data_parallel_rank is not None and not (
            0 <= data_parallel_rank < data_parallel_size
        ):
            raise ValueError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {data_parallel_size})."
            )

        if arrival_time is None:
            arrival_time = time.time()

        # Optionally generate multimodal hash overrides to avoid hashing
        # multimodal data items by their content as their identifiers.

        # NOTE: when users explicitly turn off BOTH prefix caching and input
        # processing caching, no multimodal features or embeddings will be
        # reused across requests, therefore identifying multimodal data items
        # by their content is no longer necessary, and we create uuids with
        # request id-modality-index as multimodal hash overrides.
        if (
            self.model_config.multimodal_config
            and self.model_config.multimodal_config.mm_processor_cache_gb == 0
            and not self.cache_config.enable_prefix_caching
        ):
            mm_uuids = self._maybe_build_mm_uuids(request_id, prompt)
        else:
            # Otherwise, use user-provided uuids as multimodal hash overrides
            # if provided.
            self._validate_multi_modal_uuids(prompt)
            if isinstance(prompt, dict):
                mm_uuids = cast(
                    MultiModalUUIDDict | None, prompt.get("multi_modal_uuids")
                )
            else:
                mm_uuids = None

        # Process inputs, which includes:
        # 1. Tokenize text prompt, with LoRA request if one exists.
        # 2. For multimodal models with a merged preprocessor, preprocess
        #   multimodal data and expand prompt token ids accordingly.
        processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
        from vllm.platforms import current_platform

        current_platform.validate_request(
            prompt=prompt,
            params=params,
            processed_inputs=processed_inputs,
        )

        eos_token_id = self.input_preprocessor.get_eos_token_id()

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_inputs["type"] == "embeds":
            prompt_token_ids = None
            prompt_embeds = decoder_inputs["prompt_embeds"]
        else:
            prompt_token_ids = decoder_inputs["prompt_token_ids"]
            prompt_embeds = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len
            sampling_params.update_from_generation_config(
                self.generation_config_fields, eos_token_id
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=decoder_mm_hashes[modality][idx],
                        mm_position=decoder_mm_positions[modality][idx],
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
        )

    def _validate_model_inputs(
        self, encoder_inputs: SingletonInputs | None, decoder_inputs: SingletonInputs
    ):
        if encoder_inputs is not None:
            self._validate_model_input(encoder_inputs, prompt_type="encoder")

        self._validate_model_input(decoder_inputs, prompt_type="decoder")

    def _validate_model_input(
        self,
        prompt_inputs: SingletonInputs,
        *,
        prompt_type: Literal["encoder", "decoder"],
    ):
        model_config = self.model_config

        prompt_ids = (
            None
            if prompt_inputs["type"] == "embeds"
            else prompt_inputs["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_inputs["prompt_embeds"]
            if prompt_inputs["type"] == "embeds"
            else None
        )
        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        if not prompt_ids:
            if prompt_type == "encoder" and model_config.is_multimodal_model:
                pass  # Mllama may have empty encoder inputs for text-only data
            elif prompt_inputs["type"] == "embeds":
                pass  # Prompt embeds should not have prompt_ids.
            else:
                raise ValueError(f"The {prompt_type} prompt cannot be empty")

        tokenizer = self.tokenizer
        if tokenizer is not None:
            max_input_id = max(prompt_ids or [], default=0)

            # NOTE: tokenizer.max_token_id is the tokenizer’s vocab size while
            # self.model_config.get_vocab_size() is the model’s vocab size.
            # For Qwen3 models, the language model has extra tokens that do
            # not exist in the tokenizer, and vice versa for multimodal
            # placeholder tokens in some multimodal models.
            # See https://github.com/QwenLM/Qwen3/issues/29#issuecomment-1933720399 # noqa: E501
            # and https://github.com/vllm-project/vllm/pull/22471#discussion_r2312251421 # noqa: E501

            # Here we take the max of the two to determine if a token id is
            # truly out-of-vocabulary.
            if max_input_id > max(
                tokenizer.max_token_id, self.model_config.get_vocab_size() - 1
            ):
                raise ValueError(f"Token id {max_input_id} is out of vocabulary")

        max_prompt_len = self.model_config.max_model_len
        if prompt_len > max_prompt_len:
            if prompt_type == "encoder" and model_config.is_multimodal_model:
                mm_registry = self.input_preprocessor.mm_registry
                mm_processor = mm_registry.create_processor(
                    model_config,
                    tokenizer=tokenizer,
                )
                assert isinstance(mm_processor, EncDecMultiModalProcessor)

                if mm_processor.pad_dummy_encoder_prompt:
                    return  # Skip encoder length check for Whisper

            if model_config.is_multimodal_model:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well."
                )
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens."
                )

            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )

            # TODO: Find out how many placeholder tokens are there so we can
            # check that chunked prefill does not truncate them
            # max_batch_len = self.scheduler_config.max_num_batched_tokens

        if (
            prompt_len == max_prompt_len
            and prompt_type == "decoder"
            and not model_config.is_multimodal_model
            and self.model_config.runner_type != "pooling"
        ):
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    def stat_mm_cache(self) -> MultiModalCacheStats | None:
        return self.input_preprocessor.stat_mm_cache()

    def clear_mm_cache(self) -> None:
        self.input_preprocessor.clear_mm_cache()
