# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Mapping, Sequence
from typing import Literal, Optional, Union

from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs, PromptType, SingletonInputs
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalKwargs,
                             MultiModalRegistry)
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.processing import EncDecMultiModalProcessor
from vllm.multimodal.utils import merge_and_sort_multimodal_metadata
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.mm_input_cache import MirroredProcessingCache
from vllm.v1.structured_output.backend_guidance import (
    validate_guidance_grammar)
from vllm.v1.structured_output.backend_xgrammar import (
    validate_xgrammar_grammar)


class Processor:

    def __init__(
        self,
        vllm_config: VllmConfig,
        tokenizer: BaseTokenizerGroup,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.decoding_config = vllm_config.decoding_config
        self.tokenizer = tokenizer

        self.generation_config_fields = (
            self.model_config.try_get_generation_config())
        self.input_preprocessor = InputPreprocessor(self.model_config,
                                                    self.tokenizer,
                                                    mm_registry)

        self.mm_input_cache_client = MirroredProcessingCache(self.model_config)

        # Multi-modal hasher (for images)
        self.use_hash = (
            not self.model_config.disable_mm_preprocessor_cache) or \
            self.cache_config.enable_prefix_caching

    def _validate_logprobs(
        self,
        params: SamplingParams,
    ) -> None:
        max_logprobs = self.model_config.max_logprobs
        # Validate sample logprobs.
        if params.logprobs and params.logprobs > max_logprobs:
            raise ValueError(
                f"Requested sample logprobs of {params.logprobs}, "
                f"which is greater than max allowed: {max_logprobs}")

        # Validate prompt logprobs.
        if params.prompt_logprobs and params.prompt_logprobs > max_logprobs:
            raise ValueError(
                f"Requested prompt logprobs of {params.prompt_logprobs}, "
                f"which is greater than max allowed: {max_logprobs}")

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
        vocab_size = self.model_config.get_vocab_size()
        if not all(0 <= tid < vocab_size for tid in params.allowed_token_ids):
            raise ValueError(
                "allowed_token_ids contains out-of-vocab token id!")

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
                f"out-of-vocab token ids. Vocabulary size: {vocab_size}")

    def _validate_supported_sampling_params(
        self,
        params: SamplingParams,
    ) -> None:
        # Best of not yet supported.
        if params.best_of is not None and params.best_of > 1:
            raise ValueError("vLLM V1 does not yet support best_of.")
        # Logits processors not supported.
        if params.logits_processors:
            raise ValueError("vLLM V1 does not support per request "
                             "user provided logits processors.")

    def _validate_params(
        self,
        params: Union[SamplingParams, PoolingParams],
    ):
        """
        Validate supported SamplingParam.
        Should raise ValueError if unsupported for API Server.
        """

        if not isinstance(params, SamplingParams):
            raise ValueError("V1 does not yet support Pooling models.")

        self._validate_logprobs(params)
        self._validate_sampling_params(params)
        self._validate_supported_sampling_params(params)

    def _validate_lora(self, lora_request: Optional[LoRARequest]) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")

    def _validate_structured_output(self, params: SamplingParams) -> None:
        if not params.guided_decoding or not self.decoding_config:
            return

        supported_backends = [
            "xgrammar", "xgrammar:disable-any-whitespace", "guidance",
            "guidance:disable-any-whitespace", "auto"
        ]

        engine_level_backend = self.decoding_config.guided_decoding_backend
        if engine_level_backend not in supported_backends:
            raise ValueError(f"Only {supported_backends} structured output is "
                             "supported in V1.")
        if params.guided_decoding.backend:
            # Request-level backend selection is not supported in V1.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_auto` option set on the backend in the params.
            if (params.guided_decoding.backend != engine_level_backend
                    and not (engine_level_backend == "auto" and "_auto"
                             in params.guided_decoding.backend_options())):
                raise ValueError(
                    "Request-level structured output backend selection is no "
                    "longer supported. The request specified "
                    f"'{params.guided_decoding.backend}', but vLLM was "
                    f"initialised with '{engine_level_backend}'. This error "
                    "can be resolved by removing backend selection from the "
                    "request.")
        else:
            params.guided_decoding.backend = engine_level_backend

        # Request content validation
        if engine_level_backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(params)
        elif engine_level_backend.startswith("guidance"):
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            validate_guidance_grammar(params, tokenizer=None)
        else:
            # NOTE: engine_level_backend must be "auto" here, because we have
            # checked supported_backends above.
            # "auto" is an opt-in to opinionated behavior where we try to
            # choose a backend based on request contents. This is not the
            # default as it is less predictable and subject to change
            # between releases as feature support changes.
            try:
                validate_xgrammar_grammar(params)
                params.guided_decoding.backend = "xgrammar"
            except ValueError:
                # The request includes some jsonschema feature(s) that
                # are not supported in xgrammar. Fall back to guidance.
                params.guided_decoding.backend = "guidance"
            # Remember that this backend was set automatically
            params.guided_decoding.add_option("_auto")

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> EngineCoreRequest:

        # TODO(woosuk): Support pooling models.
        # TODO(woosuk): Support encoder-decoder models.
        self._validate_lora(lora_request)
        self._validate_params(params)
        if priority != 0:
            raise ValueError("V1 does not support priority yet.")
        if trace_headers is not None:
            raise ValueError("V1 does not support tracing yet.")
        if prompt_adapter_request is not None:
            raise ValueError("V1 does not support prompt_adapter_request.")

        if arrival_time is None:
            arrival_time = time.time()

        # Process inputs, which includes:
        # 1. Tokenize text prompt, with LoRA request if one exists.
        # 2. For multimodal models with a merged preprocessor, preprocess
        #   multimodal data and expand prompt token ids accordingly.
        # 3. Apply prompt adapter to prompt token ids if one exists.
        processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
            prompt,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            return_mm_hashes=self.use_hash,
        )
        from vllm.platforms import current_platform
        current_platform.validate_request(
            prompt=prompt,
            params=params,
            processed_inputs=processed_inputs,
        )
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        self._validate_model_inputs(processed_inputs, lora_request)

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)

        # TODO: Impl encoder-decoder
        if encoder_inputs is not None:
            raise NotImplementedError

        assert isinstance(params, SamplingParams)
        # TODO: can we avoid cloning here in multiproc case?
        sampling_params = params.clone()
        # If unset max tokens, then generate up to the max_model_len.
        if sampling_params.max_tokens is None:
            sampling_params.max_tokens = (
                self.model_config.max_model_len -
                len(decoder_inputs["prompt_token_ids"]))
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)
        sampling_params.update_from_tokenizer(
            self.tokenizer.get_lora_tokenizer(lora_request))

        # Multimodal related.
        sorted_mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]] = None
        sorted_mm_positions: Optional[list[PlaceholderRange]] = None
        sorted_mm_hashes: Optional[list[str]] = None
        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            (
                sorted_item_modalities,
                sorted_mm_positions,
                sorted_mm_hashes,
            ) = merge_and_sort_multimodal_metadata(
                decoder_inputs["mm_placeholders"],
                decoder_inputs["mm_hashes"] if self.use_hash else None,
            )

            # The output of merged multi-modal processor (`decoder_mm_inputs`)
            # is a single MultiModalKwargs for all items from all modalities.
            # This code flattens kwargs for individual items in a list and
            # sorts them by each item's position in the input sequence if there
            # are multiple modalities.
            unique_modalities = set(sorted_item_modalities)
            if len(unique_modalities) > 1:
                orig_sorted_mm_inputs = []
                used_indices = {modality: 0 for modality in unique_modalities}

                for modality in sorted_item_modalities:
                    items = decoder_mm_inputs.get_items(modality)
                    item = items[used_indices[modality]]

                    orig_sorted_mm_inputs.append(
                        MultiModalKwargs.from_items([item]))
                    used_indices[modality] += 1
            else:
                orig_sorted_mm_inputs = [
                    MultiModalKwargs.from_items([item]) for item in
                    decoder_mm_inputs.get_items(sorted_item_modalities[0])
                ]

            if sorted_mm_hashes is not None:
                sorted_mm_inputs = self.mm_input_cache_client.get_and_update_p0(
                    orig_sorted_mm_inputs, sorted_mm_hashes)
            else:
                sorted_mm_inputs = orig_sorted_mm_inputs

        return EngineCoreRequest(
            request_id=request_id,
            prompt=decoder_inputs.get("prompt"),
            prompt_token_ids=decoder_inputs["prompt_token_ids"],
            mm_inputs=sorted_mm_inputs,
            mm_hashes=sorted_mm_hashes,
            mm_placeholders=sorted_mm_positions,
            sampling_params=sampling_params,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
        )

    def _validate_model_inputs(self,
                               inputs: ProcessorInputs,
                               lora_request: Optional[LoRARequest] = None):
        encoder_inputs, decoder_inputs = split_enc_dec_inputs(inputs)

        if encoder_inputs is not None:
            self._validate_model_input(encoder_inputs,
                                       lora_request,
                                       prompt_type="encoder")

        self._validate_model_input(decoder_inputs,
                                   lora_request,
                                   prompt_type="decoder")

    def _validate_model_input(
        self,
        prompt_inputs: SingletonInputs,
        lora_request: Optional[LoRARequest],
        *,
        prompt_type: Literal["encoder", "decoder"],
    ):
        model_config = self.model_config
        tokenizer = self.tokenizer.get_lora_tokenizer(lora_request)

        prompt_ids = prompt_inputs["prompt_token_ids"]
        if not prompt_ids:
            if prompt_type == "encoder" and model_config.is_multimodal_model:
                pass  # Mllama may have empty encoder inputs for text-only data
            else:
                raise ValueError(f"The {prompt_type} prompt cannot be empty")

        max_input_id = max(prompt_ids, default=0)
        if max_input_id > tokenizer.max_token_id:
            raise ValueError(f"Token id {max_input_id} is out of vocabulary")

        max_prompt_len = self.model_config.max_model_len
        if len(prompt_ids) > max_prompt_len:
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
                    "of images, and possibly their aspect ratios as well.")
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens.")

            raise ValueError(
                f"The {prompt_type} prompt (length {len(prompt_ids)}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}")

            # TODO: Find out how many placeholder tokens are there so we can
            # check that chunked prefill does not truncate them
            # max_batch_len = self.scheduler_config.max_num_batched_tokens
