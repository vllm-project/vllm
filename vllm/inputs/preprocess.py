# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping
from typing import Optional, Union, cast

from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalEncDecInputs,
                                    MultiModalInputs)
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.transformers_utils.tokenizer_group import TokenizerGroup

from .data import (DecoderOnlyInputs, EncoderDecoderInputs, ProcessorInputs,
                   PromptType, SingletonInputs, SingletonPrompt, token_inputs)
from .parse import is_explicit_encoder_decoder_prompt, parse_singleton_prompt

logger = init_logger(__name__)


class InputPreprocessor:

    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: Optional[TokenizerGroup],
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.tokenizer = tokenizer
        self.mm_registry = mm_registry

    def get_tokenizer_group(self) -> TokenizerGroup:
        if self.tokenizer is None:
            raise ValueError("You cannot pass text prompts when "
                             "`skip_tokenizer_init` is True")

        return self.tokenizer

    def get_bos_token_id(self,
                         lora_request: Optional[LoRARequest] = None
                         ) -> Optional[int]:
        if self.tokenizer is None:
            logger.warning("Using None for BOS token id because tokenizer "
                           "is not initialized")
            return None

        return self.tokenizer.get_lora_tokenizer(lora_request).bos_token_id

    def get_eos_token_id(self,
                         lora_request: Optional[LoRARequest] = None
                         ) -> Optional[int]:
        if self.tokenizer is None:
            logger.warning("Using None for EOS token id because tokenizer "
                           "is not initialized")
            return None

        return self.tokenizer.get_lora_tokenizer(lora_request).eos_token_id

    def get_decoder_start_token_id(self) -> Optional[int]:
        '''
        Obtain the decoder start token id employed by an encoder/decoder
        model. Returns None for non-encoder/decoder models or if the
        model config is unavailable.
        '''

        if not self.model_config.is_encoder_decoder:
            logger.warning_once(
                "Using None for decoder start token id because "
                "this is not an encoder/decoder model.")
            return None

        if (self.model_config is None or self.model_config.hf_config is None):
            logger.warning_once(
                "Using None for decoder start token id because "
                "model config is not available.")
            return None

        dec_start_token_id = getattr(self.model_config.hf_config,
                                     'decoder_start_token_id', None)
        if dec_start_token_id is None:
            logger.warning_once(
                "Falling back on <BOS> for decoder start token "
                "id because decoder start token id is not "
                "available.")
            dec_start_token_id = self.get_bos_token_id()

        return dec_start_token_id

    def _get_default_enc_dec_decoder_prompt(self) -> list[int]:
        '''
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
        '''

        bos_token_id = self.get_bos_token_id()
        assert bos_token_id is not None
        return [bos_token_id]

    def _prepare_decoder_input_ids_for_generation(
        self,
        decoder_input_ids: Optional[list[int]],
    ) -> list[int]:
        """
        Prepares `decoder_input_ids` for generation with encoder-decoder models.

        Based on

        https://github.com/huggingface/transformers/blob/
        4037a2b5b1278736e566aec12e169100275545ea/
        src/transformers/generation/utils.py

        specifically GenerationMixin._prepare_decoder_input_ids_for_generation()

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

        if (len(decoder_input_ids) == 0
                or decoder_input_ids[0] != decoder_start_token_id):
            decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

        return decoder_input_ids

    def _apply_prompt_adapter(
        self,
        prompt_token_ids: list[int],
        prompt_adapter_request: Optional[PromptAdapterRequest],
    ) -> list[int]:
        if prompt_adapter_request:
            prompt_token_ids = (
                [0] * prompt_adapter_request.prompt_adapter_num_virtual_tokens
                + prompt_token_ids)

        return prompt_token_ids

    def _tokenize_prompt(
        self,
        prompt: str,
        lora_request: Optional[LoRARequest],
    ) -> list[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        tokenizer = self.get_tokenizer_group()
        add_special_tokens = None
        if self.model_config.hf_config.model_type == "whisper":
            # For Whisper, special tokens should be provided by the user based
            # on the task and language of their request. Also needed to avoid
            # appending an EOS token to the prompt which disrupts generation.
            add_special_tokens = False

        if (self.model_config.encoder_config is not None
                and self.model_config.encoder_config.get(
                    "do_lower_case", False)):
            prompt = prompt.lower()

        return tokenizer.encode(prompt=prompt,
                                lora_request=lora_request,
                                add_special_tokens=add_special_tokens)

    async def _tokenize_prompt_async(
        self,
        prompt: str,
        lora_request: Optional[LoRARequest],
    ) -> list[int]:
        """Async version of :meth:`_tokenize_prompt`."""
        tokenizer = self.get_tokenizer_group()
        add_special_tokens = None
        if self.model_config.hf_config.model_type == "whisper":
            # For Whisper, special tokens should be provided by the user based
            # on the task and language of their request. Also needed to avoid
            # appending an EOS token to the prompt which disrupts generation.
            add_special_tokens = False
        return await tokenizer.encode_async(
            prompt=prompt,
            lora_request=lora_request,
            add_special_tokens=add_special_tokens)

    def _process_multimodal(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Optional[Mapping[str, object]],
        lora_request: Optional[LoRARequest],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Apply the model's multi-modal processor to a multi-modal prompt,
        returning the corresponding token IDs and metadata.
        """
        # At the moment on model (PrithviGeoSpatialMAE) requires to be
        # initialized without a tokenizer while using also multi-modal input
        if not self.tokenizer:
            tokenizer = object()  # Dummy
        else:
            tokenizer_group = self.get_tokenizer_group()
            tokenizer = tokenizer_group.get_lora_tokenizer(lora_request)

        mm_processor = self.mm_registry.create_processor(self.model_config,
                                                         tokenizer=tokenizer)

        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        return mm_processor.apply(prompt, mm_data, mm_processor_kwargs,
                                  return_mm_hashes)

    async def _process_multimodal_async(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Optional[Mapping[str, object]],
        lora_request: Optional[LoRARequest],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """Async version of :meth:`_process_multimodal`."""
        # At the moment on model (PrithviGeoSpatialMAE) requires to be
        # initialized without a tokenizer while using also multi-modal input
        if not self.tokenizer:
            tokenizer = object()  # Dummy
        else:
            tokenizer_group = self.get_tokenizer_group()
            tokenizer = await tokenizer_group.get_lora_tokenizer_async(
                lora_request)

        mm_processor = self.mm_registry.create_processor(self.model_config,
                                                         tokenizer=tokenizer)
        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        return mm_processor.apply(prompt, mm_data, mm_processor_kwargs,
                                  return_mm_hashes)

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonPrompt,
        lora_request: Optional[LoRARequest] = None,
        return_mm_hashes: bool = False,
    ) -> SingletonInputs:
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt
        * lora_request: this is only valid for decoder prompts
        * return_mm_hashes: whether to return multimodal hashes

        Returns:

        * :class:`SingletonInputs` instance
        """
        parsed = parse_singleton_prompt(prompt)

        if parsed["type"] == "str":
            prompt_text = parsed["content"]
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                lora_request=lora_request,
            )

            return token_inputs(
                prompt=prompt_text,
                prompt_token_ids=prompt_token_ids,
            )

        if parsed["type"] == "tokens":
            tokens_content = parsed["content"]

            prompt_token_ids = tokens_content["prompt_token_ids"]
            token_type_ids = tokens_content.get("token_type_ids")
            multi_modal_data = tokens_content.get("multi_modal_data")
            mm_processor_kwargs = tokens_content.get("mm_processor_kwargs")

            if multi_modal_data is not None:
                return self._process_multimodal(
                    prompt_token_ids,
                    multi_modal_data,
                    mm_processor_kwargs,
                    lora_request=lora_request,
                    return_mm_hashes=return_mm_hashes,
                )

            return token_inputs(
                prompt_token_ids=prompt_token_ids,
                token_type_ids=token_type_ids,
            )

        if parsed["type"] == "text":
            text_content = parsed["content"]

            prompt_text = text_content["prompt"]
            multi_modal_data = text_content.get("multi_modal_data")
            mm_processor_kwargs = text_content.get("mm_processor_kwargs")

            if multi_modal_data is not None:
                return self._process_multimodal(
                    prompt_text,
                    multi_modal_data,
                    mm_processor_kwargs,
                    lora_request=lora_request,
                    return_mm_hashes=return_mm_hashes,
                )

            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                lora_request=lora_request,
            )

            return token_inputs(
                prompt=prompt_text,
                prompt_token_ids=prompt_token_ids,
            )

        assert_never(parsed)

    async def _prompt_to_llm_inputs_async(
        self,
        prompt: SingletonPrompt,
        lora_request: Optional[LoRARequest] = None,
        return_mm_hashes: bool = False,
    ) -> SingletonInputs:
        """Async version of :meth:`_extract_prompt_components`."""
        parsed = parse_singleton_prompt(prompt)

        if parsed["type"] == "str":
            prompt_text = parsed["content"]
            prompt_token_ids = await self._tokenize_prompt_async(
                prompt_text,
                lora_request=lora_request,
            )

            return token_inputs(
                prompt=prompt_text,
                prompt_token_ids=prompt_token_ids,
            )

        if parsed["type"] == "tokens":
            tokens_content = parsed["content"]

            prompt_token_ids = tokens_content["prompt_token_ids"]
            multi_modal_data = tokens_content.get("multi_modal_data")
            mm_processor_kwargs = tokens_content.get("mm_processor_kwargs")

            if multi_modal_data is not None:
                return await self._process_multimodal_async(
                    prompt_token_ids,
                    multi_modal_data,
                    mm_processor_kwargs,
                    lora_request=lora_request,
                    return_mm_hashes=return_mm_hashes,
                )

            return token_inputs(prompt_token_ids=prompt_token_ids)

        if parsed["type"] == "text":
            text_content = parsed["content"]

            prompt_text = text_content["prompt"]
            multi_modal_data = text_content.get("multi_modal_data")
            mm_processor_kwargs = text_content.get("mm_processor_kwargs")

            if multi_modal_data is not None:
                return await self._process_multimodal_async(
                    prompt_text,
                    multi_modal_data,
                    mm_processor_kwargs,
                    lora_request=lora_request,
                    return_mm_hashes=return_mm_hashes,
                )

            prompt_token_ids = await self._tokenize_prompt_async(
                prompt_text,
                lora_request=lora_request,
            )

            return token_inputs(
                prompt=prompt_text,
                prompt_token_ids=prompt_token_ids,
            )

        assert_never(parsed)

    def _build_enc_dec_llm_inputs(
        self,
        encoder_inputs: SingletonInputs,
        decoder_inputs: Optional[SingletonInputs],
    ) -> EncoderDecoderInputs:
        if (encoder_inputs["type"] == "token"
                or encoder_inputs["type"] == "multimodal"):
            pass
        else:
            assert_never(encoder_inputs)  # type: ignore[arg-type]

        if decoder_inputs is None:
            if self.model_config.hf_config.model_type == "whisper":
                # For Whisper models, the text prompt should go to the decoder.
                # If no explicit encoder/decoder inputs, then copy the prompt
                # from the encoder to the decoder. The encoder tokens are later
                # overridden by the audio features.
                dec_token_ids = encoder_inputs["prompt_token_ids"].copy()
            else:
                dec_token_ids = self._prepare_decoder_input_ids_for_generation(
                    None)
            decoder_inputs = token_inputs(dec_token_ids)
        elif (decoder_inputs["type"] == "token"
              or decoder_inputs["type"] == "multimodal"):
            dec_token_ids = self._prepare_decoder_input_ids_for_generation(
                decoder_inputs["prompt_token_ids"])
            decoder_inputs["prompt_token_ids"] = dec_token_ids

            if "multi_modal_data" in decoder_inputs:
                raise ValueError("Multi-modal decoder inputs of encoder-"
                                 "decoder models are not supported yet")
        else:
            assert_never(encoder_inputs)  # type: ignore[arg-type]

        return EncoderDecoderInputs(
            encoder=encoder_inputs,
            decoder=decoder_inputs,
        )

    def _separate_enc_dec_inputs_from_mm_processor_outputs(
        self,
        inputs: SingletonInputs,
        decoder_inputs_to_override: Optional[SingletonInputs] = None,
    ) -> tuple[SingletonInputs, SingletonInputs]:
        """
        For encoder/decoder models only:
        Separate Encoder/Decoder inputs from a MultiModalEncDecInputs
        """
        encoder_inputs: SingletonInputs
        decoder_inputs: SingletonInputs
        if inputs["type"] == "multimodal":
            # Multimodal data inputs
            assert ("encoder_prompt" in inputs
                    and "encoder_prompt_token_ids" in inputs)
            inputs = cast(MultiModalEncDecInputs, inputs)
            encoder_inputs = token_inputs(
                prompt=inputs["encoder_prompt"],
                prompt_token_ids=inputs["encoder_prompt_token_ids"],
            )
            if decoder_inputs_to_override is not None:
                decoder_inputs = MultiModalInputs(
                    type="multimodal",
                    prompt=decoder_inputs_to_override.get("prompt", ""),
                    prompt_token_ids=decoder_inputs_to_override[
                        "prompt_token_ids"],
                    mm_kwargs=inputs["mm_kwargs"],
                    mm_hashes=inputs["mm_hashes"],
                    mm_placeholders=inputs["mm_placeholders"],
                )
            else:
                decoder_inputs = MultiModalInputs(
                    type="multimodal",
                    prompt=inputs["prompt"],
                    prompt_token_ids=inputs["prompt_token_ids"],
                    mm_kwargs=inputs["mm_kwargs"],
                    mm_hashes=inputs["mm_hashes"],
                    mm_placeholders=inputs["mm_placeholders"],
                )
        elif inputs["type"] == "token":
            # Text-only inputs
            encoder_inputs = token_inputs(prompt="", prompt_token_ids=[])
            decoder_inputs = decoder_inputs_to_override or inputs
        else:
            assert_never(inputs)  # type: ignore[arg-type]
        return encoder_inputs, decoder_inputs

    def _process_encoder_decoder_prompt(
        self,
        prompt: PromptType,
    ) -> EncoderDecoderInputs:
        """
        For encoder/decoder models only:
        Process an input prompt into an :class:`EncoderDecoderInputs` instance.

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

        * :class:`EncoderDecoderInputs` instance
        """
        encoder_inputs: SingletonInputs
        decoder_inputs: Optional[SingletonInputs]

        if is_explicit_encoder_decoder_prompt(prompt):
            encoder_inputs = self._prompt_to_llm_inputs(
                prompt["encoder_prompt"])
            if (decoder_input := prompt["decoder_prompt"]) is None:
                decoder_inputs = None
            else:
                decoder_inputs = self._prompt_to_llm_inputs(decoder_input)
            # For multimodal model, override decoder prompt from processor
            # with explicit decoder prompt.
            if self.model_config.is_multimodal_model:
                encoder_inputs, decoder_inputs = (
                    self._separate_enc_dec_inputs_from_mm_processor_outputs(
                        encoder_inputs, decoder_inputs))
        else:
            inputs = self._prompt_to_llm_inputs(prompt)
            if self.model_config.is_multimodal_model:
                # Encoder-Decoder Multimodal model
                encoder_inputs, decoder_inputs = (
                    self._separate_enc_dec_inputs_from_mm_processor_outputs(
                        inputs))
            else:
                encoder_inputs = inputs

                decoder_inputs = None

        return self._build_enc_dec_llm_inputs(encoder_inputs, decoder_inputs)

    async def _process_encoder_decoder_prompt_async(
        self,
        prompt: PromptType,
    ) -> EncoderDecoderInputs:
        """Async version of :meth:`_process_encoder_decoder_prompt`."""
        encoder_inputs: SingletonInputs
        decoder_inputs: Optional[SingletonInputs]

        if is_explicit_encoder_decoder_prompt(prompt):
            encoder_task = self._prompt_to_llm_inputs_async(
                prompt["encoder_prompt"])

            if (decoder_input := prompt["decoder_prompt"]) is None:
                encoder_inputs = await encoder_task
                decoder_inputs = None
            else:
                decoder_task = self._prompt_to_llm_inputs_async(decoder_input)

                encoder_inputs, decoder_inputs = await asyncio.gather(
                    encoder_task, decoder_task)

            # For multimodal model, override decoder prompt from processor
            # with explicit decoder prompt.
            if self.model_config.is_multimodal_model:
                encoder_inputs, decoder_inputs = (
                    self._separate_enc_dec_inputs_from_mm_processor_outputs(
                        encoder_inputs, decoder_inputs))
        else:
            inputs = await self._prompt_to_llm_inputs_async(prompt)
            if self.model_config.is_multimodal_model:
                # Encoder-Decoder Multimodal model
                encoder_inputs, decoder_inputs = (
                    self._separate_enc_dec_inputs_from_mm_processor_outputs(
                        inputs))
            else:
                encoder_inputs = inputs

                decoder_inputs = None

        return self._build_enc_dec_llm_inputs(encoder_inputs, decoder_inputs)

    def _build_decoder_only_llm_inputs(
        self,
        prompt_inputs: DecoderOnlyInputs,
        prompt_adapter_request: Optional[PromptAdapterRequest],
    ) -> DecoderOnlyInputs:
        if (prompt_inputs["type"] == "token"
                or prompt_inputs["type"] == "multimodal"):
            prompt_inputs["prompt_token_ids"] = self._apply_prompt_adapter(
                prompt_inputs["prompt_token_ids"],
                prompt_adapter_request=prompt_adapter_request,
            )
        else:
            assert_never(prompt_inputs)  # type: ignore[arg-type]

        return prompt_inputs

    def _process_decoder_only_prompt(
        self,
        prompt: SingletonPrompt,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        return_mm_hashes: bool = False,
    ) -> DecoderOnlyInputs:
        """
        For decoder-only models:
        Process an input prompt into an :class:`DecoderOnlyInputs` instance.

        Arguments:

        * prompt: input prompt
        * lora_request
        * prompt_adapter_request
        * return_mm_hashes

        Returns:

        * :class:`DecoderOnlyInputs` instance
        """

        prompt_comps = self._prompt_to_llm_inputs(
            prompt,
            lora_request=lora_request,
            return_mm_hashes=return_mm_hashes,
        )

        return self._build_decoder_only_llm_inputs(
            prompt_comps,
            prompt_adapter_request=prompt_adapter_request,
        )

    async def _process_decoder_only_prompt_async(
        self,
        prompt: SingletonPrompt,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        return_mm_hashes: bool = False,
    ) -> DecoderOnlyInputs:
        """Async version of :meth:`_process_decoder_only_prompt`."""
        prompt_comps = await self._prompt_to_llm_inputs_async(
            prompt,
            lora_request=lora_request,
            return_mm_hashes=return_mm_hashes,
        )

        return self._build_decoder_only_llm_inputs(
            prompt_comps,
            prompt_adapter_request=prompt_adapter_request,
        )

    def preprocess(
        self,
        prompt: PromptType,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        return_mm_hashes: bool = False,
    ) -> ProcessorInputs:
        """Preprocess the input prompt."""
        if self.model_config.is_encoder_decoder:
            assert not return_mm_hashes, (
                "Multimodal hashes for encoder-decoder models should not be ",
                "returned until they are supported on vLLM V1.")
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder
            return self._process_encoder_decoder_prompt(prompt)

        if is_explicit_encoder_decoder_prompt(prompt):
            raise ValueError("Cannot pass encoder-decoder prompt "
                             "to decoder-only models")

        # Decoder-only operation
        return self._process_decoder_only_prompt(
            prompt,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            return_mm_hashes=return_mm_hashes,
        )

    async def preprocess_async(
        self,
        prompt: PromptType,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        return_mm_hashes: bool = False,
    ) -> ProcessorInputs:
        """Async version of :meth:`preprocess`."""
        if self.model_config.is_encoder_decoder:
            assert not return_mm_hashes, (
                "Multimodal hashes for encoder-decoder models should not be ",
                "returned until they are supported on vLLM V1.")
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder
            return await self._process_encoder_decoder_prompt_async(prompt)

        if is_explicit_encoder_decoder_prompt(prompt):
            raise ValueError("Cannot pass encoder-decoder prompt "
                             "to decoder-only models")

        # Decoder-only operation
        return await self._process_decoder_only_prompt_async(
            prompt,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            return_mm_hashes=return_mm_hashes,
        )
