# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import Any, overload

from typing_extensions import assert_never

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalInputs,
    MultiModalUUIDDict,
)
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.renderers.inputs import (
    DecoderDictPrompt,
    DecoderOnlyDictPrompt,
    DictPrompt,
    EncoderDecoderDictPrompt,
    EncoderDictPrompt,
    SingletonDictPrompt,
    TokPrompt,
)
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt, parse_enc_dec_prompt
from vllm.tokenizers import TokenizerLike

from .data import (
    DecoderInputs,
    DecoderOnlyInputs,
    EmbedsInputs,
    EmbedsPrompt,
    EncoderDecoderInputs,
    EncoderInputs,
    ProcessorInputs,
    PromptType,
    SingletonInputs,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
    embeds_inputs,
    token_inputs,
)

logger = init_logger(__name__)


class InputPreprocessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        super().__init__()

        self.model_config = vllm_config.model_config
        self.renderer = renderer or renderer_from_config(vllm_config)
        self.mm_registry = mm_registry

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

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
                "Falling back on <BOS> for decoder start token id "
                "because decoder start token id is not available."
            )
            dec_start_token_id = self.renderer.get_bos_token_id()

        if dec_start_token_id is None:
            raise RuntimeError("Cannot find decoder start token id or <BOS>")

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

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer.tokenize_prompt(
            TextPrompt(prompt=prompt),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

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
        mm_processor = self.renderer.get_mm_processor()

        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        mm_items = mm_processor.info.parse_mm_data(mm_data)

        return mm_processor.apply(
            prompt,
            mm_items,
            hf_processor_mm_kwargs=mm_processor_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

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
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer.tokenize_prompt(
            TokensPrompt(prompt_token_ids=inputs),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

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

    @overload
    def _prompt_to_llm_inputs(
        self,
        prompt: EncoderDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> EncoderInputs: ...

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: DecoderDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> DecoderInputs: ...

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: DecoderOnlyDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> DecoderOnlyInputs: ...

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonDictPrompt,
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
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt_token_ids" in prompt:
            return self._process_tokens(
                prompt,  # type: ignore[arg-type]
                mm_uuids=mm_uuids,
            )

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(prompt)  # type: ignore[arg-type]

    def _validate_enc_inputs(self, inputs: SingletonInputs) -> EncoderInputs:
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

    def _validate_dec_inputs(self, inputs: SingletonInputs) -> DecoderInputs:
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
        enc_inputs = self._validate_enc_inputs(encoder_inputs)

        if decoder_inputs is None:
            dec_inputs: DecoderInputs = enc_inputs  # type: ignore[assignment]
        else:
            dec_inputs = self._validate_dec_inputs(decoder_inputs)

        enc_inputs_new: EncoderInputs
        dec_inputs_new: DecoderInputs

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
        prompt: EncoderDecoderDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> EncoderDecoderInputs:
        """
        For encoder/decoder models only:
        Process an input prompt into an
        [`EncoderDecoderInputs`][vllm.inputs.data.EncoderDecoderInputs]
        instance.

        Arguments:

        * prompt: an input prompt

        Returns:

        * [`EncoderDecoderInputs`][vllm.inputs.data.EncoderDecoderInputs]
          instance
        """
        encoder_prompt = prompt["encoder_prompt"]
        decoder_prompt = prompt["decoder_prompt"]

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
        prompt: DecoderOnlyDictPrompt,
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
        prompt: PromptType | DictPrompt | TokPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> ProcessorInputs:
        if self.model_config.is_encoder_decoder:
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder.
            return self._process_encoder_decoder_prompt(
                parse_enc_dec_prompt(prompt),
                tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        return self._process_decoder_only_prompt(
            parse_dec_only_prompt(prompt),
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def preprocess(
        self,
        prompt: PromptType | DictPrompt | TokPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> ProcessorInputs:
        """Preprocess the input prompt."""
        res = self._preprocess(prompt, tokenization_kwargs, mm_uuids=mm_uuids)

        self.renderer.update_mm_cache_stats()

        return res
