# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, overload

from typing_extensions import TypeVar

from vllm.inputs import (
    EmbedsInputs,
    EmbedsPrompt,
    EncoderDecoderInputs,
    ProcessorInputs,
    SingletonInputs,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
)
from vllm.inputs.data import build_enc_dec_inputs, embeds_inputs, token_inputs
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import AsyncMicrobatchTokenizer
from vllm.utils.counter import AtomicCounter
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.metrics.stats import MultiModalCacheStats

from .embed_utils import safe_load_prompt_embeds
from .inputs import (
    DictPrompt,
    EncoderDecoderDictPrompt,
    EncoderDecoderTokPrompt,
    SingletonDictPrompt,
    SingletonTokPrompt,
    TokPrompt,
)
from .inputs.preprocess import extract_target_prompt
from .params import ChatParams, TokenizeParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )
    from vllm.multimodal.cache import BaseMultiModalProcessorCache
    from vllm.multimodal.inputs import (
        MultiModalDataDict,
        MultiModalInputs,
        MultiModalUUIDDict,
    )
    from vllm.multimodal.parse import MultiModalDataItems, MultiModalUUIDItems
    from vllm.multimodal.processing import BaseMultiModalProcessor

logger = init_logger(__name__)


_T = TypeVar("_T", bound=TokenizerLike, default=TokenizerLike)


class BaseRenderer(ABC, Generic[_T]):
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: "VllmConfig",
        tokenizer_kwargs: dict[str, Any],
    ) -> "BaseRenderer":
        raise NotImplementedError

    def __init__(self, config: "VllmConfig", tokenizer: _T | None) -> None:
        super().__init__()

        self.config = config
        self.model_config = config.model_config

        self.tokenizer = tokenizer

        # Lazy initialization since offline LLM doesn't use async
        self._async_tokenizer: AsyncMicrobatchTokenizer | None = None

        self.mm_processor: BaseMultiModalProcessor | None = None
        self._mm_cache_stats: MultiModalCacheStats | None = None
        if config.model_config.is_multimodal_model:
            from vllm.multimodal import MULTIMODAL_REGISTRY as mm_registry
            from vllm.multimodal.registry import MultiModalTimingRegistry

            mm_processor_cache = mm_registry.processor_cache_from_config(config)

            with set_default_torch_num_threads():
                self.mm_processor = mm_registry.create_processor(
                    config.model_config,
                    tokenizer=tokenizer,
                    cache=mm_processor_cache,
                )

            if mm_processor_cache:
                self._mm_cache_stats = MultiModalCacheStats()

            # This is used to generate internal request ID for MM processing
            # It has no relation to the request ID for engine core
            self._mm_req_counter = AtomicCounter()
            self._mm_timing_registry = MultiModalTimingRegistry(
                config.observability_config
            )

    def get_tokenizer(self) -> _T:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def get_async_tokenizer(self) -> AsyncMicrobatchTokenizer:
        if self._async_tokenizer is None:
            self._async_tokenizer = AsyncMicrobatchTokenizer(self.get_tokenizer())

        return self._async_tokenizer

    def get_mm_processor(self) -> "BaseMultiModalProcessor":
        if self.mm_processor is None:
            raise ValueError("Multi-modal processor not available for text-only models")

        return self.mm_processor

    @property
    def mm_processor_cache(self) -> "BaseMultiModalProcessorCache | None":
        if self.mm_processor is None:
            return None

        return self.mm_processor.cache

    def stat_mm_cache(self) -> MultiModalCacheStats | None:
        mm_cache_stats = self._mm_cache_stats
        if mm_cache_stats is None:
            return None

        self._mm_cache_stats = MultiModalCacheStats()

        return mm_cache_stats

    def update_mm_cache_stats(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        mm_cache_stats = self._mm_cache_stats

        if mm_processor_cache and mm_cache_stats:
            delta = mm_processor_cache.make_stats(delta=True)
            mm_cache_stats.record(delta.total, delta.hits)

    def clear_mm_cache(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.clear_cache()

        if self._mm_cache_stats is not None:
            self._mm_cache_stats.reset = True

    def shutdown(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.close()

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

    def get_dec_start_token_id(self) -> int:
        """
        Obtain the decoder start token id employed by an encoder/decoder model,
        raising an error if it is not available.
        """
        dec_start_token_id = getattr(
            self.model_config.hf_config, "decoder_start_token_id", None
        )

        if dec_start_token_id is None:
            logger.warning_once(
                "Falling back on <BOS> for decoder start token id "
                "because decoder start token id is not available."
            )
            dec_start_token_id = self.get_bos_token_id()

        if dec_start_token_id is None:
            raise RuntimeError("Cannot find decoder start token id or <BOS>")

        return dec_start_token_id

    @cached_property
    def default_cmpl_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor
        if mm_processor is not None:
            return mm_processor.info.default_tok_params

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    @cached_property
    def default_chat_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor
        if mm_processor is not None:
            return mm_processor.info.default_tok_params

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=False,
        )

    # Step 1: Convert raw inputs to prompts
    def render_prompt(
        self,
        prompt: DictPrompt | bytes,
    ) -> DictPrompt:
        if isinstance(prompt, bytes):
            embeds = safe_load_prompt_embeds(self.model_config, prompt)
            prompt = EmbedsPrompt(prompt_embeds=embeds)

        return prompt

    def render_prompts(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return [self.render_prompt(prompt) for prompt in prompts]

    async def render_prompts_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        return self.render_prompts(prompts)

    @abstractmethod
    def render_messages(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], DictPrompt]:
        raise NotImplementedError

    async def render_messages_async(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], DictPrompt]:
        return self.render_messages(messages, params)

    # Step 2: Tokenize prompts if necessary
    def _tokenize_prompt(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        prompt_token_ids = tokenizer.encode(
            prompt["prompt"],
            **params.get_encode_kwargs(),
        )

        return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

    async def _tokenize_prompt_async(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_async_tokenizer()
        prompt_token_ids = await tokenizer.encode(
            prompt["prompt"],
            **params.get_encode_kwargs(),
        )

        return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)

    def _detokenize_prompt(self, prompt: TokensPrompt) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        prompt["prompt"] = tokenizer.decode(prompt["prompt_token_ids"])

        return prompt

    async def _detokenize_prompt_async(self, prompt: TokensPrompt) -> TokensPrompt:
        tokenizer = self.get_async_tokenizer()
        prompt["prompt"] = await tokenizer.decode(prompt["prompt_token_ids"])

        return prompt

    @overload
    def _tokenize_singleton_prompt(
        self,
        prompt: TextPrompt | TokensPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    @overload
    def _tokenize_singleton_prompt(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    def _tokenize_singleton_prompt(
        self,
        prompt: SingletonDictPrompt,
        params: TokenizeParams,
    ) -> SingletonTokPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = self._tokenize_prompt(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = self._detokenize_prompt(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    @overload
    async def _tokenize_singleton_prompt_async(
        self,
        prompt: TextPrompt | TokensPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    @overload
    async def _tokenize_singleton_prompt_async(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    async def _tokenize_singleton_prompt_async(
        self,
        prompt: SingletonDictPrompt,
        params: TokenizeParams,
    ) -> SingletonTokPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = await self._tokenize_prompt_async(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = await self._detokenize_prompt_async(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    def _tokenize_enc_dec_prompt(
        self,
        prompt: EncoderDecoderDictPrompt,
        params: TokenizeParams,
    ) -> EncoderDecoderTokPrompt:
        enc_prompt, dec_prompt = (
            self._tokenize_singleton_prompt(prompt["encoder_prompt"], params),
            (
                None
                if prompt["decoder_prompt"] is None
                else self._tokenize_singleton_prompt(prompt["decoder_prompt"], params)
            ),
        )

        return EncoderDecoderTokPrompt(
            encoder_prompt=enc_prompt,
            decoder_prompt=dec_prompt,
        )

    async def _tokenize_enc_dec_prompt_async(
        self,
        prompt: EncoderDecoderDictPrompt,
        params: TokenizeParams,
    ) -> EncoderDecoderTokPrompt:
        enc_prompt, dec_prompt = await asyncio.gather(
            self._tokenize_singleton_prompt_async(prompt["encoder_prompt"], params),
            (
                asyncio.sleep(0)
                if prompt["decoder_prompt"] is None
                else self._tokenize_singleton_prompt_async(
                    prompt["decoder_prompt"], params
                )
            ),
        )

        return EncoderDecoderTokPrompt(
            encoder_prompt=enc_prompt,
            decoder_prompt=dec_prompt,
        )

    def tokenize_prompt(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            return self._tokenize_enc_dec_prompt(prompt, params)  # type: ignore[arg-type]

        return self._tokenize_singleton_prompt(prompt, params)

    def tokenize_prompts(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return [self.tokenize_prompt(prompt, params) for prompt in prompts]

    async def tokenize_prompt_async(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            return await self._tokenize_enc_dec_prompt_async(prompt, params)  # type: ignore[arg-type]

        return await self._tokenize_singleton_prompt_async(prompt, params)

    async def tokenize_prompts_async(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )

    # Step 3: Add extra keys to the prompts
    def _apply_prompt_extras(
        self,
        prompts: Sequence[TokPrompt],
        prompt_extras: dict[str, Any] | None,
    ):
        if not prompt_extras:
            return

        for prompt in prompts:
            target_prompt = extract_target_prompt(self.model_config, prompt)
            target_prompt.update(prompt_extras)  # type: ignore[arg-type]

    # Step 4: Convert to engine inputs
    def _validate_mm_uuids(
        self,
        mm_data: "MultiModalDataDict",
        mm_data_items: "MultiModalDataItems",
        mm_uuid_items: "MultiModalUUIDItems",
    ) -> None:
        # NOTE: Keys corresponding to `None` in `mm_data` don't appear in
        # `mm_data_items`
        modalities = mm_data.keys() | mm_uuid_items.keys()

        for modality in modalities:
            data_items = mm_data_items.get(modality)
            uuid_items = mm_uuid_items.get(modality)

            if data_items is None:
                if uuid_items is None:
                    raise ValueError(
                        f"multi_modal_data[{modality!r}] is empty but "
                        f"multi_modal_uuids[{modality!r}] is missing."
                    )

            elif uuid_items is not None:
                if len(data_items) != len(uuid_items):
                    raise ValueError(
                        f"If given, multi_modal_uuids[{modality!r}] must have "
                        f"same length as multi_modal_data[{modality!r}], but "
                        f"got {len(uuid_items)} vs {len(data_items)}."
                    )

                for i, item in enumerate(data_items):
                    if item is None and uuid_items[i] is None:
                        raise ValueError(
                            f"multi_modal_data[{modality!r}][{i}] is empty but "
                            f"multi_modal_uuids[{modality!r}][{i}] is missing."
                        )

    def _process_mm_uuids(
        self,
        mm_data: "MultiModalDataDict",
        mm_data_items: "MultiModalDataItems",
        mm_uuid_items: "MultiModalUUIDItems",
        mm_req_id: str,
    ):
        model_config = self.model_config

        # NOTE: When users explicitly turn off BOTH prefix caching and input
        # processing caching, no multimodal features or embeddings will be
        # reused across requests, therefore identifying multimodal data items
        # by their content is no longer necessary, and we create uuids with
        # `<mm_req_id>-<modality>-<index>`, overriding even user-provided ones.
        if (
            model_config.multimodal_config
            and model_config.multimodal_config.mm_processor_cache_gb == 0
            and not self.config.cache_config.enable_prefix_caching
        ):
            mm_uuid_items = {
                modality: [f"{mm_req_id}-{modality}-{i}" for i in range(data_count)]
                for modality, data_count in mm_data_items.get_all_counts().items()
            }

        self._validate_mm_uuids(mm_data, mm_data_items, mm_uuid_items)

        return mm_uuid_items

    # TODO: Remove str and tokenization_kwargs after deprecating InputPreprocessor
    def _process_multimodal(
        self,
        prompt: list[int] | str,
        mm_data: "MultiModalDataDict",
        mm_uuids: "MultiModalUUIDDict | None",
        mm_processor_kwargs: Mapping[str, object] | None,
        tokenization_kwargs: dict[str, Any] | None,
    ) -> "MultiModalInputs":
        from vllm.multimodal.parse import parse_mm_uuids
        from vllm.multimodal.processing import ProcessorInputs as MMProcessorInputs

        mm_req_id = f"renderer-mm-{self._mm_req_counter.inc(1)}"

        mm_processor = self.get_mm_processor()

        mm_data_items = mm_processor.info.parse_mm_data(mm_data)
        mm_uuid_items = parse_mm_uuids(mm_uuids)

        mm_uuid_items = self._process_mm_uuids(
            mm_data, mm_data_items, mm_uuid_items, mm_req_id
        )

        mm_processor_inputs = MMProcessorInputs(
            prompt,
            mm_data_items,
            mm_uuid_items,
            hf_processor_mm_kwargs=mm_processor_kwargs or {},
            tokenization_kwargs=tokenization_kwargs or {},
        )
        mm_timing_ctx = self._mm_timing_registry.get(mm_req_id)

        with set_default_torch_num_threads():
            mm_inputs = mm_processor.apply(mm_processor_inputs, mm_timing_ctx)

        self.update_mm_cache_stats()

        return mm_inputs

    def _process_tokens(
        self,
        prompt: TokensPrompt,
    ) -> "TokenInputs | MultiModalInputs":
        prompt_token_ids = prompt["prompt_token_ids"]

        inputs: TokenInputs | MultiModalInputs
        if multi_modal_data := prompt.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                mm_processor_kwargs=prompt.get("mm_processor_kwargs"),
                tokenization_kwargs=None,  # Tokenization already done in Step 2
                mm_uuids=prompt.get("multi_modal_uuids"),
            )
        else:
            inputs = token_inputs(prompt_token_ids)

        if prompt_text := prompt.get("prompt"):
            inputs["prompt"] = prompt_text
        if cache_salt := prompt.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_embeds(
        self,
        prompt: EmbedsPrompt,
    ) -> EmbedsInputs:
        if not self.model_config.enable_prompt_embeds:
            raise ValueError(
                "You must set `--enable-prompt-embeds` to input `prompt_embeds`."
            )

        prompt_embeds = prompt["prompt_embeds"]

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
            prompt_embeds=prompt_embeds,
            cache_salt=prompt.get("cache_salt"),
        )

    def _process_singleton(
        self,
        prompt: SingletonTokPrompt,
    ) -> SingletonInputs:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        return self._process_tokens(prompt)  # type: ignore[arg-type]

    def _process_enc_dec(
        self,
        prompt: EncoderDecoderTokPrompt,
    ) -> EncoderDecoderInputs:
        enc_prompt = prompt["encoder_prompt"]
        dec_prompt = prompt["decoder_prompt"]

        return build_enc_dec_inputs(
            encoder_inputs=self._process_singleton(enc_prompt),
            decoder_inputs=(
                None if dec_prompt is None else self._process_singleton(dec_prompt)
            ),
            decoder_start_token_id=self.get_dec_start_token_id(),
        )

    def process_for_engine(
        self, prompt: TokPrompt, arrival_time: float
    ) -> ProcessorInputs:
        engine_prompt: ProcessorInputs
        if "encoder_prompt" in prompt:
            engine_prompt = self._process_enc_dec(prompt)  # type: ignore[arg-type]
        else:
            engine_prompt = self._process_singleton(prompt)

        engine_prompt["arrival_time"] = arrival_time

        return engine_prompt

    # Top-level methods
    def render_cmpl(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = self.render_prompts(prompts)
        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return [self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts]

    async def render_cmpl_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = await self.render_prompts_async(prompts)
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return [self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts]

    def render_chat(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in rendered:
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = [
            self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts
        ]

        return out_conversations, eng_prompts

    async def render_chat_async(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages_async(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in await asyncio.gather(*rendered):
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = [
            self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts
        ]

        return out_conversations, eng_prompts
