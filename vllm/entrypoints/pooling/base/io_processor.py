# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any, Final, Literal, cast

from vllm import PoolingParams, PoolingRequestOutput, PromptType
from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateConfig,
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.openai.engine.serving import RendererChatRequest, RendererRequest
from vllm.inputs import EngineInput, SingletonPrompt, TokensInput
from vllm.renderers import BaseRenderer, TokenizeParams, merge_kwargs
from vllm.renderers.inputs.preprocess import parse_model_prompt, prompt_to_seq
from vllm.tool_parsers import ToolParser
from vllm.utils.mistral import is_mistral_tokenizer

from ..scoring.typing import ScoringData
from ..typing import (
    OfflineInputsContext,
    OfflineOutputsContext,
    PoolingChatLikeRequest,
    PoolingCompletionLikeRequest,
    PoolingServeContext,
)


class PoolingIOProcessor:
    """Processor for handling preprocessing & postprocessing ops for pooling requests.

    This class manages both online (serving) and offline (batch) processing of pooling
    requests, handling chat and completion formats.
    """

    name: str

    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.renderer = renderer

        self.chat_template = chat_template_config.chat_template
        self.chat_template_content_format: Final = (
            chat_template_config.chat_template_content_format
        )
        self.trust_request_chat_template = (
            chat_template_config.trust_request_chat_template
        )

    #######################################
    # online APIs

    def create_pooling_params(self, request):
        return request.to_pooling_params()

    def pre_process_online(self, ctx: PoolingServeContext):
        request = ctx.request

        if isinstance(request, PoolingChatLikeRequest):
            self._validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            _, engine_inputs = self._preprocess_chat_online(
                request,
                request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=None,
            )
        elif isinstance(request, PoolingCompletionLikeRequest):
            engine_inputs = self._preprocess_cmpl_online(
                request,
                prompt_input=request.input,
                prompt_embeds=None,
            )
        else:
            raise ValueError(f"Invalid {self.name} request type")

        ctx.engine_inputs = engine_inputs

    def post_process_online(
        self,
        ctx: PoolingServeContext,
    ):
        pass

    #######################################
    # offline APIs

    def pre_process_offline(self, ctx: OfflineInputsContext) -> Sequence[EngineInput]:
        assert not isinstance(ctx.prompts, ScoringData) and not (
            isinstance(ctx.prompts, dict) and "data" in ctx.prompts
        )

        prompts_seq = prompt_to_seq(ctx.prompts)
        tokenization_kwargs = dict(ctx.tokenization_kwargs or {})
        input_type = tokenization_kwargs.pop("input_type", None)
        if input_type is not None and self.name != "token_embed":
            raise ValueError("input_type is only supported with task 'token_embed'.")
        tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(
            **tokenization_kwargs
        )
        prompt_extras = {"input_type": input_type} if input_type is not None else None
        return self._preprocess_cmpl_offline(
            prompts=prompts_seq,
            tok_params=tok_params,
            prompt_extras=prompt_extras,
        )

    def post_process_offline(
        self,
        ctx: OfflineOutputsContext,
    ) -> list[PoolingRequestOutput]:
        return ctx.outputs

    #######################################
    # helpers

    def _preprocess_cmpl_online(
        self,
        request: RendererRequest,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
    ) -> list[EngineInput]:
        renderer = self.renderer
        model_config = self.model_config

        prompts = list[SingletonPrompt | bytes]()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = request.build_tok_params(model_config)

        engine_inputs = renderer.render_cmpl(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )
        self._apply_colbert_input_type(
            engine_inputs,
            getattr(request, "input_type", None),
        )
        return engine_inputs

    def _preprocess_chat_online(
        self,
        request: RendererChatRequest,
        messages: list[ChatCompletionMessageParam],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: type[ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[EngineInput]]:
        renderer = self.renderer

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=is_mistral_tokenizer(renderer.tokenizer),
            ),
        )

        mm_config = self.model_config.multimodal_config

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
        )

        (conversation,), (engine_input,) = renderer.render_chat(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        return conversation, [engine_input]

    def _preprocess_cmpl_offline(
        self,
        prompts: PromptType | Sequence[PromptType],
        tok_params: TokenizeParams,
        prompt_extras: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        prompts = prompt_to_seq(prompts)
        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(self.model_config, prompt)
            )
            for prompt in prompts
        ]

        engine_inputs = self.renderer.render_cmpl(
            parsed_prompts, tok_params, prompt_extras=prompt_extras
        )
        input_type = None if prompt_extras is None else prompt_extras.get("input_type")
        self._apply_colbert_input_type(engine_inputs, input_type)
        return engine_inputs

    def _apply_colbert_input_type(
        self,
        engine_inputs: Sequence[EngineInput],
        input_type: Literal["query", "document"] | None,
    ) -> None:
        if input_type is None:
            return

        if input_type not in ("query", "document"):
            raise ValueError("input_type must be 'query' or 'document'")
        if not self._is_colbert_model():
            raise ValueError("input_type is only supported for ColBERT models.")

        for engine_input in engine_inputs:
            if engine_input["type"] == "enc_dec":
                encoder_input = engine_input["encoder_prompt"]
                if encoder_input["type"] != "token":
                    raise ValueError(
                        "input_type is only supported for tokenized prompts"
                    )
                encoder_tokens = cast(TokensInput, encoder_input)
                encoder_tokens["prompt_token_ids"] = self._colbert_token_ids(
                    list(encoder_tokens["prompt_token_ids"]),
                    input_type,
                )
                continue

            if engine_input["type"] != "token":
                raise ValueError("input_type is only supported for tokenized prompts")

            token_input = cast(TokensInput, engine_input)
            token_input["prompt_token_ids"] = self._colbert_token_ids(
                list(token_input["prompt_token_ids"]),
                input_type,
            )

    def _colbert_token_ids(
        self,
        token_ids: list[int],
        input_type: Literal["query", "document"],
    ) -> list[int]:
        prefix_id = self._colbert_prefix_token_id(input_type)

        if input_type == "query":
            max_base_len = 31 if prefix_id is not None else 32
            token_ids = self._truncate_colbert_query(token_ids, max_base_len)
        elif prefix_id is not None:
            # Reserve one token slot for the document marker to keep length stable.
            token_ids = self._truncate_colbert_query(token_ids, len(token_ids) - 1)

        if prefix_id is not None:
            token_ids = token_ids[:1] + [prefix_id] + token_ids[1:]

        if input_type == "query":
            pad_id = self._colbert_query_pad_token_id()
            token_ids += [pad_id] * max(32 - len(token_ids), 0)

        return token_ids

    def _is_colbert_model(self) -> bool:
        colbert_archs = {
            "HF_ColBERT",
            "ColBERTModel",
            "ColBERTModernBertModel",
            "ColBERTJinaRobertaModel",
            "ColBERTLfm2Model",
        }
        arch = getattr(self.model_config, "architecture", None)
        hf_archs = (
            getattr(getattr(self.model_config, "hf_config", None), "architectures", ())
            or ()
        )
        return arch in colbert_archs or any(a in colbert_archs for a in hf_archs)

    def _truncate_colbert_query(
        self,
        token_ids: list[int],
        max_len: int,
    ) -> list[int]:
        if max_len <= 0:
            return []
        if len(token_ids) <= max_len:
            return token_ids

        tokenizer = self.renderer.tokenizer
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        sep_token_id = getattr(tokenizer, "sep_token_id", None)
        if token_ids[-1] in (eos_token_id, sep_token_id):
            return token_ids[: max_len - 1] + [token_ids[-1]]

        return token_ids[:max_len]

    def _colbert_prefix_token_id(
        self,
        input_type: Literal["query", "document"],
    ) -> int | None:
        candidates = {
            "query": ("[QueryMarker]", "[Q]", "[unused0]"),
            "document": ("[DocumentMarker]", "[D]", "[unused1]"),
        }[input_type]
        tokenizer = self.renderer.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is required for ColBERT input_type support.")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)

        for token in candidates:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id != unk_token_id:
                return token_id

        return None

    def _colbert_query_pad_token_id(self) -> int:
        tokenizer = self.renderer.tokenizer
        token_id = getattr(tokenizer, "mask_token_id", None)
        if token_id is not None:
            return token_id
        raise ValueError("ColBERT query expansion requires tokenizer.mask_token_id.")

    def _validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ):
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            raise ValueError(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    def _params_to_seq(
        self,
        params: PoolingParams | Sequence[PoolingParams],
        num_requests: int,
    ) -> Sequence[PoolingParams]:
        if isinstance(params, Sequence):
            if len(params) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and params ({len(params)}) must be the same."
                )

            return params

        return [params] * num_requests
