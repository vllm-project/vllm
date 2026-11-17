# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import replace
from typing import Any, Final, cast

from vllm import (
    PoolingParams,
    PoolingRequestOutput,
    PromptType,
)
from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.lora.request import LoRARequest
from vllm.renderers import BaseRenderer, TokenizeParams, merge_kwargs
from vllm.renderers.chat_utils import ChatTemplateConfig
from vllm.renderers.inputs.preprocess import parse_model_prompt, prompt_to_seq
from vllm.utils.async_utils import make_async
from vllm.utils.mistral import is_mistral_tokenizer

from ..types import (
    AnyOfflineInputsContext,
    AnyRenderParam,
    ChatType,
    EncodeChatRenderParams,
    EncodeCMPLRenderParams,
    OfflineEncodeInputsContext,
    OfflineOutputsContext,
    PoolingChatLikeRequest,
    PoolingCompletionLikeRequest,
    PoolingEngineInput,
    PoolingServeContext,
    RequestFactory,
    RequestGenerator,
    ScoringRenderParams,
    is_conversations,
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

        self.template_kwargs = None
        self.tool_dicts = None

        # Shared thread pool executor for preprocessing
        self._executor: Executor = self.renderer._executor
        self.render_async = make_async(self.render, executor=self._executor)

    #######################################
    # online APIs

    def create_pooling_params(self, request):
        return request.to_pooling_params()

    def get_request_factory_online(
        self, ctx: PoolingServeContext
    ) -> Sequence[AnyRenderParam]:
        request = ctx.request
        renderer = self.renderer
        requests: Sequence[AnyRenderParam]

        if isinstance(request, PoolingChatLikeRequest):
            self._validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )

            num_requests = 1
            default_template_kwargs = merge_kwargs(
                self.template_kwargs,
                dict(
                    tools=self.tool_dicts,
                    tokenize=is_mistral_tokenizer(renderer.tokenizer),
                ),
            )

            mm_config = self.model_config.multimodal_config
            tok_params = request.build_tok_params(self.model_config)
            chat_params = request.build_chat_params(
                self.chat_template, self.chat_template_content_format
            ).with_defaults(
                default_template_kwargs,
                default_media_io_kwargs=(
                    mm_config.media_io_kwargs if mm_config else None
                ),
            )

            params_seq = self._params_to_seq(ctx.pooling_params, num_requests)
            seq_lora_requests = self._lora_request_to_seq(
                ctx.lora_request, num_requests
            )
            seq_priority = self._priority_to_seq(ctx.priorities, num_requests)

            requests = [
                EncodeChatRenderParams(
                    conversations=request.messages,
                    chat_params=chat_params,
                    tok_params=tok_params,
                    prompt_extras=ctx.prompt_extras,
                    skip_mm_cache=False,
                    params=params_seq[i],
                    lora_requests=seq_lora_requests[i],
                    priorities=seq_priority[i],
                )
                for i in range(num_requests)
            ]

            return requests

        elif isinstance(request, PoolingCompletionLikeRequest):
            model_config = self.model_config
            prompts_seq = prompt_to_seq(request.input)
            num_requests = len(prompts_seq)

            parsed_prompts = [
                (
                    prompt
                    if isinstance(prompt, bytes)
                    else parse_model_prompt(model_config, prompt)
                )
                for prompt in prompts_seq
            ]
            tok_params = request.build_tok_params(model_config)

            params_seq = self._params_to_seq(ctx.pooling_params, num_requests)
            seq_lora_requests = self._lora_request_to_seq(
                ctx.lora_request, num_requests
            )
            seq_priority = self._priority_to_seq(ctx.priorities, num_requests)

            requests = [
                EncodeCMPLRenderParams(
                    prompts=parsed_prompts[i],
                    tok_params=tok_params,
                    prompt_extras=ctx.prompt_extras,
                    skip_mm_cache=False,
                    params=params_seq[i],
                    lora_requests=seq_lora_requests[i],
                    priorities=seq_priority[i],
                )
                for i in range(num_requests)
            ]

            return requests
        else:
            raise ValueError(f"Invalid {self.name} request type")

    def post_process_online(
        self,
        ctx: PoolingServeContext,
    ):
        pass

    #######################################
    # offline APIs

    def get_request_factory_offline(
        self, ctx: AnyOfflineInputsContext
    ) -> tuple[RequestFactory, int]:
        assert isinstance(ctx, OfflineEncodeInputsContext)

        num_requests = len(ctx.inputs_seq)
        pooling_task = ctx.pooling_task

        pooling_params: PoolingParams | Sequence[PoolingParams]
        if ctx.pooling_params is None:
            pooling_params = PoolingParams()
        else:
            pooling_params = ctx.pooling_params

        # Clone to avoid modifying the caller's pooling parameters.
        params_seq = [
            param.clone() for param in self._params_to_seq(pooling_params, num_requests)
        ]

        for param in params_seq:
            if param.task is None:
                param.task = pooling_task
            elif pooling_task == "plugin":
                # `plugin` task uses io_processor.parse_data to verify inputs.
                # We actually allow plugin to overwrite pooling_task.
                pass
            elif param.task != pooling_task:
                msg = f"You cannot overwrite {param.task=!r} with {pooling_task=!r}!"
                raise VLLMValidationError(msg)

        seq_lora_requests = self._lora_request_to_seq(ctx.lora_request, num_requests)
        seq_priority = self._priority_to_seq(ctx.priorities, num_requests)

        # `add_special_tokens` is tri-state on the offline context:
        #   - `None`: keep the per-path default. For completion this means the
        #     renderer's `default_cmpl_tok_params` is used as-is, which for some
        #     multimodal models (Paligemma, Gemma4-MM, Ovis2.5, LFM2-VL, ...) is
        #     `add_special_tokens=False` to avoid a duplicate BOS. For chat this
        #     means `False`, mirroring the online `ChatRequestOptionsMixin`.
        #   - explicit `True` / `False`: overrides both paths.
        # User-provided `tokenization_kwargs` still wins over the default.
        def _build_tok_params(add_special_tokens: bool | None) -> TokenizeParams:
            tok_kwargs: dict[str, Any] = {}
            if add_special_tokens is not None:
                tok_kwargs["add_special_tokens"] = add_special_tokens
            tok_kwargs.update(ctx.tokenization_kwargs or {})
            if not tok_kwargs:
                return self.renderer.default_cmpl_tok_params
            return self.renderer.default_cmpl_tok_params.with_kwargs(**tok_kwargs)

        if ctx.add_special_tokens is not None:
            cmpl_add_special_tokens = ctx.add_special_tokens
            chat_add_special_tokens = ctx.add_special_tokens
        else:
            # completion: leave the renderer/model default untouched so that
            # multimodal models with `add_special_tokens=False` keep it.
            cmpl_add_special_tokens = None
            # chat: mirror the online default (`False`).
            chat_add_special_tokens = False

        cmpl_tok_params = _build_tok_params(cmpl_add_special_tokens)
        chat_tok_params = _build_tok_params(chat_add_special_tokens)

        extra_kwargs: dict[str, Any] = {}
        if isinstance(ctx.pooling_params, PoolingParams):
            extra_kwargs = dict(ctx.pooling_params.extra_kwargs or {})

        merged_chat_template_kwargs = merge_kwargs(
            extra_kwargs.get("chat_template_kwargs"),
            ctx.chat_template_kwargs,
        )

        prompt_extras_dict = dict(ctx.build_prompt_extras() or {})
        for key in ("mm_processor_kwargs", "cache_salt"):
            if key not in prompt_extras_dict and extra_kwargs.get(key) is not None:
                prompt_extras_dict[key] = extra_kwargs[key]
        prompt_extras: dict[str, Any] | None = prompt_extras_dict or None

        ctx = replace(ctx, chat_template_kwargs=merged_chat_template_kwargs)

        # Mirror `get_request_factory_online`:
        default_template_kwargs = merge_kwargs(
            self.template_kwargs,
            dict(
                tools=self.tool_dicts,
                tokenize=is_mistral_tokenizer(self.renderer.tokenizer),
            ),
        )
        mm_config = self.model_config.multimodal_config
        chat_params = ctx.build_chat_params(
            self.chat_template, self.chat_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=prompt_extras_dict.get("mm_processor_kwargs"),
        )

        def request_factory() -> RequestGenerator:
            for i in range(num_requests):
                prompt_or_conversations = ctx.inputs_seq[i]

                if is_conversations(prompt_or_conversations):
                    conversations: ChatType = prompt_or_conversations
                    yield EncodeChatRenderParams(
                        conversations=conversations,
                        chat_params=chat_params,
                        tok_params=chat_tok_params,
                        prompt_extras=prompt_extras,
                        skip_mm_cache=False,
                        params=params_seq[i],
                        lora_requests=seq_lora_requests[i],
                        priorities=seq_priority[i],
                    )
                else:
                    prompt: PromptType = cast(PromptType, prompt_or_conversations)
                    parsed_prompt = (
                        prompt
                        if isinstance(prompt, bytes)
                        else parse_model_prompt(self.model_config, prompt)
                    )
                    yield EncodeCMPLRenderParams(
                        prompts=parsed_prompt,
                        tok_params=cmpl_tok_params,
                        prompt_extras=prompt_extras,
                        skip_mm_cache=False,
                        params=params_seq[i],
                        lora_requests=seq_lora_requests[i],
                        priorities=seq_priority[i],
                    )

        return request_factory, num_requests

    def post_process_offline(
        self,
        ctx: OfflineOutputsContext,
    ) -> list[PoolingRequestOutput]:
        return ctx.outputs

    #######################################
    # helpers

    def render(
        self,
        render_params: EncodeCMPLRenderParams
        | EncodeChatRenderParams
        | ScoringRenderParams,
    ) -> PoolingEngineInput:
        if "conversations" in render_params:
            render_params = cast(EncodeChatRenderParams, render_params)
            (_,), engine_input = self.renderer.render_chat(
                conversations=[render_params["conversations"]],
                chat_params=render_params["chat_params"],
                tok_params=render_params["tok_params"],
                prompt_extras=render_params["prompt_extras"],
                skip_mm_cache=render_params["skip_mm_cache"],
            )
        elif "prompts" in render_params:
            render_params = cast(EncodeCMPLRenderParams, render_params)
            engine_input = self.renderer.render_cmpl(
                prompts=[render_params["prompts"]],
                tok_params=render_params["tok_params"],
                prompt_extras=render_params["prompt_extras"],
                skip_mm_cache=render_params["skip_mm_cache"],
            )
        else:
            raise ValueError(
                f"Unsupported render_params type {render_params.__class__.__name__}"
            )

        return PoolingEngineInput(
            prompts=engine_input[0],
            params=render_params["params"],
            lora_requests=render_params["lora_requests"],
            priorities=render_params["priorities"],
        )

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
            raise VLLMValidationError(
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
                raise VLLMValidationError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and params ({len(params)}) must be the same."
                )

            return params

        return [params] * num_requests

    def _lora_request_to_seq(
        self,
        lora_request: LoRARequest | None | Sequence[LoRARequest | None],
        num_requests: int,
    ) -> Sequence[LoRARequest | None]:
        if isinstance(lora_request, Sequence):
            if len(lora_request) != num_requests:
                raise VLLMValidationError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and lora_request ({len(lora_request)}) must be the same."
                )

            return lora_request

        return [lora_request] * num_requests

    def _priority_to_seq(
        self,
        priority: int | Sequence[int] | None,
        num_requests: int,
    ) -> Sequence[int]:
        if priority is not None:
            if isinstance(priority, int):
                return [priority] * num_requests

            if len(priority) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and priority ({len(priority)}) must be the same."
                )

            return priority

        return [0] * num_requests
