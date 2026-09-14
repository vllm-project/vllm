# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import Any, Final, cast

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    make_tool_call_id,
)
from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    FunctionCall,
    PerRequestMetrics,
    RequestResponseMetadata,
    ToolCall,
)
from vllm.entrypoints.generate.base.serving import (
    GenerateBaseServing,
    build_per_request_timing_metrics,
    build_spec_decoding_metrics,
    clamp_prompt_logprobs,
    format_token_id_placeholder,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import (
    CompletionTokenUsageInfo,
    ErrorResponse,
    PromptTokenUsageInfo,
    UsageInfo,
)
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens, should_include_usage
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)
from vllm.exceptions import GenerationError
from vllm.inputs import EngineInput, MultiModalPlaceholders
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import as_list
from vllm.utils.serial_utils import numpy2base64

# [CN] OpenAI Chat Completions API 的服务实现。
# [CN] 职责边界：本文件只做「协议翻译」——把 OpenAI 的请求/响应结构与
# [CN] vLLM 的 RequestOutput 互转；实际的渲染交给 OnlineRenderer，
# [CN] token 级解析交给 Parser（工具调用 / 推理内容分离）。
# [CN] 两条主线：chat_completion_full_generator（非流式）与
# [CN] chat_completion_stream_generator（SSE 流式），二者共享使用量统计逻辑。

logger = init_logger(__name__)


# [CN] 按模态汇总多模态占位符占用的 token 数。

def _get_mm_token_counts(engine_input: EngineInput) -> dict[str, int]:
    """Sum per-modality placeholder tokens from ``mm_placeholders``.

    Keyed by modality name; ``PlaceholderRange.length`` is the placeholder's
    prompt token span, so each sum matches the placeholder tokens already
    counted in ``usage.prompt_tokens``.
    """
    # [CN] 类型提示而已：mm_placeholders 是可选字段，缺失时按空字典处理。

    mm_placeholders = cast(
        MultiModalPlaceholders | None, engine_input.get("mm_placeholders")
    )
    return {
        # [CN] PlaceholderRange.length 是占位符在 prompt 里占的 token 跨度，
        # [CN] 所以这里的求和与 usage.prompt_tokens 里已经计入的部分是同一批 token，
        # [CN] 不能再重复相加，只能作为明细展示。

        modality: sum(p.length for p in ranges)
        for modality, ranges in (mm_placeholders or {}).items()
        if ranges
    }


# [CN] 构造 prompt_tokens_details 明细字段。

def _make_prompt_tokens_details(
    enable_prompt_tokens_details: bool,
    num_cached_tokens: int | None,
    num_cache_creation_tokens: int | None,
    mm_token_counts: dict[str, int] | None,
) -> PromptTokenUsageInfo | None:
    """Build ``prompt_tokens_details`` from cached + multimodal token counts."""
    # [CN] 开关优先：即使有数据也不输出。这是为了避免改变默认响应形状。

    if not enable_prompt_tokens_details:
        return None
    # [CN] 三个来源都为空时返回 None 而非全零对象 —— 全零明细会误导调用方。

    if (
        num_cached_tokens is None
        and num_cache_creation_tokens is None
        and not mm_token_counts
    ):
        return None
    return PromptTokenUsageInfo(
        cached_tokens=num_cached_tokens,
        created_cache_tokens=num_cache_creation_tokens,
        multimodal_tokens=mm_token_counts or None,
    )


# [CN] 推理 token 明细。目前只承载 reasoning_tokens 一项。

def _make_completion_tokens_details(
    reasoning_tokens: int,
) -> CompletionTokenUsageInfo:
    return CompletionTokenUsageInfo(reasoning_tokens=reasoning_tokens)


# [CN] 继承 GenerateBaseServing，复用基类里的模型校验、采样参数构造、日志等能力。

class OpenAIServingChat(GenerateBaseServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        online_renderer: "OnlineRenderer",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        enable_per_request_metrics: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.online_renderer = online_renderer
        self.response_role = response_role
        self.chat_template = chat_template
        # [CN] Final 而非普通属性：运行期不允许被改动，防止不同请求互相污染模板格式。

        self.chat_template_content_format: Final = chat_template_content_format
        # [CN] 是否允许请求自带 chat_template。默认关：模板里可执行 Jinja，有安全风险。

        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        # [CN] 增量日志默认开，但它与完整响应日志会重复记录，故两者是独立开关。

        self.enable_log_deltas = enable_log_deltas

        # [CN] auto 模式下由模型输出触发工具调用（而非 probability schema），
        # [CN] 需要 parser 能从自由文本里识别出工具调用片段。

        self.enable_auto_tools: bool = enable_auto_tools
        # [CN] 只有配了 reasoning_parser 才有意义，故用 bool(reasoning_parser) 推导。

        self._include_reasoning_tokens_details = bool(reasoning_parser)
        # [CN] 一次性把工具解析器与推理解析器打包成一个 Parser 类。
        # [CN] is_harmony 走 gpt_oss 的特殊分支：它的工具调用语法与其它模型完全不同。

        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
            is_harmony=self.model_config.hf_config.model_type == "gpt_oss",
        )
        # [CN] tool_choice=none 时是否彻底不下发工具定义。有些模型即使不用工具，
        # [CN] 看到工具定义也会影响输出分布，所以这里做成可配置。

        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.enable_per_request_metrics = enable_per_request_metrics
        # [CN] 取 generation_config 里与 vLLM 默认值不同的那部分作为请求默认值。
        # [CN] 用 diff 而非全量，避免把模型没指定的项也强加给用户请求。

        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        # [CN] generation_config 有三种取值："auto"(读 HF)、"vllm"(用内置默认)、
        # [CN] 或直接给字典 override_generation_config。三者要分开处理 max_tokens 上限。

        mc = self.model_config
        # [CN] generation_config="auto" 或 "vllm" 时，用户没有自己的 max_new_tokens，
        # [CN] 此时改为读 override_generation_config，避免拿到 None 导致后续比较报错。

        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        # [CN] 浏览是 OpenAI Responses API 的能力，Chat API 明确不支持。

        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        # [CN] 同理，代码解释器也只在 Responses API 提供。

        self.python_tool = None

    # [CN] 优先级链：请求自带 -> 服务端 default_chat_template_kwargs -> 模板默认。

    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    # [CN] 这个 hook 存在的唯一理由是进程边界：同一份 kwargs 要既给本进程的 parser，
    # [CN] 又要跨 ZMQ 以 msgpack 发给 engine core。msgpack 编码不了的对象、
    # [CN] 以及只有服务端 parser 需要的请求级状态，可以在这里裁掉。
    # [CN] 注意必须返回新字典而不是原地修改 —— 调用方还需要完整的那份。

    def _engine_chat_template_kwargs(
        self, chat_template_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Subclass hook to narrow ``chat_template_kwargs`` for the engine.

        The same dict is used twice: to build the API-server-side parser
        instance, and to populate
        ``EngineCoreRequest.reasoning_parser_kwargs`` for the engine-core
        side. The latter crosses ZMQ as msgpack, so a handler that stashes
        request-scoped state only the API-server-side parser needs (values
        msgpack can't encode, or payloads not worth shipping) can drop
        those entries here without affecting the in-process parser.

        Must not mutate the argument -- the caller still needs the full
        dict. The default forwards it unchanged.
        """
        return chat_template_kwargs

    # [CN] 返回联合类型而非抛异常：错误响应要作为正常值回传给 HTTP 层，
    # [CN] 这样调用方能用统一的 isinstance 判断，不必依赖异常流控。

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OnlineRenderer, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
            or an ErrorResponse on failure.
        """
        # [CN] 模型存在性 / LoRA 合法性 / 引擎健康，三合一的前置检查。

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # [CN] n 为空时算 1 份。预检会校验 n 是否超过 max_num_seqs，避免下发后被拒。

        self._preflight(request.n or 1)

        # [CN] 渲染结果可能拆成多个 engine_input（多图/多段→多请求），这是 n>1 之外的
        # [CN] 另一种一对多情形，后续靠 sub_request_id 区分。

        return await self.online_renderer.render_chat(request)

    # [CN] 外层包装：KV 传输失败时要做清理回调，真正的实现在 _create_chat_completion。

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion(request, raw_request), request, raw_request
        )

    # [CN] 返回类型也是联合：流式时返回 AsyncGenerator，非流式返回完整响应对象。

    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # Streaming response
        # [CN] 即使引擎开了 skip_tokenizer_init，API 层仍需 tokenizer 来做增量解码与
        # [CN] 工具调用解析，所以这里断言它必须存在。

        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        # [CN] 提前构造 parser 是因为后面判断 reasoning_ended 需要它，
        # [CN] 而不是等到拿到输出才建 —— 少一次延迟。

        parser: Parser | None = None
        if self.parser_cls is not None:
            parser = self.parser_cls(
                tokenizer,
                request.tools,
                chat_template_kwargs=chat_template_kwargs,
                model_config=self.model_config,
            )
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result

        # [CN] chatcmpl- 前缀是 OpenAI 约定，客户端常拿它判断响应类型。

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        # [CN] 挂到 raw_request.state 上，供 FastAPI 中间件在响应结束后统计用量。
        # [CN] 这条路径是绕过返回值传递数据的：生成器里无法直接回传，只能借 request state。

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # [CN] supports_default_mm_loras=True：允许把多模态 LoRA 作为默认适配器。

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        # [CN] 路由器可以在 HTTP 头里注入目标 DP rank，实现亲和性调度。

        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        # [CN] 循环外定义是因为多个 engine_input 只需最后一个的多模态计数用于明细。

        mm_token_counts: dict[str, int] | None = None
        # [CN] 一次 API 调用可能展开成多个引擎请求（多模态分段），逐个下发。

        for i, engine_input in enumerate(engine_inputs):
            # [CN] 这里取出的是原始 prompt token，用于判断推理内容是否已经结束。

            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
            mm_token_counts = _get_mm_token_counts(engine_input)

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            # [CN] 单个输入时复用原 request_id，避免给客户端造成"我明明发了 1 条"的困惑。

            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            # [CN] max_completion_tokens 优先级高于 max_tokens（OpenAI 新字段），
            # [CN] 再结合模型长度上限、prompt 长度、truncate_prompt_tokens 共同夹逼。

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            # [CN] beam search 走完全不同的引擎调用：它不能用普通 SamplingParams。

            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            # [CN] 只在存在 raw_request 时才提取链路追踪头，纯引擎内部调用会跳过。

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            session_id = self._get_session_id(request, raw_request)

            # [CN] beam search 是同步迭代接口，与流式 SSE 的路径完全不同。

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    session_id=session_id,
                )
            else:
                # [CN] reasoning_ended 三态：True=已结束，None=不知道（交给引擎自行判断）。
                # [CN] 不输出推理内容时直接认定"推理已结束"，引擎就可以跳过推理分隔符的等待。

                if not request.include_reasoning:
                    reasoning_ended = True
                # [CN] Mistral 的 grammar 里已经有可选的 think? 规则，能同时覆盖两种输出形态，
                # [CN] 因此不需要引擎侧再做推理结束判断。

                elif request._grammar_from_parser:
                    # The Mistral grammar already includes an optional
                    # `think?` rule that handles both reasoning and
                    # non-reasoning outputs.
                    reasoning_ended = True
                # [CN] 常规路径：看 prompt 里推理区块是否已经闭合。

                elif parser is not None and parser.reasoning_parser is not None:
                    reasoning_ended = parser.is_reasoning_end(prompt_token_ids or [])
                # [CN] reasoning 为空时退化为普通内容。

                else:
                    reasoning_ended = None

                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=self._get_priority(request, raw_request),
                    data_parallel_rank=data_parallel_rank,
                    session_id=session_id,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": self._engine_chat_template_kwargs(
                            chat_template_kwargs
                        ),
                    }
                    if parser is not None and parser.reasoning_parser is not None
                    else None,
                )

            # [CN] 收集所有 generator，下面立刻断言只有一个 —— 多引擎请求与流式组合
            # [CN] 目前不支持，用断言明确卡死而不是静默只处理第一条。

            generators.append(generator)

        # [CN] 硬性约束：Chat API 只支持单个结果流。assert 而非抛错，说明这是内部不变式。

        assert len(generators) == 1
        (result_generator,) = generators

        # [CN] 注意流式返回的是生成器对象本身（不 await），由 FastAPI 逐块推送；
        # [CN] 非流式则要 await 到拿到最终结果。

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                chat_template_kwargs=chat_template_kwargs,
                mm_token_counts=mm_token_counts,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
        )

    # [CN] add_generation_prompt 为真表示模板会追加 assistant 起始标记，
    # [CN] 此时角色固定用 response_role；否则沿用最后一条消息的角色。

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    # [CN] 所有构造点都走这个方法，是为了让子类能替换 ChatMessage 具体类型
    # [CN] （如 Cohere v2 的 CohereChatMessage）而不用复制下面那套工具分支判断。

    def _create_chat_message(self, *args: Any, **kwargs: Any) -> ChatMessage:
        """Construct the response :class:`ChatMessage` for the non-streaming path.

        The full-generator calls this at every construction site so
        subclasses can swap in a specialized :class:`ChatMessage`
        subclass (e.g. :class:`CohereServingChatV2` returning
        :class:`CohereChatMessage`) without duplicating the branchy
        tool-choice / auto-tools logic that decides which fields are
        populated. The default returns a plain :class:`ChatMessage`.
        """
        return ChatMessage(*args, **kwargs)

    # [CN] 后置 hook：用于注入 (reasoning, content, tool_calls) 三元组装不下的
    # [CN] 额外信息，例如 parser 在推理阶段缓存下来的引用溯源。基类是空操作。

    def _finalize_response_message(
        self,
        message: ChatMessage,
        *,
        parser: Parser | None,
    ) -> ChatMessage:
        """Subclass hook to enrich a fully-constructed :class:`ChatMessage`.

        Default is a no-op. Subclasses that need to surface parser-side
        extras (e.g. :class:`CohereServingChatV2` reading grounding
        citations off the reasoning parser and populating
        :class:`CohereChatMessage.citations`) override this to inspect
        ``parser`` and mutate/replace ``message``.
        """
        return message

    # [CN] SSE 流式主控。难点在于要在不知道总长度的情况下，
    # [CN] 一边累加 previous_xxx 状态一边增量产出 delta。

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        chat_template_kwargs: dict[str, Any] | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> AsyncGenerator[str, None]:
        # [CN] 秒级时间戳。整个响应所有块共用同一个值，代表请求开始时刻而非产出时刻。

        created_time = int(time.time())
        # [CN] 每个 SSE 块的 object 字段固定为 chat.completion.chunk。

        chunk_object_type: Final = "chat.completion.chunk"
        # [CN] 首轮要额外承担「发角色块」与「抓取缓存统计」两件一次性工作。

        first_iteration = True

        # Send response for each token for each request.n (index)
        # [CN] n 个并行序列各自维护一套状态，所有 previous_* 都是列表而非标量。

        num_choices = 1 if request.n is None else request.n
        # [CN] 已交付 token 数的累加器。之所以用累加而非直接读累积字段，
        # [CN] 是因为部分迭代器（分块 prefill）会出现重复载荷。

        previous_num_tokens = [0] * num_choices
        # TODO: Remove once all reasoning parsers use the Parser Engine.
        # [CN] 需要保留完整 token 历史：推理 token 计数要反复回算，
        # [CN] 因为增量解析无法只凭当前片段判断推理是否还在继续。

        generated_token_ids: list[list[int]] = [[] for _ in range(num_choices)]
        previous_reasoning_tokens = [0] * num_choices
        # [CN] 每个序列只能发一次 finish_reason，用这个标志屏蔽后续迭代。

        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        # [CN] 缓存命中数只在第一次迭代取：后续 RequestOutput 不再重复携带该信息。

        num_cached_tokens = None
        num_cache_creation_tokens = None
        # [CN] 用于决定 finish_reason 是 tool_calls 还是 stop —— 只有当 parser
        # [CN] 真的吐出过工具调用片段时才算。

        tools_streamed = [False] * num_choices

        # [CN] 具名工具调用时 OpenAI 规定 finish_reason 是 stop 而非 tool_calls，
        # [CN] 因此要提前记住名字以便后面区分两种情形。

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # [CN] 保留完整文本用于末尾写日志；中间过程的 delta 不足以还原全貌。

        previous_texts = [""] * num_choices

        # [CN] parser 构造失败要作为首块 SSE 错误发出，而不是让连接直接断开。

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                parsers: list[Parser | None] = [
                    self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                        model_config=self.model_config,
                    )
                    for _ in range(num_choices)
                ]
            # [CN] 兜底分支：理论上不可达，但协议兼容性上宁可降级为普通消息也不报错。

            else:
                parsers = [None] * num_choices
        # [CN] 捕获所有异常并转成 SSE 错误块：生成器里抛出的异常无法被 FastAPI
        # [CN] 的错误处理器截获（响应头已经发出），只能自己兜底。

        except Exception as e:
            logger.exception("Error in parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        # [CN] include_usage / continuous_usage_stats 等非标准扩展从这里读。

        stream_options = request.stream_options
        # [CN] 两个开关独立：continuous usage 是每块都带，include_usage 只在末尾一块带。

        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        # [CN] 保留最后一个 RequestOutput 是因为结尾要从中取 metrics 与投机解码统计。

        last_res: RequestOutput | None = None
        # [CN] 主循环。异常必须在生成器内部捕获成 SSE 错误块，
        # [CN] 否则客户端只会看到连接中断，拿不到任何错误信息。

        try:
            async for res in result_generator:
                last_res = res
                # [CN] 编码器-解码器模型还要加上 encoder 侧 prompt token，
                # [CN] 否则 usage.prompt_tokens 会偏小。

                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                # [CN] 首块必须先发角色 delta。注意：此刻就要记录 num_cached_tokens，
                # [CN] 因为一旦后续迭代抛异常，外层 catch 会把错误作为第一响应发出，
                # [CN] 缓存信息就再也拿不到了。

                if first_iteration:
                    # [CN] 前缀缓存命中数。必须在首轮取：后续 RequestOutput 不再携带。

                    num_cached_tokens = res.num_cached_tokens
                    num_cache_creation_tokens = res.num_cache_creation_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    # [CN] OpenAI 协议要求首块是 role-only 的空 delta，客户端据此建立角色。

                    role = self.get_chat_request_role(request)

                    # ``res.prompt`` is the rendered chat-templated prompt
                    # [CN] 这里的 res.prompt 是渲染后的完整提示词，而非用户原始输入。

                    prompt_text = res.prompt if request.return_prompt_text else None

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    # [CN] n 个序列各自发一条 choice=0..n-1 的角色块，index 用来重建对应关系。

                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                            prompt_text=prompt_text,
                        )

                        # if continuous usage stats are requested, add it
                        # [CN] 连续用量模式下每块都要带 usage，客户端可据此实时计费。

                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                                completion_tokens_details=(
                                    _make_completion_tokens_details(0)
                                    if self._include_reasoning_tokens_details
                                    else None
                                ),
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    # [CN] echo 只有在最后一条消息的 role 与 assistant 角色一致时才回显内容，
                    # [CN] 否则把 user 的话塞进 assistant 回复里在语义上是错的。

                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                        completion_tokens_details=(
                                            _make_completion_tokens_details(0)
                                            if self._include_reasoning_tokens_details
                                            else None
                                        ),
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                # [CN] 每次迭代携带的是截至目前为止的完整累积文本，下面用 previous_* 做差分量。

                for output in res.outputs:
                    i = output.index
                    parser = parsers[i]
                    # [CN] 已终结的序列不再处理，避免发送 finish 之后的残留 delta
                    # [CN] （后续迭代仍会携带该序列的数据）。
                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and (
                        request.top_logprobs is not None or request.logprob_token_ids
                    ):
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            logprob_token_ids=request.logprob_token_ids,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    # [CN] 这里是累积文本，真正的增量在下游 parser 或减法后体现。

                    delta_text = output.text

                    # [CN] 分块 prefill 的头几次迭代输出为空，此时不要给客户端发空块，
                    # [CN] 否则会出现大量无意义的 SSE 帧。

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    # [CN] parser 负责把自由文本切成 reasoning / content / tool_calls 三路增量。

                    if parser is not None:
                        delta_message = parser.parse_delta(
                            delta_text=delta_text,
                            delta_token_ids=as_list(output.token_ids),
                            request=request,
                            prompt_token_ids=res.prompt_token_ids,
                            finished=output.finish_reason is not None,
                        )
                        if delta_message is not None and delta_message.tool_calls:
                            tools_streamed[i] = True

                    # handle streaming just a content delta (no parsers)
                    # [CN] 无 parser 时全部当作普通内容直传。

                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # [CN] 注意加的是完整累积文本的差量部分，这里已经在上游被 parser 处理过。

                    previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)
                    if parser is not None:
                        generated_token_ids[i].extend(output.token_ids)
                        previous_reasoning_tokens[i] = parser.count_reasoning_tokens(
                            tuple(generated_token_ids[i])
                        )

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    # When reasoning is hidden, suppress per-token
                    # metadata (logprobs, token_ids) on every chunk to
                    # prevent leaking reasoning tokens through decoded
                    # token text in logprob entries or raw token IDs.
                    # [CN] 推理内容被隐藏时，必须连 logprobs 和 token_ids 一起屏蔽：
                    # [CN] 它们会把推理 token 的原始文本完整泄露出来，等于绕过隐藏。

                    hide_stream_metadata = (
                        not request.include_reasoning and parser is not None
                    )
                    if hide_stream_metadata:
                        logprobs = None

                    # [CN] parser 尚未攒够半个 token 时会返回 None，此时不应推送空块。

                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if output.finish_reason is None and (
                            not request.return_token_ids or hide_stream_metadata
                        ):
                            continue
                        # [CN] 空 delta 而非跳过：某些场景（如只要 token_ids）客户端依赖块的到达。

                        delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content_parts = []
                        if delta_message.content:
                            delta_content_parts.append(delta_message.content)
                        if delta_message.reasoning:
                            reasoning = delta_message.reasoning
                            delta_content_parts.append(f"[reasoning: {reasoning}]")
                        if delta_message.tool_calls:
                            tool_args = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )
                            if tool_args:
                                delta_content_parts.append(f"[tool_calls: {tool_args}]")

                        if delta_content_parts and self.enable_log_deltas:
                            delta_content = " ".join(delta_content_parts)
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    # [CN] token_ids 与否必须和 logprobs 用同一套隐藏判定，否则会漏数据。

                    include_token_ids = (
                        request.return_token_ids and not hide_stream_metadata
                    )

                    # [CN] 未终结时只发 delta，不带 stop_reason / finish_reason。

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids) if include_token_ids else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check for error finish reason and abort streaming
                        # finish_reason='error' indicates a retryable error
                        # [CN] finish_reason='error' 表示可重试的内部错误，要转成异常而不是当正常终止。

                        self._raise_if_error(output.finish_reason, request_id)

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        # [CN] OpenAI 语义：auto/required 触发的工具调用
                        # [CN] finish_reason 才是 tool_calls；具名调用是 stop。

                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids) if include_token_ids else None
                            ),
                        )

                        # [CN] 标记后在后续迭代中跳过该序列，保证每序列只发一次终结块。

                        finish_reason_sent[i] = True

                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    # Stamp the fingerprint on terminal chunks only (those with
                    # finish_reason set). When ``include_usage`` is on, the
                    # trailing usage chunk below overrides this as the true
                    # final message.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        # [CN] 没有末尾用量块时，终端块才是最后一块，此时才适合带指纹。

                        chunk.system_fingerprint = self.system_fingerprint

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                            completion_tokens_details=(
                                _make_completion_tokens_details(
                                    previous_reasoning_tokens[i]
                                )
                                if self._include_reasoning_tokens_details
                                else None
                            ),
                        )

                    # [CN] exclude_unset 很关键：未设置的字段不能序列化成 null，
                    # [CN] 否则客户端严格模式解析会失败。

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            # [CN] 末尾追加独立的 usage 块。注意 completion_tokens 是 n 份之和，
            # [CN] 而中间的 continuous usage 是单份计数 —— 两者语义不同。

            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                    completion_tokens_details=_make_completion_tokens_details(
                        sum(previous_reasoning_tokens)
                    )
                    if self._include_reasoning_tokens_details
                    else None,
                )
                final_usage.prompt_tokens_details = _make_prompt_tokens_details(
                    self.enable_prompt_tokens_details,
                    num_cached_tokens,
                    num_cache_creation_tokens,
                    mm_token_counts,
                )

                # In streaming, metrics ride on this final usage chunk, which is
                # only emitted when usage reporting is enabled (i.e.
                # ``stream_options.include_usage=true`` or
                # ``--enable-force-include-usage``).
                stream_per_request_metrics: PerRequestMetrics | None = None
                # See note in chat_completion_full_generator: suppress for n>1.
                # [CN] n>1 时各项耗时只属于其中一条序列，无法归属到整个请求，故直接抑制。

                if (request.n or 1) == 1:
                    if self.enable_per_request_metrics:
                        last_metrics = (
                            last_res.metrics if last_res is not None else None
                        )
                        stream_per_request_metrics = build_per_request_timing_metrics(
                            last_metrics, completion_tokens
                        )
                    spec_stats = build_spec_decoding_metrics(last_res)
                    if spec_stats is not None:
                        if stream_per_request_metrics is None:
                            stream_per_request_metrics = PerRequestMetrics()
                        stream_per_request_metrics.speculative_decoding = spec_stats

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                    metrics=stream_per_request_metrics,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            # [CN] 即使客户端没要 usage，也要回填到 request state 供中间件聚合统计。

            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
                completion_tokens_details=_make_completion_tokens_details(
                    sum(previous_reasoning_tokens)
                )
                if self._include_reasoning_tokens_details
                else None,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        # [CN] GenerationError 有明确的转换规则（含重试语义），单独分支处理。

        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        # [CN] 非 GenerationError 的异常统一转成内部错误块。

        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        # [CN] 无论成功失败都要发 [DONE]：SSE 客户端靠它判断流结束，
        # [CN] 不发会导致连接挂起直到超时。

        yield "data: [DONE]\n\n"

    # [CN] 非流式路径：先耗尽 generator 取最后一个 RequestOutput，再一次性构造响应。

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        parser: Parser | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        # [CN] 非流式：这个时间戳同样代表进入处理的时间点。

        created_time = int(time.time())
        # [CN] 只保留最后一个结果：RequestOutput 是累积快照，最后一个即完整结果。

        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        # [CN] 客户端断开会抛 CancelledError，这里转成错误响应而不是让异常穿透。

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        # [CN] 引擎一次都没输出属于内部错误，必须显式报错而不是返回空 choices。

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []
        # [CN] 跨 n 个序列累加的推理 token 总数，用于 usage 明细。

        total_reasoning_tokens = 0

        # [CN] 非流式同样需要判断 role，规则与流式首块完全一致。

        role = self.get_chat_request_role(request)
        # [CN] 是否真的能解析工具取决于 parser 实现，不能只看 enable_auto_tools。

        tool_parser_cls = (
            self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
        )
        # [CN] n 个序列各自生成一个 choice。

        for output in final_res.outputs:
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            # [CN] 非流式可以直接抛：响应还没发出，错误处理器能正常拦截。

            self._raise_if_error(output.finish_reason, request_id)
            # [CN] 完整 token 序列而非增量：parser 需要从头-parse 才能正确分离推理块。

            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and (
                request.top_logprobs is not None or request.logprob_token_ids
            ):
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    logprob_token_ids=request.logprob_token_ids,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            # [CN] parser.parse 一次性把完整输出切成三部分，与流式的增量解析是两套实现。

            if parser is not None:
                reasoning, content, tool_calls = parser.parse(
                    output.text,
                    request,
                    enable_auto_tools=self.enable_auto_tools,
                    model_output_token_ids=token_ids,
                )
                suppress_metadata = not request.include_reasoning and parser is not None
                if not request.include_reasoning:
                    reasoning = None
                total_reasoning_tokens += parser.count_reasoning_tokens(token_ids)
                if suppress_metadata:
                    logprobs = None
            else:
                reasoning = None
                content = output.text
                tool_calls = []
                # [CN] 无 parser 时不存在推理泄露风险，不需要屏蔽附件字段。

                suppress_metadata = False

            # [CN] 先置 False，只有真正走到 auto 分支且产出工具调用时才置 True。

            auto_tools_called = False
            # [CN] 用 type() 而非 isinstance 判断具名工具：联合类型里各成员没有继承关系，
            # [CN] isinstance 会把各种取值都算进来。

            is_named_tool_choice = (
                request.tool_choice is not None
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            )
            # [CN] required 与具名是两种不同的强制方式，但在 OpenAI 里 finish_reason 语义不同。

            is_required_tool_choice = request.tool_choice == "required"

            # All six construction sites route through ``self._create_chat_message``
            # so subclasses can swap in a specialized :class:`ChatMessage`
            # (e.g. the Cohere v2 handler's ``CohereChatMessage``) without
            # having to duplicate this branch logic.
            # [CN] 分支一：不具备工具能力且没指定具名/强制工具 -> 纯消息。

            if (not self.enable_auto_tools or not tool_parser_cls) and (
                not is_named_tool_choice and not is_required_tool_choice
            ):
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            # [CN] 分支二：明确点名或强制用工具 -> 必须携带 tool_calls。

            elif is_named_tool_choice or is_required_tool_choice:
                message = self._create_chat_message(
                    role=role,
                    reasoning=reasoning,
                    content=content or "",
                    tool_calls=[
                        ToolCall(id=tc.id or make_tool_call_id(), function=tc)
                        for tc in (tool_calls or [])
                    ],
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            # [CN] 分支三：显式禁用工具 -> 即使有 parser 也不展开。

            elif not request.tool_choice or request.tool_choice == "none":
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            # handle when there are tools and tool choice is auto
            # [CN] 分支四：auto 且同时具备工具与解析能力 -> 由输出内容决定是否有工具。

            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and tool_parser_cls
            ):
                # [CN] 记下来用于决定 finish_reason：auto 模式下真调用过才标 tool_calls。

                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    message = self._create_chat_message(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=[
                            ToolCall(id=tc.id or make_tool_call_id(), function=tc)
                            for tc in tool_calls
                        ],
                    )

                else:
                    message = self._create_chat_message(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            # Subclass hook: enrich the constructed message with any
            # parser-side extras that don't fit through the plain
            # ``(reasoning, content, tool_calls)`` tuple. Base is a no-op;
            # citation-aware handlers use this to surface grounding
            # metadata cached on the reasoning parser.
            # [CN] 最后统一过一遍子类 hook，保证所有分支产出的 message 都被同等处理。

            message = self._finalize_response_message(message, parser=parser)

            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            # [CN] OpenAI 只在 auto/required 且成功调用时给 tool_calls，
            # [CN] 具名调用即使命中工具也仍是 stop。

            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )

            # [CN] MoE 路由信息以 base64 透出，用于调试专家负载；体积大故只在需要时才带。

            routed_experts_b64 = (
                numpy2base64(output.routed_experts)
                if output.routed_experts is not None
                else None
            )

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids)
                    if request.return_token_ids and not suppress_metadata
                    else None
                ),
                routed_experts=routed_experts_b64,
            )
            # [CN] parallel_tool_calls=false 时裁剪到第一个工具调用，兼容 OpenAI 语义。

            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        # [CN] echo 要把最后一条用户输入拼回 content 前面，且只对同角色的 content 生效。

        # [CN] 非流式 echo 要把原始输入拼到 content 前面；
        # [CN] 多模态 content 是分段列表，需要先 join 成文本才能拼接。

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        # [CN] 非流式必然拿到完整 prompt_token_ids，据此计算 usage.prompt_tokens。

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        # [CN] n 个序列的生成 token 都要计入 —— 与 prompt token 只算一次不同。

        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            completion_tokens_details=_make_completion_tokens_details(
                total_reasoning_tokens
            )
            if self._include_reasoning_tokens_details
            else None,
        )
        # [CN] 明细走二次赋值而不是构造时传入：它依赖开关和多项状态，单独判断更清楚。

        usage.prompt_tokens_details = _make_prompt_tokens_details(
            self.enable_prompt_tokens_details,
            final_res.num_cached_tokens,
            final_res.num_cache_creation_tokens,
            mm_token_counts,
        )

        # [CN] 回填给中间件用于全局配额与统计，与返回给客户端的 usage 是同一份数据。

        request_metadata.final_usage_info = usage

        per_request_metrics: PerRequestMetrics | None = None
        # Per-request metrics (timing + spec-decode acceptance) describe a single
        # generation stream. For n>1 the stats belong to only one of the n
        # sequences, so they cannot be attributed to the request; suppress.
        # [CN] 与流式同样的规则：n>1 时 per-request 指标无法归属到请求，抑制输出。

        if (request.n or 1) == 1:
            if self.enable_per_request_metrics:
                per_request_metrics = build_per_request_timing_metrics(
                    final_res.metrics, num_generated_tokens
                )
            spec_stats = build_spec_decoding_metrics(final_res)
            if spec_stats is not None:
                if per_request_metrics is None:
                    per_request_metrics = PerRequestMetrics()
                per_request_metrics.speculative_decoding = spec_stats

        # ``final_res.prompt`` is the rendered chat-templated prompt text
        # [CN] 渲染后的完整 prompt 文本随响应返回，便于调试模板是否写对。

        prompt_text = final_res.prompt if request.return_prompt_text else None

        # [CN] prompt_logprobs 必须 clamp：原始值可能超出客户端能表示的范围。

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            prompt_text=prompt_text,
            kv_transfer_params=final_res.kv_transfer_params,
            ec_transfer_params=final_res.ec_transfer_params,
            metrics=per_request_metrics,
        )

        # Log complete response if output logging is enabled
        # [CN] 完整输出日志等到响应全部构造完成后再写，避免记录解析中途状态。

        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:  # type: ignore
                        function_call: FunctionCall = tc.function  # type: ignore
                        tool_call_descriptions.append(
                            f"{function_call.name}({function_call.arguments})"
                        )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        # [CN] 非流式返回完整响应对象，由上层 FastAPI 序列化成 JSON。

        return response

    # [CN] 把引擎侧的 top-k Logprob 转成 OpenAI 的列表结构。

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
        return_all: bool = False,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                # [CN] -9999.0 是 OpenAI 约定的 -inf 替身：-inf 无法进 JSON，
                # [CN] 而不夹.perm터会用 -inf 让客户端 JSON 解析失败。

                logprob=max(p[1].logprob, -9999.0),
                # [CN] errors="replace"：罕见 token 解码出的不是合法 UTF-8，
                # [CN] 严格模式会直接抛异常导致整个响应失败。

                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            # [CN] return_all 由 logprob_token_ids 驱动：显式指定 token 列表时全部返回。

            if return_all
            or top_logprobs == -1
            or (top_logprobs is not None and i < top_logprobs)
        ]

    # [CN] 逐 token 构造 OpenAI 的 logprobs.content。

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        logprob_token_ids: list[int] | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        # [CN] 请求级覆盖全局配置：单个请求可以要求用 token id 占位符而非解码文本。

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            # [CN] 采样到的 token 恰好不在 top-k 里时的退化分支：补一个没有 logprob 的条目。
            # [CN] 必须补而不能跳过，否则 content 的长度会和 token 数对不上。

            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = format_token_id_placeholder(token_id)
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                # [CN] 引擎侧可能已经缓存过解码结果，优先复用以避免重复解码。

                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                            # [CN] 指定了 logprob_token_ids 就返回全词表而非 top-k —— 由调用方自行筛选。

                            return_all=bool(logprob_token_ids),
                        ),
                    )
                )

        # [CN] content 的长度严格等于输出 token 数，缺失位置也要补占位条目。

        return ChatCompletionLogProbs(content=logprobs_content)
