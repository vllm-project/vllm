# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.generate.base.serving import resolve_token_id_placeholder
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderStreamState,
    GenerateResponse,
    GenerateStreamResponse,
)
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)
from vllm.logger import init_logger
from vllm.parser import Parser, ParserManager
from vllm.renderers import BaseRenderer
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.utils import random_uuid
from vllm.utils.async_utils import make_async

logger = init_logger(__name__)


class OnlineDerenderer:
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        log_error_stack: bool = False,
    ) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.request_logger = request_logger

        self.enable_auto_tools = enable_auto_tools
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        self.parser: type[Parser] | None = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=model_config.model,
            is_harmony=self.use_harmony,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: ChatTemplateContentFormatOption = (
            chat_template_content_format
        )
        self.default_chat_template_kwargs: dict[str, Any] = (
            default_chat_template_kwargs or {}
        )
        self.trust_request_chat_template = trust_request_chat_template

        self.log_error_stack = log_error_stack
        self.supports_browsing = False
        self.supports_code_interpreter = False

        # Detokenization, logprob resolution and parsing are CPU-bound;
        # offload them in one hop to keep the event loop responsive.
        self._derender_chat_async = make_async(
            self._derender_chat, executor=renderer._executor
        )
        self._derender_completion_async = make_async(
            self._derender_completion, executor=renderer._executor
        )
        # Replay is O(n) per chunk (unlike the O(delta) detok-only paths),
        # so it must not run on the event loop either.
        self._derender_chat_stream_parsed_async = make_async(
            self._derender_chat_stream_parsed, executor=renderer._executor
        )

    async def derender_chat(
        self,
        generate_response: GenerateResponse,
        chat_request: ChatCompletionRequest | None = None,
    ) -> list[ChatCompletionResponseChoice]:
        return await self._derender_chat_async(generate_response, chat_request)

    def _derender_chat(
        self,
        generate_response: GenerateResponse,
        chat_request: ChatCompletionRequest | None = None,
    ) -> list[ChatCompletionResponseChoice]:
        tokenizer = self.renderer.get_tokenizer()
        choices: list[ChatCompletionResponseChoice] = []

        for choice in generate_response.choices:
            if not choice.token_ids:
                raise ValueError(f"choice {choice.index} has empty or null token_ids")

            resolved_logprobs = (
                _resolve_logprobs(choice.logprobs, tokenizer)
                if choice.logprobs is not None
                else None
            )

            if self.parser is not None and chat_request is not None:
                # Parser path: decode with special tokens preserved
                # so the parser can see markers like </think>,
                # <tool_call>, or Harmony channel tokens.
                decoded_text = tokenizer.decode(
                    choice.token_ids, skip_special_tokens=False
                )

                chat_template_kwargs: dict[str, Any] = {}
                if not self.use_harmony:
                    chat_template_kwargs = (
                        chat_request.build_chat_params(
                            self.chat_template,
                            self.chat_template_content_format,
                        )
                        .with_defaults(self.default_chat_template_kwargs)
                        .chat_template_kwargs
                    )

                parser = self.parser(
                    tokenizer,
                    chat_request.tools,
                    chat_template_kwargs=chat_template_kwargs,
                    model_config=self.model_config,
                )
                reasoning, content, tool_calls = parser.parse(
                    decoded_text,
                    chat_request,
                    enable_auto_tools=self.enable_auto_tools,
                    model_output_token_ids=choice.token_ids,
                )

                if not getattr(chat_request, "include_reasoning", True):
                    reasoning = None

                tc_items = (
                    [
                        ToolCall(
                            id=random_uuid(),
                            function=tc,
                        )
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else []
                )

                is_named_tool_choice = (
                    type(chat_request.tool_choice) is ChatCompletionNamedToolChoiceParam
                )
                is_required_tool_choice = chat_request.tool_choice == "required"
                if is_named_tool_choice or is_required_tool_choice:
                    content = content or ""

                message = ChatMessage(
                    role="assistant",
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tc_items,
                )
            else:
                # No parser: plain detokenization honouring the request's
                # skip_special_tokens (default True when no request was given).
                skip_special = (
                    chat_request.skip_special_tokens
                    if chat_request is not None
                    else True
                )
                decoded_text = tokenizer.decode(
                    choice.token_ids, skip_special_tokens=skip_special
                )
                message = ChatMessage(role="assistant", content=decoded_text)

            choices.append(
                ChatCompletionResponseChoice(
                    index=choice.index,
                    message=message,
                    logprobs=resolved_logprobs,
                    finish_reason=choice.finish_reason,
                )
            )

        return choices

    def _detokenize_delta(
        self,
        tokenizer: TokenizerLike,
        delta_token_ids: list[int],
        state: DerenderStreamState,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> tuple[str, DerenderStreamState]:
        """Incrementally detokenize ``delta_token_ids`` from prior stream state.

        Resumes decoding from the offsets carried in ``state`` rather than
        replaying token history. ``state.prev_tokens`` holds the trailing decode
        window (from ``prefix_offset`` onward) that ``detokenize_incrementally``
        still needs to reproduce any partially read multi-byte character
        (tracked by ``read_offset``). The delta tokens are fed straight onto it.

        The window is bounded. ``detokenize_incrementally`` never reads before
        ``prefix_offset``, so after processing we trim ``prev_tokens`` to that
        tail and rebase the offsets to it. State transport therefore stays
        O(window) per chunk instead of re-sending the full token history.

        Args:
            tokenizer: The tokenizer to decode with.
            delta_token_ids: New token IDs from this generate chunk.
            state: Client carried detok state from the previous call.
            skip_special_tokens: Passed through to the tokenizer.
            spaces_between_special_tokens: Passed through to the tokenizer.

        Returns:
            (new_text, updated_state) — the delta text for this chunk and the
            state to pass to the next call.
        """
        prev_tokens = list(state.prev_tokens)
        prefix_offset = state.prefix_offset
        read_offset = state.read_offset

        text_parts: list[str] = []
        for tok_id in delta_token_ids:
            # prev_tokens is a (possibly empty) list, never None, so this
            # always takes the non first iter path and only consumes
            # all_input_ids[-1].
            new_toks, text, prefix_offset, read_offset = detokenize_incrementally(
                tokenizer=tokenizer,
                all_input_ids=[tok_id],
                prev_tokens=prev_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens,
            )
            prev_tokens = prev_tokens + new_toks
            text_parts.append(text)

        # Trim to the tail still readable by detokenize_incrementally
        # (everything before prefix_offset is dead) and rebase the offsets so
        # the carried window stays bounded regardless of generation length.
        trimmed = prev_tokens[prefix_offset:]
        updated_state = state.model_copy(
            update={
                "prev_tokens": trimmed,
                "prefix_offset": 0,
                "read_offset": read_offset - prefix_offset,
            }
        )
        return "".join(text_parts), updated_state

    async def derender_chat_stream(
        self,
        model: str,
        generate_chunk: GenerateStreamResponse,
        state: DerenderStreamState | None = None,
        chat_request: ChatCompletionRequest | None = None,
        prompt_tokens: int | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> tuple[ChatCompletionStreamResponse, DerenderStreamState]:
        """Process one GenerateStreamResponse chunk for streaming chat derender.

        Unlike OpenAI's API, which always emits `role: "assistant"` on the
        very first chunk, this emits it on the first chunk with a non empty
        `choices` list. A leading usage only chunk therefore defers the
        role to the following content chunk instead of sending an empty
        role only delta.

        Args:
            model: Model name for the response object.
            generate_chunk: One SSE chunk from `/inference/v1/generate`.
            state: Client carried detok state (`None` for first call).
            chat_request: Original ChatCompletionRequest from `/render`.
                Required when a reasoning or tool parser is configured
                (validated by the caller — see `ServingDerender`) because
                plain detokenization would leak raw parser markup into `content`.
            prompt_tokens: Prompt token count for the usage chunk.
            prompt_token_ids: Prompt token IDs. Parser path onlyThe IDs lets
                `parse_delta` settle its initial reasoning state. Required
                when a reasoning or tool parser is configured (validated by
                the caller — see ``ServingDerender``). Without it reasoning
                left open by the prompt would be misclassified as content.

        Returns:
            (chunk, updated_state) — the derendered SSE chunk and the state
            the client must pass to the next call.
        """
        if state is None:
            state = DerenderStreamState()

        # A single DerenderStreamState is threaded through every choice in
        # this chunk. Correct only when there is at most one choice per SSE
        # event (n=1, one call per index), as the streaming derender
        # protocol assumes. Multiple choices sharing one chunk would corrupt
        # each other's detok/parser state.
        if len(generate_chunk.choices) > 1:
            raise ValueError(
                "derender_chat_stream expects at most one choice per chunk"
            )

        parser_cls = self.parser
        if parser_cls is not None:
            # Fail (mirrors ServingDerender's pre-check) because a parser
            # configured model must never fall through to plain detok or
            # reasoning/tool markup would leak into `delta.content`.
            if chat_request is None:
                raise ValueError(
                    "chat_request is required for streaming chat derender "
                    "when a tool or reasoning parser is configured"
                )
            return await self._derender_chat_stream_parsed_async(
                parser_cls,
                model,
                generate_chunk,
                state,
                chat_request,
                prompt_tokens,
                prompt_token_ids,
            )

        tokenizer = self.renderer.get_tokenizer()
        skip_special = (
            chat_request.skip_special_tokens if chat_request is not None else True
        )
        stream_choices: list[ChatCompletionResponseStreamChoice] = []
        updated_state = state

        for choice in generate_chunk.choices:
            delta_tids = choice.token_ids or []
            new_text, updated_state = self._detokenize_delta(
                tokenizer, delta_tids, updated_state, skip_special_tokens=skip_special
            )

            include_role = not updated_state.role_sent
            if include_role:
                updated_state = updated_state.model_copy(update={"role_sent": True})

            delta = DeltaMessage(
                role="assistant" if include_role else None,
                content=new_text if new_text else None,
            )
            stream_choices.append(
                ChatCompletionResponseStreamChoice(
                    index=choice.index,
                    delta=delta,
                    finish_reason=choice.finish_reason,
                )
            )

        usage: UsageInfo | None = None
        if generate_chunk.usage is not None:
            u = generate_chunk.usage
            pt = prompt_tokens if prompt_tokens is not None else (u.prompt_tokens or 0)
            ct = u.completion_tokens or 0
            usage = UsageInfo(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
            )

        chunk = ChatCompletionStreamResponse(
            id=generate_chunk.request_id,
            model=model,
            choices=stream_choices,
            usage=usage,
        )
        return chunk, updated_state

    def _derender_chat_stream_parsed(
        self,
        parser_cls: type[Parser],
        model: str,
        generate_chunk: GenerateStreamResponse,
        state: DerenderStreamState,
        chat_request: ChatCompletionRequest,
        prompt_tokens: int | None,
        prompt_token_ids: list[int] | None,
    ) -> tuple[ChatCompletionStreamResponse, DerenderStreamState]:
        """Parser path for streaming chat derender: replay + `parse_delta`.

        Parser internal state (buffered markup, reasoning/tool phase, etc.)
        cannot be serialized into `DerenderStreamState`, so each call
        builds a fresh parser and replays every prior output token through
        `parse_delta` (discarding the result) before processing this
        chunk's tokens for real. This makes the emission for chunk *k*
        exactly what chunk *k+1*'s replay would reconstruct. The output is
        therefore independent of how the client chunks the generated stream.

        Tokens are fed one at a time (uniform per-token granularity) through
        a fresh incremental detokenizer with special tokens preserved
        (``skip_special_tokens=False``), so the parser sees markers like
        ``</think>`` or ``<tool_call>`` exactly as the generate streaming
        path does.
        """
        tokenizer = self.renderer.get_tokenizer()

        chat_template_kwargs: dict[str, Any] = {}
        if not self.use_harmony:
            chat_template_kwargs = (
                chat_request.build_chat_params(
                    self.chat_template,
                    self.chat_template_content_format,
                )
                .with_defaults(self.default_chat_template_kwargs)
                .chat_template_kwargs
            )

        parser = parser_cls(
            tokenizer,
            chat_request.tools,
            chat_template_kwargs=chat_template_kwargs,
            model_config=self.model_config,
        )

        # Ephemeral incremental detok window, local to this call. Threaded
        # across both the replay and current chunk phases (via `nonlocal`)
        # so multi-byte characters split across that boundary still decode
        # correctly. Discarded once the call returns.
        detok_state = DerenderStreamState()

        def _feed(token_ids: list[int], finished: bool) -> DeltaMessage | None:
            nonlocal detok_state
            acc: DeltaMessage | None = None
            last = len(token_ids) - 1
            for i, tok_id in enumerate(token_ids):
                text, detok_state = self._detokenize_delta(
                    tokenizer, [tok_id], detok_state, skip_special_tokens=False
                )
                delta = parser.parse_delta(
                    text,
                    [tok_id],
                    chat_request,
                    prompt_token_ids=prompt_token_ids,
                    finished=finished and i == last,
                )
                acc = _merge_delta_messages(acc, delta)
            return acc

        # Replay history to reconstruct parser state. The result is thrown
        # away and only the current chunk's emission goes to the client.
        _feed(state.output_token_ids, finished=False)

        stream_choices: list[ChatCompletionResponseStreamChoice] = []
        output_token_ids = list(state.output_token_ids)
        role_sent = state.role_sent
        tools_streamed = state.tools_streamed
        last_tool_call_ids = list(state.last_tool_call_ids)

        # At most one choice: the caller (derender_chat_stream) already
        # rejects >1 before dispatching here. role_sent/tools_streamed/
        # output_token_ids below are updated for that single choice, not
        # accumulated across choices. Looping over generate_chunk.choices
        # could silently corrupt output if chunks were ever allowed to
        # contain multiple choices.
        if generate_chunk.choices:
            choice = generate_chunk.choices[0]
            delta_tids = choice.token_ids or []
            is_finished = choice.finish_reason is not None

            if delta_tids:
                delta_message = _feed(delta_tids, finished=is_finished)
            elif is_finished:
                # Finish only chunk (no new tokens). Still flush any
                # buffered tool call arguments.
                delta_message = parser.parse_delta(
                    "",
                    [],
                    chat_request,
                    prompt_token_ids=prompt_token_ids,
                    finished=True,
                )
            else:
                delta_message = None

            output_token_ids.extend(delta_tids)

            if delta_message is None:
                delta_message = DeltaMessage()

            if delta_message.tool_calls:
                tools_streamed = True
                for tc in delta_message.tool_calls:
                    if tc.id is None:
                        continue
                    if tc.index < len(last_tool_call_ids):
                        # Pin: reuse the ID already recorded for this index
                        # rather than one a from scratch replay regenerated.
                        # Real trigger not just defensive with
                        # tool_choice="required",
                        # extract_required_tool_call_streaming resets
                        # function_name_returned to False whenever the
                        # partial JSON transiently fails to parse which
                        # re-emits id+name for the same index on replay.
                        tc.id = last_tool_call_ids[tc.index]
                    else:
                        last_tool_call_ids.append(tc.id)

            if not role_sent:
                delta_message.role = "assistant"
                role_sent = True

            finish_reason = choice.finish_reason
            if finish_reason is not None:
                is_named_tool_choice = (
                    type(chat_request.tool_choice) is ChatCompletionNamedToolChoiceParam
                )
                if tools_streamed and not is_named_tool_choice:
                    finish_reason = "tool_calls"

            stream_choice = ChatCompletionResponseStreamChoice(
                index=choice.index,
                delta=delta_message,
                finish_reason=finish_reason,
            )
            stream_choices.append(
                maybe_filter_parallel_tool_calls(stream_choice, chat_request)
            )

        updated_state = state.model_copy(
            update={
                "output_token_ids": output_token_ids,
                "role_sent": role_sent,
                "tools_streamed": tools_streamed,
                "last_tool_call_ids": last_tool_call_ids,
            }
        )

        usage: UsageInfo | None = None
        if generate_chunk.usage is not None:
            u = generate_chunk.usage
            pt = prompt_tokens if prompt_tokens is not None else (u.prompt_tokens or 0)
            ct = u.completion_tokens or 0
            usage = UsageInfo(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
            )

        chunk = ChatCompletionStreamResponse(
            id=generate_chunk.request_id,
            model=model,
            choices=stream_choices,
            usage=usage,
        )
        return chunk, updated_state

    async def derender_completion(
        self,
        generate_responses: list[GenerateResponse],
        prompt_tokens: list[int] | None = None,
        completion_request: CompletionRequest | None = None,
    ) -> tuple[list[CompletionResponseChoice], int, int]:
        return await self._derender_completion_async(
            generate_responses, prompt_tokens, completion_request
        )

    def _derender_completion(
        self,
        generate_responses: list[GenerateResponse],
        prompt_tokens: list[int] | None = None,
        completion_request: CompletionRequest | None = None,
    ) -> tuple[list[CompletionResponseChoice], int, int]:
        n = len(generate_responses)
        prompt_tokens_list: list[int] = (
            prompt_tokens if prompt_tokens is not None else [0] * n
        )

        skip_special = (
            completion_request.skip_special_tokens
            if completion_request is not None
            else True
        )
        tokenizer = self.renderer.get_tokenizer()
        choices: list[CompletionResponseChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        index = 0

        for gen, pt in zip(generate_responses, prompt_tokens_list):
            for choice in gen.choices:
                if not choice.token_ids:
                    raise ValueError(
                        f"choice {choice.index} in response {gen.request_id} "
                        "has empty or null token_ids"
                    )

                decoded_text = tokenizer.decode(
                    choice.token_ids, skip_special_tokens=skip_special
                )
                completion_logprobs = None
                if choice.logprobs is not None:
                    resolved = _resolve_logprobs(choice.logprobs, tokenizer)
                    completion_logprobs = _convert_chat_logprobs_to_completion_logprobs(
                        resolved
                    )
                choices.append(
                    CompletionResponseChoice(
                        index=index,
                        text=decoded_text,
                        finish_reason=choice.finish_reason,
                        logprobs=completion_logprobs,
                    )
                )
                total_completion_tokens += len(choice.token_ids)
                index += 1
            total_prompt_tokens += pt

        return choices, total_prompt_tokens, total_completion_tokens

    async def derender_completion_stream(
        self,
        model: str,
        generate_chunk: GenerateStreamResponse,
        state: DerenderStreamState | None = None,
        prompt_tokens: int | None = None,
        completion_request: CompletionRequest | None = None,
    ) -> tuple[CompletionStreamResponse, DerenderStreamState]:
        """Process one GenerateStreamResponse chunk for streaming completions.

        Each call takes one SSE chunk from ``/inference/v1/generate`` plus the
        client carried ``stream_state`` and returns a ``CompletionStreamResponse``
        chunk and the updated state.

        The generate stream emits one choice per SSE event, so this method
        processes one output sequence at a time.  For ``n > 1`` the client
        maintains one ``DerenderStreamState`` per ``choice.index``.

        Args:
            model: Model name for the response object.
            generate_chunk: One SSE chunk from ``/inference/v1/generate``.
            state: Client carried detok state (``None`` → first call).
            prompt_tokens: Prompt token count for usage (from the render step).
            completion_request: Original CompletionRequest from ``/render``;
                supplies ``skip_special_tokens``.

        Returns:
            (chunk, updated_state) — the derendered chunk and updated state.
        """
        if state is None:
            state = DerenderStreamState()

        # See the equivalent check in derender_chat_stream: a single
        # DerenderStreamState is threaded through every choice in this
        # chunk, so more than one choice per chunk would corrupt the
        # detok window across choices.
        if len(generate_chunk.choices) > 1:
            raise ValueError(
                "derender_completion_stream expects at most one choice per chunk"
            )

        tokenizer = self.renderer.get_tokenizer()
        skip_special = (
            completion_request.skip_special_tokens
            if completion_request is not None
            else True
        )
        stream_choices: list[CompletionResponseStreamChoice] = []
        updated_state = state

        for choice in generate_chunk.choices:
            delta_tids = choice.token_ids or []
            new_text, updated_state = self._detokenize_delta(
                tokenizer, delta_tids, updated_state, skip_special_tokens=skip_special
            )
            stream_choices.append(
                CompletionResponseStreamChoice(
                    index=choice.index,
                    text=new_text,
                    finish_reason=choice.finish_reason,
                )
            )

        usage: UsageInfo | None = None
        if generate_chunk.usage is not None:
            u = generate_chunk.usage
            pt = prompt_tokens if prompt_tokens is not None else (u.prompt_tokens or 0)
            ct = u.completion_tokens or 0
            usage = UsageInfo(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
            )

        chunk = CompletionStreamResponse(
            id=generate_chunk.request_id,
            model=model,
            choices=stream_choices,
            usage=usage,
        )
        return chunk, updated_state


def _merge_delta_messages(
    acc: DeltaMessage | None, new: DeltaMessage | None
) -> DeltaMessage | None:
    """Merge one per-token `parse_delta` result into a per chunk accumulator.

    `content`/`reasoning` concatenate. For `tool_calls`, an incoming
    delta that carries an `id` or a function `name` starts a new entry.
    Otherwise its `function.arguments` are concatenated onto the existing
    entry with the matching `index`.
    """
    if new is None:
        return acc
    if acc is None:
        acc = DeltaMessage()

    if new.content:
        acc.content = (acc.content or "") + new.content
    if new.reasoning:
        acc.reasoning = (acc.reasoning or "") + new.reasoning

    for tc in new.tool_calls:
        starts_new_call = tc.id is not None or (
            tc.function is not None and tc.function.name is not None
        )
        existing = (
            None
            if starts_new_call
            else next((t for t in acc.tool_calls if t.index == tc.index), None)
        )
        if existing is None:
            acc.tool_calls.append(tc)
            continue
        if existing.function is None:
            existing.function = DeltaFunctionCall()
        if tc.function is not None and tc.function.arguments:
            existing.function.arguments = (
                existing.function.arguments or ""
            ) + tc.function.arguments

    return acc


def _parse_token_id_placeholder(token: str) -> int | None:
    """Extract token ID from a 'token_id:N' placeholder string."""
    if not token.startswith("token_id:"):
        return None
    try:
        return int(token[len("token_id:") :])
    except ValueError:
        return None


def _correct_decoded_token(
    token_id: int, context_token_ids: list[int], tokenizer: TokenizerLike
) -> str:
    """Use preceding tokens as context to fix U+FFFD from byte-fallback.

    Mirrors LogprobsProcessor._correct_decoded_token in v1/engine/logprobs.py.
    """
    max_ctx = min(len(context_token_ids), 4)

    for num_ctx in range(1, max_ctx + 1):
        context = context_token_ids[-num_ctx:]
        full_decoded = tokenizer.decode(context + [token_id])

        if full_decoded.endswith("�"):
            continue

        clean_end = len(context)
        for j in range(len(context) - 1, -1, -1):
            if tokenizer.decode([context[j]]).endswith("�"):
                clean_end = j
            else:
                break

        clean_prefix = tokenizer.decode(context[:clean_end]) if clean_end > 0 else ""

        if full_decoded.startswith(clean_prefix):
            return full_decoded[len(clean_prefix) :]

        common_len = 0
        for a, b in zip(clean_prefix, full_decoded):
            if a != b:
                break
            common_len += 1
        return full_decoded[common_len:]

    return ""


def _resolve_logprobs(
    logprobs: ChatCompletionLogProbs, tokenizer: TokenizerLike
) -> ChatCompletionLogProbs:
    """Resolve token_id:N placeholders in a ChatCompletionLogProbs object."""
    if logprobs.content is None:
        return logprobs

    context_token_ids: list[int] = []
    resolved_content = []

    for entry in logprobs.content:
        token_str, token_bytes = resolve_token_id_placeholder(entry.token, tokenizer)
        sampled_id = _parse_token_id_placeholder(entry.token)

        if token_str.endswith("�") and sampled_id is not None:
            token_str = _correct_decoded_token(sampled_id, context_token_ids, tokenizer)
            token_bytes = list(token_str.encode("utf-8"))

        resolved_top = []
        for top in entry.top_logprobs:
            top_str, top_bytes = resolve_token_id_placeholder(top.token, tokenizer)
            top_id = _parse_token_id_placeholder(top.token)
            if top_str.endswith("�") and top_id is not None:
                top_str = _correct_decoded_token(top_id, context_token_ids, tokenizer)
                top_bytes = list(top_str.encode("utf-8"))
            resolved_top.append(
                top.model_copy(update={"token": top_str, "bytes": top_bytes})
            )

        resolved_content.append(
            entry.model_copy(
                update={
                    "token": token_str,
                    "bytes": token_bytes,
                    "top_logprobs": resolved_top,
                }
            )
        )

        if sampled_id is not None:
            context_token_ids.append(sampled_id)

    return ChatCompletionLogProbs(content=resolved_content)


def _convert_chat_logprobs_to_completion_logprobs(
    logprobs: ChatCompletionLogProbs,
) -> CompletionLogProbs:
    """Convert ChatCompletionLogProbs (per-token objects) to CompletionLogProbs
    (parallel flat lists) as required by the /v1/completions response schema."""
    if logprobs.content is None:
        return CompletionLogProbs()

    tokens: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs_list: list[dict[str, float] | None] = []
    text_offset: list[int] = []

    offset = 0
    for entry in logprobs.content:
        text_offset.append(offset)
        tokens.append(entry.token)
        token_logprobs.append(entry.logprob)
        top_logprobs_list.append(
            {t.token: t.logprob for t in entry.top_logprobs}
            if entry.top_logprobs
            else None
        )
        offset += len(entry.token)

    return CompletionLogProbs(
        text_offset=text_offset,
        token_logprobs=token_logprobs,
        tokens=tokens,
        top_logprobs=top_logprobs_list,
    )
