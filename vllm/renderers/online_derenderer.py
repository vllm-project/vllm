# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any

from vllm.config import ModelConfig
from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ToolCall,
)
from vllm.entrypoints.generate.base.serving import decode_token_ids
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
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
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderStreamState,
    GenerateLogProbs,
    GenerateTokensResponse,
    GenerateTokensStreamResponse,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)
from vllm.logger import init_logger
from vllm.parser import Parser, ParserManager
from vllm.renderers import BaseRenderer
from vllm.renderers.chat_utils import ChatTemplateContentFormatOption
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
    get_leading_space_marker,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import make_async
from vllm.v1.engine.detokenizer import uses_fast_detokenizer

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
        tool_strict_level: str = "auto",
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
            tool_strict_level=tool_strict_level,
            model_name=model_config.model,
            is_harmony=self.use_harmony,
            tokenizer=renderer.tokenizer,
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
        self._detokenize_delta_async = make_async(
            self._detokenize_delta, executor=renderer._executor
        )
        # Replay is O(n) per chunk, so it must not run on the event loop.
        self._derender_chat_stream_parsed_async = make_async(
            self._derender_chat_stream_parsed, executor=renderer._executor
        )

    async def derender_chat(
        self,
        generate_response: GenerateTokensResponse,
        chat_request: ChatCompletionRequest | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> list[ChatCompletionResponseChoice]:
        return await self._derender_chat_async(
            generate_response, chat_request, prompt_token_ids
        )

    def _derender_chat(
        self,
        generate_response: GenerateTokensResponse,
        chat_request: ChatCompletionRequest | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> list[ChatCompletionResponseChoice]:
        tokenizer = self.renderer.get_tokenizer()
        choices: list[ChatCompletionResponseChoice] = []

        has_parser = self.parser is not None and chat_request is not None
        skip_special, spaces_between = _decode_params(
            tokenizer, chat_request, preserve_special=has_parser
        )
        seed_ids = (
            prompt_token_ids
            if prompt_token_ids is not None
            else generate_response.prompt_token_ids
        )
        # Choices continue the same prompt, so they share one seeded window.
        seed_state = _seed_stream_state(
            tokenizer,
            seed_ids,
            skip_special_tokens=skip_special,
        )

        chat_template_kwargs = (
            self._resolve_chat_template_kwargs(chat_request)
            if has_parser and chat_request is not None
            else {}
        )

        for choice in generate_response.choices:
            if not choice.token_ids:
                raise ValueError(f"choice {choice.index} has empty or null token_ids")

            resolved_logprobs = (
                _resolve_logprobs(choice.logprobs, tokenizer)
                if choice.logprobs is not None
                else None
            )

            # With a parser, special tokens are preserved so it can see
            # markers like </think>, <tool_call>, or Harmony channel tokens.
            # Without one, the request's skip_special_tokens is honoured
            # (default True when no request was given).
            decoded_text, _ = self._detokenize_delta(
                tokenizer,
                choice.token_ids,
                seed_state,
                skip_special_tokens=skip_special,
                spaces_between_special_tokens=spaces_between,
            )

            if has_parser:
                assert chat_request is not None
                message = self._parse_and_assemble(
                    tokenizer,
                    decoded_text,
                    choice.token_ids,
                    chat_request,
                    chat_template_kwargs,
                    generate_response.prompt_token_ids,
                )
            else:
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

    def _resolve_chat_template_kwargs(
        self, chat_request: ChatCompletionRequest
    ) -> dict[str, Any]:
        if self.use_harmony:
            return {}
        return (
            chat_request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    def _parse_and_assemble(
        self,
        tokenizer: TokenizerLike,
        text: str,
        token_ids: Sequence[int],
        chat_request: ChatCompletionRequest,
        chat_template_kwargs: dict[str, Any],
        prompt_token_ids: list[int] | None,
    ) -> ChatMessage:
        """Parse decoded output text into an assistant `ChatMessage`.

        Args:
            tokenizer: Tokenizer handed to the parser.
            text: Decoded output text to parse.
            token_ids: Output token IDs the text was decoded from.
            chat_request: Request supplying tools, `tool_choice` and
                `include_reasoning`.
            chat_template_kwargs: Already resolved template kwargs.
            prompt_token_ids: Prompt token IDs, if known.

        Returns:
            The assembled assistant message.

        """
        assert self.parser is not None
        parser = self.parser(
            tokenizer,
            chat_request.tools,
            chat_template_kwargs=chat_template_kwargs,
            model_config=self.model_config,
        )
        if prompt_token_ids is not None:
            parser.set_prompt_token_ids(prompt_token_ids)
        reasoning, content, tool_calls = parser.parse(
            text,
            chat_request,
            enable_auto_tools=self.enable_auto_tools,
            model_output_token_ids=token_ids,
        )

        if not getattr(chat_request, "include_reasoning", True):
            reasoning = None

        tc_items = (
            [ToolCall(id=random_uuid(), function=tc) for tc in tool_calls]
            if tool_calls
            else []
        )

        is_named_tool_choice = (
            type(chat_request.tool_choice) is ChatCompletionNamedToolChoiceParam
        )
        is_required_tool_choice = chat_request.tool_choice == "required"
        if is_named_tool_choice or is_required_tool_choice:
            content = content or ""

        return ChatMessage(
            role="assistant",
            reasoning=reasoning,
            content=content,
            tool_calls=tc_items,
        )

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
        ``prefix_offset``, so after each token we trim ``prev_tokens`` to that
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
            # Trim to the tail still readable by detokenize_incrementally
            # (everything before prefix_offset is dead) and rebase the
            # offsets, so the window stays bounded within a long batch
            # decode as well as across chunks.
            prev_tokens = (prev_tokens + new_toks)[prefix_offset:]
            read_offset -= prefix_offset
            prefix_offset = 0
            text_parts.append(text)

        updated_state = state.model_copy(
            update={
                "prev_tokens": prev_tokens,
                "prefix_offset": prefix_offset,
                "read_offset": read_offset,
            }
        )
        return "".join(text_parts), updated_state

    async def derender_chat_stream(
        self,
        model: str,
        generate_chunk: GenerateTokensStreamResponse,
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
            prompt_token_ids: Prompt token IDs. Required on the parser path
                (validated by the caller) and optional otherwise, falling
                back to ``generate_chunk.prompt_token_ids``. See
                `DerenderChatStreamRequest.prompt_token_ids`.

        Returns:
            (chunk, updated_state) — the derendered SSE chunk and the state
            the client must pass to the next call.

        """
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
                state if state is not None else DerenderStreamState(),
                chat_request,
                prompt_tokens,
                prompt_token_ids,
            )

        tokenizer = self.renderer.get_tokenizer()
        skip_special, spaces_between = _decode_params(tokenizer, chat_request)
        # Seed on the first chunk only. A carried state already has the prompt.
        if state is None:
            state = _seed_stream_state(
                tokenizer,
                prompt_token_ids
                if prompt_token_ids is not None
                else generate_chunk.prompt_token_ids,
                skip_special_tokens=skip_special,
            )
        stream_choices: list[ChatCompletionResponseStreamChoice] = []
        updated_state = state

        for choice in generate_chunk.choices:
            delta_tids = choice.token_ids or []
            new_text, updated_state = await self._detokenize_delta_async(
                tokenizer,
                delta_tids,
                updated_state,
                skip_special_tokens=skip_special,
                spaces_between_special_tokens=spaces_between,
            )

            # NOTE: parser-configured servers dispatch to
            # _derender_chat_stream_parsed above and never reach this plain
            # path. That parsed path does not resolve logprobs yet; when it
            # does, mirror the generate chat streaming path, which suppresses
            # logprobs entirely when a parser is configured and reasoning is
            # hidden, because decoded logprob token text would leak hidden
            # reasoning.
            resolved_logprobs = None
            if choice.logprobs is not None:
                resolved_logprobs = _resolve_logprobs(
                    choice.logprobs,
                    tokenizer,
                    initial_context_token_ids=state.logprob_context_token_ids,
                )

            include_role = not updated_state.role_sent
            updated_state = updated_state.model_copy(
                update={
                    "role_sent": True,
                    "logprob_context_token_ids": _logprob_context_tail(
                        state.logprob_context_token_ids, delta_tids
                    ),
                }
            )

            delta = DeltaMessage(
                role="assistant" if include_role else None,
                content=new_text if new_text else None,
            )
            stream_choices.append(
                ChatCompletionResponseStreamChoice(
                    index=choice.index,
                    delta=delta,
                    logprobs=resolved_logprobs,
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
            metrics=generate_chunk.metrics,
        )
        return chunk, updated_state

    def _derender_chat_stream_parsed(
        self,
        parser_cls: type[Parser],
        model: str,
        generate_chunk: GenerateTokensStreamResponse,
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
        chunk's tokens for real.

        Every chunk, replayed or live, goes through one `parse_delta` call
        with the tokens it arrived with (more than one under e.g.
        speculative decoding). Replay recovers those boundaries from
        `state.output_chunk_lens`, so the rebuilt parser state matches
        standard serving, which calls `parse_delta` once per engine step.
        Text comes from a fresh incremental detokenizer with special tokens
        preserved (``skip_special_tokens=False``), seeded from the prompt
        tail on every call since replay starts from scratch.
        """
        tokenizer = self.renderer.get_tokenizer()

        parser = parser_cls(
            tokenizer,
            chat_request.tools,
            chat_template_kwargs=self._resolve_chat_template_kwargs(chat_request),
            model_config=self.model_config,
        )

        # Ephemeral incremental detok window, local to this call. Threaded
        # across both the replay and current chunk phases (via `nonlocal`)
        # so multi-byte characters split across that boundary still decode
        # correctly. Discarded once the call returns.
        skip_special, spaces_between = _decode_params(
            tokenizer, chat_request, preserve_special=True
        )
        detok_state = _seed_stream_state(
            tokenizer, prompt_token_ids, skip_special_tokens=skip_special
        )

        def _replay(token_ids: list[int], chunk_lens: list[int]) -> None:
            """Replay prior chunks through `parse_delta` to rebuild parser
            state, discarding the result. Never `finished` since that only
            applies to the current chunk.
            """
            nonlocal detok_state
            start = 0
            for chunk_len in chunk_lens:
                chunk = token_ids[start : start + chunk_len]
                start += chunk_len
                text, detok_state = self._detokenize_delta(
                    tokenizer,
                    chunk,
                    detok_state,
                    skip_special_tokens=skip_special,
                    spaces_between_special_tokens=spaces_between,
                )
                parser.parse_delta(
                    text,
                    chunk,
                    chat_request,
                    prompt_token_ids=prompt_token_ids,
                    finished=False,
                )

        # Replay history to reconstruct parser state. The result is thrown
        # away and only the current chunk's emission goes to the client.
        _replay(state.output_token_ids, state.output_chunk_lens)

        stream_choices: list[ChatCompletionResponseStreamChoice] = []
        output_token_ids = list(state.output_token_ids)
        output_chunk_lens = list(state.output_chunk_lens)
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
                # One parse_delta call for the whole chunk (producer
                # granularity), not one per token. See the granularity note
                # in this method's docstring.
                text, detok_state = self._detokenize_delta(
                    tokenizer,
                    delta_tids,
                    detok_state,
                    skip_special_tokens=skip_special,
                    spaces_between_special_tokens=spaces_between,
                )
                delta_message = parser.parse_delta(
                    text,
                    delta_tids,
                    chat_request,
                    prompt_token_ids=prompt_token_ids,
                    finished=is_finished,
                )
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

            if delta_tids:
                output_token_ids.extend(delta_tids)
                output_chunk_lens.append(len(delta_tids))

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
                "output_chunk_lens": output_chunk_lens,
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
            metrics=generate_chunk.metrics,
        )
        return chunk, updated_state

    async def derender_completion(
        self,
        generate_responses: list[GenerateTokensResponse],
        prompt_tokens: list[int] | None = None,
        completion_request: CompletionRequest | None = None,
        prompt_token_ids: list[list[int] | None] | None = None,
    ) -> tuple[list[CompletionResponseChoice], int, int]:
        return await self._derender_completion_async(
            generate_responses, prompt_tokens, completion_request, prompt_token_ids
        )

    def _derender_completion(
        self,
        generate_responses: list[GenerateTokensResponse],
        prompt_tokens: list[int] | None = None,
        completion_request: CompletionRequest | None = None,
        prompt_token_ids: list[list[int] | None] | None = None,
    ) -> tuple[list[CompletionResponseChoice], int, int]:
        n = len(generate_responses)
        prompt_tokens_list: list[int] = (
            prompt_tokens if prompt_tokens is not None else [0] * n
        )
        prompt_token_ids_list: list[list[int] | None] = (
            prompt_token_ids if prompt_token_ids is not None else [None] * n
        )

        tokenizer = self.renderer.get_tokenizer()
        skip_special, spaces_between = _decode_params(tokenizer, completion_request)
        choices: list[CompletionResponseChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        index = 0

        for gen, pt, seed_ids_override in zip(
            generate_responses, prompt_tokens_list, prompt_token_ids_list
        ):
            seed_ids = (
                seed_ids_override
                if seed_ids_override is not None
                else gen.prompt_token_ids
            )
            seed_state = _seed_stream_state(
                tokenizer, seed_ids, skip_special_tokens=skip_special
            )
            for choice in gen.choices:
                if not choice.token_ids:
                    raise ValueError(
                        f"choice {choice.index} in response {gen.request_id} "
                        "has empty or null token_ids"
                    )

                decoded_text, _ = self._detokenize_delta(
                    tokenizer,
                    choice.token_ids,
                    seed_state,
                    skip_special_tokens=skip_special,
                    spaces_between_special_tokens=spaces_between,
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
        generate_chunk: GenerateTokensStreamResponse,
        state: DerenderStreamState | None = None,
        prompt_tokens: int | None = None,
        completion_request: CompletionRequest | None = None,
        prompt_token_ids: list[int] | None = None,
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
            prompt_token_ids: Seeds the first chunk's decode. Falls back to
                ``generate_chunk.prompt_token_ids``.

        Returns:
            (chunk, updated_state) — the derendered chunk and updated state.

        """
        # See the equivalent check in derender_chat_stream: a single
        # DerenderStreamState is threaded through every choice in this
        # chunk, so more than one choice per chunk would corrupt the
        # detok window across choices.
        if len(generate_chunk.choices) > 1:
            raise ValueError(
                "derender_completion_stream expects at most one choice per chunk"
            )

        tokenizer = self.renderer.get_tokenizer()
        skip_special, spaces_between = _decode_params(tokenizer, completion_request)
        # Seed on the first chunk only. A carried state already has the prompt.
        if state is None:
            state = _seed_stream_state(
                tokenizer,
                prompt_token_ids
                if prompt_token_ids is not None
                else generate_chunk.prompt_token_ids,
                skip_special_tokens=skip_special,
            )
        stream_choices: list[CompletionResponseStreamChoice] = []
        updated_state = state

        for choice in generate_chunk.choices:
            delta_tids = choice.token_ids or []
            new_text, updated_state = await self._detokenize_delta_async(
                tokenizer,
                delta_tids,
                updated_state,
                skip_special_tokens=skip_special,
                spaces_between_special_tokens=spaces_between,
            )

            completion_logprobs = None
            if choice.logprobs is not None:
                resolved = _resolve_logprobs(
                    choice.logprobs,
                    tokenizer,
                    initial_context_token_ids=state.logprob_context_token_ids,
                )
                completion_logprobs = _convert_chat_logprobs_to_completion_logprobs(
                    resolved, initial_text_offset=state.logprob_text_offset
                )

            updated_state = updated_state.model_copy(
                update={
                    "logprob_context_token_ids": _logprob_context_tail(
                        state.logprob_context_token_ids, delta_tids
                    ),
                    "logprob_text_offset": state.logprob_text_offset + len(new_text),
                }
            )

            stream_choices.append(
                CompletionResponseStreamChoice(
                    index=choice.index,
                    text=new_text,
                    logprobs=completion_logprobs,
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
            metrics=generate_chunk.metrics,
        )
        return chunk, updated_state


# Number of preceding sampled token IDs `_correct_decoded_token` reads to
# repair U+FFFD from byte-fallback tokenization.
_LOGPROB_CONTEXT_WINDOW = 4


def _logprob_context_tail(
    context_token_ids: list[int], delta_token_ids: list[int]
) -> list[int]:
    """Advance the carried logprob context by this chunk's sampled tokens."""
    return (list(context_token_ids) + list(delta_token_ids))[-_LOGPROB_CONTEXT_WINDOW:]


def _decode_params(
    tokenizer: TokenizerLike,
    request: ChatCompletionRequest | CompletionRequest | None,
    preserve_special: bool = False,
) -> tuple[bool, bool]:
    """Derive `(skip_special_tokens, spaces_between_special_tokens)` the way
    the engine does (`IncrementalDetokenizer`).

    Args:
        tokenizer: Picks the engine's fast or slow detokenizer rules.
        request: The original request, if the caller supplied one.
        preserve_special: Keep special tokens so a parser can see markers.
            The serving side does this via `adjust_request`.

    """
    skip_special = (
        False
        if preserve_special
        else (request.skip_special_tokens if request is not None else True)
    )
    spaces_between = (
        request.spaces_between_special_tokens if request is not None else True
    )
    # Only the fast detokenizer forces spaces on when skipping special tokens.
    if skip_special and uses_fast_detokenizer(tokenizer):
        spaces_between = True
    return skip_special, spaces_between


def _seed_stream_state(
    tokenizer: TokenizerLike,
    prompt_token_ids: list[int] | None,
    skip_special_tokens: bool,
) -> DerenderStreamState:
    """Build the initial decode state from the prompt tail, the same way the
    engine primes its incremental detokenizer. Without it, Metaspace
    tokenizers drop the first output token's leading space.

    Returns an empty state when `prompt_token_ids` is empty or omitted.
    """
    if not prompt_token_ids:
        if prompt_token_ids is None and get_leading_space_marker(tokenizer) is not None:
            logger.warning_once(
                "derender got no prompt_token_ids, so the first output token "
                "may lose its leading space compared to the coupled endpoint. "
                "Pass prompt_token_ids from the render step to avoid this."
            )
        return DerenderStreamState()

    prev_tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
        tokenizer, prompt_token_ids, skip_special_tokens=skip_special_tokens
    )
    return DerenderStreamState(
        prev_tokens=prev_tokens,
        prefix_offset=prefix_offset,
        read_offset=read_offset,
    )


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
    logprobs: GenerateLogProbs,
    tokenizer: TokenizerLike,
    initial_context_token_ids: Sequence[int] = (),
) -> ChatCompletionLogProbs:
    """Convert generate's integer-id logprobs to the OpenAI chat shape.

    The generate server has no tokenizer, so `token` and `bytes` are filled
    here, with the same U+FFFD byte-fallback correction the coupled path
    applies (`_correct_decoded_token`, which needs the preceding sampled token
    ids as context). ``initial_context_token_ids`` seeds that context with
    sampled IDs from preceding chunks (streaming), so multi-byte characters
    split across chunk boundaries still resolve.
    """
    if logprobs.content is None:
        return ChatCompletionLogProbs()

    context_token_ids: list[int] = list(initial_context_token_ids)
    resolved_content = []

    for entry in logprobs.content:
        # One batch per position: the sampled id and its top-k ids.
        (token_str, token_bytes), *top_decoded = decode_token_ids(
            [entry.token_id, *(top.token_id for top in entry.top_logprobs)],
            tokenizer,
        )

        if token_str.endswith("\ufffd"):
            token_str = _correct_decoded_token(
                entry.token_id, context_token_ids, tokenizer
            )
            token_bytes = list(token_str.encode("utf-8"))

        resolved_top = []
        for top, (top_str, top_bytes) in zip(entry.top_logprobs, top_decoded):
            if top_str.endswith("\ufffd"):
                top_str = _correct_decoded_token(
                    top.token_id, context_token_ids, tokenizer
                )
                top_bytes = list(top_str.encode("utf-8"))
            resolved_top.append(
                ChatCompletionLogProb(
                    token=top_str,
                    logprob=top.logprob,
                    bytes=top_bytes,
                )
            )

        resolved_content.append(
            ChatCompletionLogProbsContent(
                token=token_str,
                logprob=entry.logprob,
                bytes=token_bytes,
                top_logprobs=resolved_top,
            )
        )

        context_token_ids.append(entry.token_id)

    return ChatCompletionLogProbs(content=resolved_content)


def _convert_chat_logprobs_to_completion_logprobs(
    logprobs: ChatCompletionLogProbs,
    initial_text_offset: int = 0,
) -> CompletionLogProbs:
    """Convert ChatCompletionLogProbs (per-token objects) to CompletionLogProbs
    (parallel flat lists) as required by the /v1/completions response schema.

    ``initial_text_offset`` keeps ``text_offset`` absolute across streaming
    chunks, mirroring the generate streaming path.
    """
    if logprobs.content is None:
        return CompletionLogProbs()

    tokens: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs_list: list[dict[str, float] | None] = []
    text_offset: list[int] = []

    offset = initial_text_offset
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
