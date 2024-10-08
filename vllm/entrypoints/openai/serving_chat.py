import asyncio
import json
import time
from typing import (AsyncGenerator, AsyncIterator, Callable, Dict, Final, List,
                    Optional)
from typing import Sequence as GenericSequence
from typing import Union

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ConversationMessage,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         load_chat_template,
                                         parse_chat_messages_futures)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb, ChatCompletionLogProbs,
    ChatCompletionLogProbsContent, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ErrorResponse, FunctionCall, RequestResponseMetadata,
    ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (BaseModelPath,
                                                    LoRAModulePath,
                                                    OpenAIServing,
                                                    PromptAdapterPath,
                                                    TextTokensPrompt)
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.tracing import (contains_trace_headers, extract_trace_headers,
                          log_tracing_disabled_warning)
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import iterate_with_cancellation, random_uuid

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine_client: EngineClient,
                 model_config: ModelConfig,
                 base_model_paths: List[BaseModelPath],
                 response_role: str,
                 *,
                 lora_modules: Optional[List[LoRAModulePath]],
                 prompt_adapters: Optional[List[PromptAdapterPath]],
                 request_logger: Optional[RequestLogger],
                 chat_template: Optional[str],
                 return_tokens_as_token_ids: bool = False,
                 enable_auto_tools: bool = False,
                 tool_parser: Optional[str] = None):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=lora_modules,
                         prompt_adapters=prompt_adapters,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        self.response_role = response_role
        self.use_tool_use_model_template = False
        self.chat_template = load_chat_template(chat_template)

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                "\"auto\" tool choice has been enabled please note that while"
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored.")

        self.tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None
        if self.enable_auto_tools:
            try:
                self.tool_parser = ToolParserManager.get_tool_parser(
                    tool_parser)
            except Exception as e:
                raise TypeError("Error: --enable-auto-tool-choice requires "
                                f"tool_parser:'{tool_parser}' which has not "
                                "been registered") from e

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse,
               ErrorResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        ChatCompletion API.

        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            model_config = self.model_config
            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            conversation, mm_data_future = parse_chat_messages_futures(
                request.messages, model_config, tokenizer)

            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            prompt: Union[str, List[int]]
            is_mistral_tokenizer = isinstance(tokenizer, MistralTokenizer)
            if is_mistral_tokenizer:
                prompt = apply_mistral_chat_template(
                    tokenizer,
                    messages=request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    tools=tool_dicts,
                    documents=request.documents,
                    **(request.chat_template_kwargs or {}),
                )
            else:
                prompt = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=request.chat_template or self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    tools=tool_dicts,
                    documents=request.documents,
                    **(request.chat_template_kwargs or {}),
                )
        except Exception as e:
            logger.exception("Error in applying chat template from request")
            return self.create_error_response(str(e))

        try:
            mm_data = await mm_data_future
        except Exception as e:
            logger.exception("Error in loading multi-modal data")
            return self.create_error_response(str(e))

        # validation for OpenAI tools
        # tool_choice = "required" is not supported
        if request.tool_choice == "required":
            return self.create_error_response(
                "tool_choice = \"required\" is not supported!")

        if not is_mistral_tokenizer and request.tool_choice == "auto" and not (
                self.enable_auto_tools and self.tool_parser is not None):
            # for hf tokenizers, "auto" tools requires
            # --enable-auto-tool-choice and --tool-call-parser
            return self.create_error_response(
                "\"auto\" tool choice requires "
                "--enable-auto-tool-choice and --tool-call-parser to be set")

        request_id = f"chat-{random_uuid()}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            if self.enable_auto_tools and self.tool_parser:
                request = self.tool_parser(tokenizer).adjust_request(
                    request=request)

            if isinstance(prompt, str):
                prompt_inputs = self._tokenize_prompt_input(
                    request,
                    tokenizer,
                    prompt,
                    truncate_prompt_tokens=request.truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                assert isinstance(prompt, list) and isinstance(
                    prompt[0], int
                ), "Prompt has to be either a string or a list of token ids"
                prompt_inputs = TextTokensPrompt(
                    prompt=tokenizer.decode(prompt), prompt_token_ids=prompt)

            assert prompt_inputs is not None

            sampling_params: Union[SamplingParams, BeamSearchParams]
            default_max_tokens = self.max_model_len - len(
                prompt_inputs["prompt_token_ids"])
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    default_max_tokens)
            else:
                sampling_params = request.to_sampling_params(
                    default_max_tokens)

            self._log_inputs(request_id,
                             prompt_inputs,
                             params=sampling_params,
                             lora_request=lora_request,
                             prompt_adapter_request=prompt_adapter_request)

            engine_inputs = TokensPrompt(
                prompt_token_ids=prompt_inputs["prompt_token_ids"])
            if mm_data is not None:
                engine_inputs["multi_modal_data"] = mm_data

            is_tracing_enabled = (await
                                  self.engine_client.is_tracing_enabled())
            trace_headers = None
            if is_tracing_enabled and raw_request:
                trace_headers = extract_trace_headers(raw_request.headers)
            if (not is_tracing_enabled and raw_request
                    and contains_trace_headers(raw_request.headers)):
                log_tracing_disabled_warning()

            if isinstance(sampling_params, BeamSearchParams):
                assert isinstance(self.engine_client,
                                    (AsyncLLMEngine,
                                    MQLLMEngineClient)), \
                    "Beam search is only supported with" \
                    "AsyncLLMEngine and MQLLMEngineClient."
                result_generator = self.engine_client.beam_search(
                    engine_inputs['prompt_token_ids'],
                    request_id,
                    sampling_params,
                )
            else:
                result_generator = self.engine_client.generate(
                    engine_inputs,
                    sampling_params,
                    request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=request.priority,
                )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        if raw_request:
            result_generator = iterate_with_cancellation(
                result_generator, raw_request.is_disconnected)

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, conversation, tokenizer,
                request_metadata)

        try:
            return await self.chat_completion_full_generator(
                request, result_generator, request_id, conversation, tokenizer,
                request_metadata)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        model_name = self.base_model_paths[0].name
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request))

        all_previous_token_ids: Optional[List[List[int]]]
        if tool_choice_auto:
            # These are only required in "auto" tool choice case
            previous_texts = [""] * num_choices
            all_previous_token_ids = [[]] * num_choices
        else:
            previous_texts, all_previous_token_ids = None, None

        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
                tool_parsers: List[Optional[ToolParser]] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except RuntimeError as e:
            logger.error("Error in tool parser creation: %s", e)
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        tool_parser = tool_parsers[i]
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)

                        # if usage should be included
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            # if continuous usage stats are requested, add it
                            if request.stream_options.continuous_usage_stats:
                                usage = UsageInfo(
                                    prompt_tokens=num_prompt_tokens,
                                    completion_tokens=0,
                                    total_tokens=num_prompt_tokens)
                                chunk.usage = usage
                            # otherwise don't
                            else:
                                chunk.usage = None

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo or request.continue_final_message:
                        last_msg_content: str = ""
                        if conversation and "content" in conversation[
                                -1] and conversation[-1].get("role") == role:
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        logprobs=None,
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)
                                if (request.stream_options and
                                        request.stream_options.include_usage):
                                    if (request.stream_options.
                                            continuous_usage_stats):
                                        usage = UsageInfo(
                                            prompt_tokens=num_prompt_tokens,
                                            completion_tokens=0,
                                            total_tokens=num_prompt_tokens)
                                        chunk.usage = usage
                                    else:
                                        chunk.usage = None

                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text
                    delta_message: Optional[DeltaMessage]

                    # handle streaming deltas for tools with named tool_choice
                    if tool_choice_function_name:
                        delta_message = DeltaMessage(tool_calls=[
                            DeltaToolCall(function=DeltaFunctionCall(
                                name=tool_choice_function_name,
                                arguments=delta_text),
                                          index=i)
                        ])

                    # handle streaming deltas for tools with "auto" tool choice
                    elif tool_choice_auto:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        assert tool_parser is not None
                        #TODO optimize manipulation of these lists
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        current_token_ids = previous_token_ids + list(
                            output.token_ids)

                        delta_message = (
                            tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=output.token_ids,
                                request=request))

                        # update the previous values for the next iteration
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids

                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        continue

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n

                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)

                        # handle usage stats if requested & if continuous
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            if request.stream_options.continuous_usage_stats:
                                completion_tokens = len(output.token_ids)
                                usage = UsageInfo(
                                    prompt_tokens=num_prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=num_prompt_tokens +
                                    completion_tokens,
                                )
                                chunk.usage = usage
                            else:
                                chunk.usage = None

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using guided decoding
                        if tool_parser:
                            index = len(
                                tool_parser.prev_tool_call_arr) - 1 if len(
                                    tool_parser.prev_tool_call_arr) > 0 else 0
                        else:
                            index = 0

                        if self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output) and tool_parser:
                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            expected_call = json.dumps(
                                tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}))

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[
                                index]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(
                                actual_call, "", 1)

                            # set that as a delta message
                            delta_message = DeltaMessage(tool_calls=[
                                DeltaToolCall(index=index,
                                              function=DeltaFunctionCall(
                                                  arguments=remaining_call).
                                              model_dump(exclude_none=True))
                            ])

                        # Send the finish response for each request.n only once
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason
                            if not (tool_parser
                                    and len(tool_parser.prev_tool_call_arr))
                            else "tool_calls",
                            stop_reason=output.stop_reason)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            if request.stream_options.continuous_usage_stats:
                                completion_tokens = len(output.token_ids)
                                usage = UsageInfo(
                                    prompt_tokens=num_prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=num_prompt_tokens +
                                    completion_tokens,
                                )
                                chunk.usage = usage
                            else:
                                chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if (request.stream_options
                    and request.stream_options.include_usage):
                completion_tokens = previous_num_tokens[i]
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens)

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            logger.error("error in chat completion stream generator: %s", e)
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = self.base_model_paths[0].name
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        assert final_res is not None

        choices: List[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                )
            else:
                logprobs = None

            # by default, tools are not used.
            tools_called = False

            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools
                    or not self.tool_parser) and not isinstance(
                        request.tool_choice,
                        ChatCompletionNamedToolChoiceParam):
                message = ChatMessage(role=role, content=output.text)

            # if the request uses tools and specified a tool choice
            elif request.tool_choice and type(
                    request.tool_choice) is ChatCompletionNamedToolChoiceParam:

                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        ToolCall(function=FunctionCall(
                            name=request.tool_choice.function.name,
                            arguments=output.text))
                    ])
                tools_called = True

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":

                message = ChatMessage(role=role, content=output.text)

            # handle when there are tools and tool choice is auto
            elif request.tools and (
                    request.tool_choice == "auto"
                    or request.tool_choice is None) and self.enable_auto_tools \
                    and self.tool_parser:

                try:
                    tool_parser = self.tool_parser(tokenizer)
                except RuntimeError as e:
                    logger.error("Error in tool parser creation: %s", e)
                    return self.create_error_response(str(e))

                tool_call_info = tool_parser.extract_tool_calls(
                    output.text, request=request)
                tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(role=role,
                                          content=tool_call_info.content,
                                          tool_calls=tool_call_info.tool_calls)

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    message = ChatMessage(role=role, content=output.text)

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion.")
                message = ChatMessage(role=role, content=output.text)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls" if tools_called else
                output.finish_reason if output.finish_reason else "stop",
                stop_reason=output.stop_reason)
            choices.append(choice_data)

        if request.echo or request.continue_final_message:
            last_msg_content = ""
            if conversation and "content" in conversation[-1] and conversation[
                    -1].get("role") == role:
                last_msg_content = conversation[-1]["content"] or ""

            for choice in choices:
                full_message = last_msg_content + (choice.message.content
                                                   or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=final_res.prompt_logprobs,
        )

        return response

    def _get_top_logprobs(
            self, logprobs: Dict[int, Logprob], top_logprobs: Optional[int],
            tokenizer: AnyTokenizer) -> List[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(token=(token := self._get_decoded_token(
                p[1],
                p[0],
                tokenizer,
                return_as_token_id=self.return_tokens_as_token_ids)),
                                  logprob=max(p[1].logprob, -9999.0),
                                  bytes=list(
                                      token.encode("utf-8", errors="replace")))
            for i, p in enumerate(logprobs.items())
            if top_logprobs and i < top_logprobs
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[Dict[int, Logprob]]],
        tokenizer: AnyTokenizer,
        num_output_top_logprobs: Optional[int] = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: List[ChatCompletionLogProbsContent] = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if self.return_tokens_as_token_ids:
                    token = f"token_id:{token_id}"

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    ))
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            self.return_tokens_as_token_ids,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=None if step_decoded is None else list(
                            step_decoded.encode("utf-8", errors="replace")),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                        ),
                    ))

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self,
                                              request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (request.tools and self.tool_parser and self.enable_auto_tools
                and request.tool_choice in ['auto', None])

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: Optional[DeltaMessage],
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        # yapf: disable
        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools and self.tool_parser and delta_message
            and delta_message.tool_calls and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )
