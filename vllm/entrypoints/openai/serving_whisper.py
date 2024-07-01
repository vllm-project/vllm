import codecs
import time
import torchaudio
import numpy as np
from dataclasses import dataclass, field
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Dict, Iterable,
                    List, Optional)
from typing import Sequence as GenericSequence
from typing import TypedDict, Union, cast, final

from fastapi import Request
from openai.types.chat import (ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartTextParam)

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionContentPartParam, ChatCompletionLogProb,
    ChatCompletionLogProbs, ChatCompletionLogProbsContent,
    ChatCompletionMessageParam, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    FunctionCall, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing)
from vllm.inputs import PromptInputs
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.multimodal.image import ImagePixelData
from vllm.multimodal.utils import (async_get_and_parse_image,
                                   get_full_image_text_prompt)
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob
from vllm.tracing import (contains_trace_headers, extract_trace_headers,
                          log_tracing_disabled_warning)
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = init_logger(__name__)

buffer_size = 4096
sample_rate = 16000
segment_length = sample_rate * 1
maxlen = 30
replaces = ['<|startoftranscript|>', '<|endoftext|>', '<|transcribe|>']
pattern = r'<\|\-?\d+\.?\d*\|>'
pattern_pair = r'<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>'


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]
    image_futures: List[Awaitable[ImagePixelData]] = field(
        default_factory=list)


class OpenAIServingWhisper(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 model_config: ModelConfig,
                 served_model_names: List[str],
                 max_size_whisper: 100,
                 ):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=None)
        
        self._check_whisper_mode(model_config.whisper_mode)
        self.max_size_whisper = max_size_whisper
    
    def _check_whisper_mode(self, whisper_mode: bool):
        if not whisper_mode:
            logger.warning(
                "whisper_mode is False. Whisper API will not work.")
        else:
            logger.info("Activating the server engine with whisper enabled.")

    async def create_audio_transcriptions(
        self,
        file,
        language,
        response_format,
        timestamp_granularities,
        repetition_penalty,
        stream,
        raw_request: Optional[Request] = None
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:

        if len(file) > self.max_size_whisper:
            return self.create_error_response(f"maximum size for file is {self.max_size_whisper}MB only")

        request_id = f"cmpl-{random_uuid()}"

        sampling_params = SamplingParams(
            max_tokens = self.engine.model_config.max_model_len - 4, 
            temperature = 0.0,
            skip_special_tokens = False,
            stop_token_ids = [50257],
            repetition_penalty = repetition_penalty
        )

        if isinstance(language, str) and language.lower() == 'null':
            language = None

        streamer = StreamReader(
            src=file,
            format=None,
            option=None,
            buffer_size=buffer_size
        )
        streamer.add_basic_audio_stream(
            frames_per_chunk=segment_length,
            sample_rate=sample_rate
        )
        stream_iterator = streamer.stream()

        is_tracing_enabled = await self.engine.is_tracing_enabled()
        trace_headers = None
        if is_tracing_enabled and raw_request:
            trace_headers = extract_trace_headers(raw_request.headers)
        if not is_tracing_enabled and raw_request and contains_trace_headers(
                raw_request.headers):
            log_tracing_disabled_warning()
        
        # Streaming response
        if request.stream:
            return self.audio_transcription_stream_generator(
                request, sampling_params, stream_iterator, language, request_id, trace_headers)
        else:
            try:
                return await self.audio_transcription_full_generator(
                    request, sampling_params, stream_iterator, language, request_id, trace_headers)
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

    async def generate(
        self, 
        sampling_params, 
        language, 
        wav_data, 
        last_timestamp,
        request_id,
        trace_headers,
    ):
        prompt_text = '<|startoftranscript|>'
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens = False)

        inputs: PromptInputs = {
            "prompt": prompt_text,
            "prompt_token_ids": prompt_ids,
            "whisper_data": wav_data
        }
        sampling_params = SamplingParams(max_tokens = 1, temperature = 0, skip_special_tokens = False)

        result_generator = self.engine.generate(
            inputs,
            sampling_params,
            request_id=request_id,
            lora_request = None,
            trace_headers=trace_headers,
        )
        


    async def audio_transcription_stream_generator(
        self, 
        request: ChatCompletionRequest,
        sampling_params, 
        stream_iterator, 
        language,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        
        wav_data = np.array([], dtype=np.float32)
        last_timestamp = 0
        for chunk in stream_iterator:
            wav_data = np.concatenate([wav_data, frame])
            audio_len = len(wav_data) / sample_rate
            if audio_len >= maxlen:
                async for t in generate(
                    sampling_params=sampling_params,
                    language=language,
                    wav_data=wav_data,
                    last_timestamp=last_timestamp,
                ):
                    yield t

                last_timestamp += audio_len
                wav_data = np.array([], dtype=np.float32)

            if len(wav_data):


        inputs: PromptInputs = {
            "prompt": prompt_text,
            "prompt_token_ids": prompt_ids,
        }

        # Send response for each token for each request.n (index)
        result_generator = self.engine.generate(
            inputs,
            sampling_params,
            request_id,
            lora_request,
            trace_headers=trace_headers,
        )
        assert request.n is not None
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        try:
            async for res in result_generator:
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)
                    for i in range(request.n):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role=role),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if conversation and conversation[-1].get(
                                "content") and conversation[-1].get(
                                    "role") == role:
                            last_msg_content = conversation[-1]["content"]

                        if last_msg_content:
                            for i in range(request.n):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    logprobs=None,
                                    model=model_name)
                                if (request.stream_options and
                                        request.stream_options.include_usage):
                                    chunk.usage = None
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                    out_logprobs = output.logprobs[
                        previous_num_tokens[i]:] if output.logprobs else None

                    if request.logprobs and request.top_logprobs is not None:
                        assert out_logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_chat_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.top_logprobs,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)

                    if request.tool_choice and type(
                            request.tool_choice
                    ) is ChatCompletionNamedToolChoiceParam:
                        delta_message = DeltaMessage(tool_calls=[
                            ToolCall(function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=delta_text))
                        ])
                    else:
                        delta_message = DeltaMessage(content=delta_text)

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
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                    else:
                        # Send the finish response for each request.n only once
                        prompt_tokens = len(res.prompt_token_ids)
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True

            if (request.stream_options
                    and request.stream_options.include_usage):
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=previous_num_tokens[i],
                    total_tokens=prompt_tokens + previous_num_tokens[i],
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

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self, request: ChatCompletionRequest, raw_request: Optional[Request],
        result_generator: AsyncIterator[RequestOutput], request_id: str,
        conversation: List[ConversationMessage]
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = self.served_model_names[0]
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
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
                )
            else:
                logprobs = None

            if request.tool_choice and type(
                    request.tool_choice) is ChatCompletionNamedToolChoiceParam:
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        ToolCall(function=FunctionCall(
                            name=request.tool_choice.function.name,
                            arguments=output.text))
                    ])
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, content=output.text)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason)
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == role:
                last_msg_content = conversation[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response
