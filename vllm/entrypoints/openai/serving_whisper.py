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
    ):

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
        if language is None:
            prompt_ids = [50258]
            inputs: PromptInputs = {
                "prompt": None,
                "prompt_token_ids": prompt_ids,
                "whisper_data": wav_data
            }
            lang_sampling_params = SamplingParams(
                max_tokens = 1, temperature = 0, skip_special_tokens = False)

            result_generator = self.engine.generate(
                inputs,
                lang_sampling_params,
                request_id=request_id,
                lora_request = None,
                trace_headers=trace_headers,
            )
            async for res in result_generator:
                if raw_request is not None and await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.engine.abort(request_id)
                    yield self.create_error_response("Client disconnected")
                final_res = res
            assert final_res is not None

            lang_token = final_res.outputs[0].token_ids[0]
        else:
            lang_token = self.tokenizer.encode(
                lang_token = f'<|{language}|>', add_special_tokens = False)[0]

        prompt_ids = [50258, lang_token, 50360, 50365]
        inputs: PromptInputs = {
            "prompt": None,
            "prompt_token_ids": prompt_ids,
            "whisper_data": wav_data
        }

        text = processor.tokenizer.decode(prompt_ids, decode_with_timestamps = True)

        async for res in result_generator:
            print(res.outputs)
            yield res.outputs


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
        try:
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
                    async for t in generate(
                        sampling_params=sampling_params,
                        language=language,
                        wav_data=wav_data,
                        last_timestamp=last_timestamp,
                    ):
                        yield t

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"