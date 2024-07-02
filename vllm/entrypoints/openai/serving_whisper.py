import codecs
import time
import re
import json
import torchaudio
import numpy as np
from torchaudio.io import StreamReader
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
    Segment, TranscriptionVerboseJsonResponse, TranscriptionJsonResponse)
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

def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class OpenAIServingWhisper(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 model_config: ModelConfig,
                 served_model_names: List[str],
                 max_size_mb_whisper: 200,
                 ):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=None)
        
        self._check_whisper_mode(model_config.whisper_mode)
        self.model_config = model_config
        self.max_size_mb_whisper = max_size_mb_whisper * 1024 * 1024
    
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

        if len(file) > self.max_size_mb_whisper:
            return self.create_error_response(f"maximum size for `file` is {self.max_size_mb_whisper}MB only")
        
        if timestamp_granularities.lower().strip() != 'segment':
            return self.create_error_response("currently `timestamp_granularities` only support `segment`")

        if response_format.lower() not in {'text', 'json', 'verbose_json', 'srt'}:
            return self.create_error_response(
                'currently `response_format` only support `text`, `json`, `verbose_json` and `srt`')

        request_id = f"cmpl-{random_uuid()}"

        sampling_params = SamplingParams(
            max_tokens = self.model_config.max_model_len - 4, 
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
        if stream:
            return self.audio_transcription_stream_generator(
                sampling_params, 
                stream_iterator, 
                language, 
                response_format, 
                request_id, 
                trace_headers,
                raw_request,
            )
        else:
            try:
                return await self.audio_transcription_full_generator(
                    sampling_params, 
                    stream_iterator, 
                    language, 
                    response_format, 
                    request_id, 
                    trace_headers,
                    raw_request,
                )
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

    async def generate(
        self, 
        sampling_params, 
        language,
        wav_data, 
        last_timestamp,
        last_i,
        response_format,
        request_id,
        trace_headers,
        raw_request,
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
                request_id=request_id + '-predict-lang',
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
            language = self.tokenizer.decode([lang_token])[2:-2]
        else:
            lang_token = self.tokenizer.encode(f'<|{language}|>', add_special_tokens = False)[0]

        prompt_ids = [50258, lang_token, 50360, 50365]
        inputs: PromptInputs = {
            "prompt": None,
            "prompt_token_ids": prompt_ids,
            "whisper_data": wav_data
        }

        result_generator = self.engine.generate(
            inputs,
            sampling_params,
            request_id=request_id + f'-{last_i}',
            lora_request = None,
            trace_headers=trace_headers,
        )

        texts = f'<|{language}|><|{last_timestamp}|>'

        if response_format != 'srt':
            text = texts
            if response_format == 'json':
                text = json.dumps({'token': texts})
            
            yield text
        
        """
        [CompletionOutput(index=0, text=' and', token_ids=[293], cumulative_logprob=-1.7037980556488037, logprobs=None, finish_reason=None, stop_reason=None)]
        """
        async for res in result_generator:

            token = self.tokenizer.convert_ids_to_tokens([res.outputs[0].token_ids[-1]])
            text = self.tokenizer.convert_tokens_to_string(token)

            for r in replaces:
                text = text.replace(r, '')
            matches = re.findall(pattern, text)
            for match in matches:
                timestamp = float(match.split('|')[1])
                timestamp += last_timestamp
                timestamp = f'<|{timestamp}|>'
                text = text.replace(match, timestamp)
            if len(text):
                texts += text
                matches = re.findall(pattern_pair, texts)
                if response_format == 'srt':
                    if len(matches):
                        match = matches[0]
                        if len(match[1]) > 2:
                            start = float(match[0]) + last_timestamp
                            end = float(match[-1]) + last_timestamp
                            text_between = match[1].strip()
                            ids = f"{last_i + 1}\n"
                            r = [
                                ids,
                                f"{format_timestamp(start, always_include_hours=True, decimal_marker=',')} --> ",
                                f"{format_timestamp(end, always_include_hours=True, decimal_marker=',')}\n",
                                f"{text_between.replace('-->', '->')}\n"]

                            combined = ''.join(r) + '\n'
                            last_i += 1
                            yield combined
                        
                        texts = text.split('|>')[-2] + '|>'
                else:
                    if response_format == 'json':
                        text = json.dumps({'token': text})

                    yield text


    async def audio_transcription_stream_generator(
        self, 
        sampling_params, 
        stream_iterator, 
        language,
        response_format,
        request_id: str,
        trace_headers,
        raw_request,
    ) -> AsyncGenerator[str, None]:
        wav_data = np.array([], dtype=np.float32)
        last_i = 0
        last_timestamp = 0.0
        try:
            for chunk in stream_iterator:
                frame = chunk[0][:, 0].numpy()
                wav_data = np.concatenate([wav_data, frame])
                audio_len = len(wav_data) / sample_rate
                if audio_len >= maxlen:
                    async for t in self.generate(
                        sampling_params=sampling_params,
                        language=language,
                        wav_data=wav_data,
                        last_timestamp=last_timestamp,
                        last_i=last_i,
                        response_format=response_format,
                        request_id=request_id,
                        trace_headers=trace_headers,
                        raw_request=raw_request,
                    ):
                        yield f"data: {t}\n\n"
                        last_i += 1

                    last_timestamp += audio_len
                    wav_data = np.array([], dtype=np.float32)

            if len(wav_data):
                async for t in self.generate(
                    sampling_params=sampling_params,
                    language=language,
                    wav_data=wav_data,
                    last_timestamp=last_timestamp,
                    last_i=last_i,
                    response_format=response_format,
                    request_id=request_id,
                    trace_headers=trace_headers,
                    raw_request=raw_request,
                ):
                    yield f"data: {t}\n\n"
                    last_i += 1

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"
    
    async def audio_transcription_full_generator(
        self,
        sampling_params, 
        stream_iterator, 
        language,
        response_format,
        request_id: str,
        trace_headers,
        raw_request,
    ):
        tokens = []
        async for data in self.audio_transcription_stream_generator(
            sampling_params=sampling_params,
            stream_iterator=stream_iterator,
            language=language,
            response_format='json',
            request_id=request_id,
            trace_headers=trace_headers,
            raw_request=raw_request,
        ):
            if isinstance(data, str):
                if '[DONE]' in data:
                    break
                data = json.loads(data.split('data:')[1].strip())
                tokens.append(data['token'])

        tokens = ''.join(tokens)
        lang = tokens.split('|')[1]
        matches = re.findall(pattern_pair, tokens)
        print(matches)
        segments = []
        all_texts = []
        for no, (start, substring, end) in enumerate(matches):
            start_timestamp = float(start)
            end_timestamp = float(end)
            segment = Segment(
                id=no,
                seek=0,
                start=start_timestamp,
                end=end_timestamp,
                text=substring.strip(),
                tokens=self.tokenizer.encode(substring.strip(), add_special_tokens=False),
                temperature=0.0,
                avg_logprob=0.0,
                compression_ratio=1.0,
                no_speech_prob=0.0,
            )
            segments.append(segment)
            all_texts.append(substring)

        all_texts = ''.join(all_texts).strip()
        if response_format == 'verbose_json':
            return TranscriptionVerboseJsonResponse(
                task='transcribe',
                language=lang,
                duration=segments[-1].end,
                text=all_texts,
                segments=segments
            )
        elif response_format == 'json':
            return TranscriptionJsonResponse(
                text=all_texts
            )
        else:
            return all_texts