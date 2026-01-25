# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM streaming
"""
import asyncio
import os
import signal
from collections.abc import AsyncGenerator

from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
import psutil
from mistral_common.tokens.tokenizers.audio import Audio, AudioConfig
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.data import TokensPrompt
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput
from vllm.v1.engine.core import EngineCoreRequest


class StreamingInputGenerator(AsyncGenerator):
    def __init__(self):
        self.queue: asyncio.Queue[StreamingInput | None] = asyncio.Queue()
    
    async def add(self, prompt: TokensPrompt):
        await self.queue.put(StreamingInput(prompt))
    
    def stop(self):
        self.queue.put_nowait(None)
    
    def __aiter__(self):
        return self
    
    async def __anext__(self) -> StreamingInput:
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        return item
    
    async def asend(self, value):
        return await self.__anext__()
    
    async def athrow(self, typ, val=None, tb=None):
        raise StopAsyncIteration


async def main():
    model_id = "mistralai/Voxtral-Mini-3B-Realtime-2602"
    tokenizer = MistralTokenizer.from_hf_hub(model_id)
    config: AudioConfig = tokenizer.instruct_tokenizer.audio_encoder.audio_config

    audio = Audio.from_file("/mnt/vast/ci/fixtures/audio/obama.mp3", strict=False)
    req = TranscriptionRequest(streaming=StreamingMode.ONLINE, audio=RawAudio.from_audio(audio), language=None)

    audio_enc = tokenizer.encode_transcription(req)

    audio = audio_enc.audios[0]
    prompt = audio_enc.tokens

    stream_size_in_s = 1 / config.frame_rate
    _stream_size_in_samples = 16000 * stream_size_in_s
    assert _stream_size_in_samples.is_integer(), "Stream size in samples must exactly match an integer"
    stream_size_in_samples  = int(_stream_size_in_samples)

    def get_len_in_samples(len_in_s: float) -> int:
        _len_in_s = 16000 * len_in_s
        assert _len_in_s.is_integer(), _len_in_s
        len_in_s = int(_len_in_s)

        return len_in_s

    streaming_delay_in_s: float = (config.transcription_delay_ms / 1000)
    streaming_delay_in_samples = get_len_in_samples(streaming_delay_in_s)

    look_ahead = get_len_in_samples(0.0025)
    # the last 4 log mel frame => 40ms
    # last log mel needs look back of 12.5ms on top
    look_back = get_len_in_samples(0.0525)

    start_idx = 0
    left_pad_in_samples = config.raw_audio_length_per_tok * config.n_left_pad_tokens
    end_idx = streaming_delay_in_samples + stream_size_in_samples + left_pad_in_samples

    engine_args = AsyncEngineArgs(
        model=model_id,
        tokenizer=model_id,
        config_format="mistral",
        load_format="mistral",
        tokenizer_mode="mistral",
        enforce_eager=True,
        max_num_batched_tokens=8192,
        # enable_log_requests=True,
        enable_prefix_caching=False,
    )
    engine = AsyncLLM.from_engine_args(engine_args, usage_context=UsageContext.API_SERVER)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    request_id = "session-123456789"

    streaming_generator = StreamingInputGenerator()

    def get_core_req(tokens: list[int], start: int, end: int) -> EngineCoreRequest:
        _end = end + look_ahead
        _start = max(0, start - look_back)

        multi_modal_data = {"audio": (audio.audio_array[_start: _end], None)}
        return TokensPrompt(prompt_token_ids=tokens, multi_modal_data=multi_modal_data)

    audio_len_in_samples = audio.audio_array.shape[0]

    req = get_core_req(prompt, start_idx, end_idx)
    await streaming_generator.add(req)
    async for resp in engine.generate(
        prompt=streaming_generator, 
        sampling_params=sampling_params, 
        request_id=request_id
    ):
        is_last = end_idx + stream_size_in_samples >= audio_len_in_samples

        prompt = resp.outputs[0].token_ids[-1:]
        start_idx = end_idx
        end_idx += stream_size_in_samples
        for t in prompt:
            print(tokenizer.instruct_tokenizer.tokenizer.id_to_piece(t) if t != 32 else "_", end="", flush=True)

        req = get_core_req(prompt, start_idx, end_idx)

        if is_last:
            break

        await streaming_generator.add(req)

    left_over = audio_len_in_samples - end_idx
    assert left_over == -stream_size_in_samples, f"{left_over=} {audio_len_in_samples=} {end_idx=}"
    print()
    print("end!")
    engine.shutdown()
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGTERM)



if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
