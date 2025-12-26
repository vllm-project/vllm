# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM streaming
"""
import asyncio
import os
import signal
import time

import psutil
import torch
from mistral_common.tokens.tokenizers.audio import Audio, AudioConfig, TranscriptionFormat
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.data import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreRequest
from vllm.multimodal.cache import MultiModalBatchedField
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldElem, MultiModalFlatField, MultiModalKwargsItem, PlaceholderRange

async def main():
    tok_file = "/mnt/vast/ci/models/voxtral_transcribe_streaming_3b_new/consolidated/tekken.json"
    model_file = "mistralai/voxstral-3b-streaming-new"

    tokenizer = MistralTokenizer.from_file(tok_file)

    config: AudioConfig = tokenizer.instruct_tokenizer.audio_encoder.audio_config
    config.transcription_format = TranscriptionFormat.STREAMING
    config.transcription_delay_ms = 480

    audio = Audio.from_file("/mnt/vast/ci/fixtures/audio/obama.mp3", strict=False)
    audio.resample(config.sampling_rate)
    audio_len_in_samples = audio.audio_array.shape[0]

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
    end_idx = streaming_delay_in_samples + stream_size_in_samples

    ins_tokenizer = tokenizer.instruct_tokenizer
    prompt = ins_tokenizer.start() + [ins_tokenizer.STREAMING_PAD] * config.num_delay_tokens

    engine_args = AsyncEngineArgs(
        model=model_file,
        tokenizer=model_file,
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
    request_id = "session"

    def get_prompt(tokens: list[int], start: int, end: int, len: int) -> EngineCoreRequest:
        _end = end + look_ahead
        _start = max(0, start - look_back)

        multi_modal_data = {"audio": (audio.audio_array[_start: _end], None)}
        return TokensPrompt(prompt_token_ids=tokens, multi_modal_data=multi_modal_data)

    while end_idx <= audio_len_in_samples:
        core_request = get_prompt(prompt, start_idx, end_idx, len(prompt))
        is_last = end_idx + stream_size_in_samples >= audio_len_in_samples
        async for resp in engine.generate(prompt=core_request, sampling_params=sampling_params, request_id=request_id, resumable=not is_last):
            response = resp

        prompt = response.outputs[0].token_ids[-1:]
        start_idx = end_idx
        end_idx += stream_size_in_samples
        for t in prompt:
            print(tokenizer.instruct_tokenizer.tokenizer.id_to_piece(t) if t != 40 else "_", end="", flush=True)

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
