# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM streaming
"""
import asyncio
import io
import soundfile as sf
from typing import Any
import uuid

from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.tokens.tokenizers.audio import Audio, AudioConfig, TranscriptionFormat
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from openai import OpenAI

model = "mistralai/voxstral-3b-streaming-new"
tokenizer = MistralTokenizer.from_hf_hub(model)
config: AudioConfig = tokenizer.instruct_tokenizer.audio_encoder.audio_config

SAMPLE_RATE = 16_000

openai_api_key = "EMPTY"
openai_api_base = "http://slurm-h100-reserved-rno-199-065:21001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

async def main():
    model = "mistralai/voxstral-3b-streaming-new"
    tokenizer = MistralTokenizer.from_hf_hub(model)

    files = [
        "/mnt/vast/ci/fixtures/audio/fn_calling.wav",
        "/mnt/vast/ci/fixtures/audio/bcn_weather.mp3",
        # "/mnt/vast/ci/fixtures/audio/obama_58min.mp3",
        # "/mnt/vast/ci/fixtures/audio/obama.mp3",
    ]
    for file in files:
        await sub_main(tokenizer, file)

async def sub_main(tokenizer: MistralTokenizer, file: str):
    config: AudioConfig = tokenizer.instruct_tokenizer.audio_encoder.audio_config
    config.transcription_format = TranscriptionFormat.STREAMING

    inst_tok = tokenizer.instruct_tokenizer
    audio = Audio.from_file(file, strict=False)
    audio.resample(inst_tok.audio_encoder.audio_config.sampling_rate)
    audio = RawAudio.from_audio(audio)
    enc = inst_tok._encode_audio(audio.data, is_online_streaming=False)
    audio = enc.audios[0]
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

    # add 2.5ms
    look_ahead = get_len_in_samples(0.0025)
    # the last 4 log mel frame => 40ms
    # last log mel needs look back of 12.5ms on top
    look_back = get_len_in_samples(0.0525)

    start_idx = 0
    end_idx = streaming_delay_in_samples + stream_size_in_samples

    request_id = str(uuid.uuid4())

    def get_prompt(start: int, end: int):
        _end = end + look_ahead
        _start = max(0, start - look_back)

        audio_arr = audio.audio_array[_start: _end]
        return audio_arr

    def decode_print(t: int) -> str:
        if t == 32:
            print("_", end="", flush=True) 
        elif t == 33:
            print("{w}", end="", flush=True) 
        else:
            print(tokenizer.instruct_tokenizer.tokenizer.id_to_piece(t), end="", flush=True)

    collected_out = []
    prev_tokens = []
    while end_idx <= audio_len_in_samples:
        audio_chunk = get_prompt(start_idx, end_idx)
        is_last = end_idx + stream_size_in_samples >= audio_len_in_samples

        buffer = io.BytesIO()
        sf.write(buffer, audio_chunk, SAMPLE_RATE, format="wav")
        buffer.seek(0)

        req = {"model": model, "temperature": 0.0, "file": buffer}

        req["extra_body"] = {
            "max_completion_tokens": 1,
            "resumable": True,
        }
        if prev_tokens:
            req["extra_body"]["prev_token"] = prev_tokens[-1]

        req["extra_headers"] = {
            "X-Request-Id": request_id,
        }
        response = client.audio.transcriptions.create(**req)

        prev_tokens = response.tokens
        start_idx = end_idx
        end_idx += stream_size_in_samples
        for t in prev_tokens[-1:]:
            decode_print(t)
            collected_out.append(t)

    left_over = audio_len_in_samples - end_idx
    assert left_over == -stream_size_in_samples, f"{left_over=} {audio_len_in_samples=} {end_idx=}"
    print()
    print("end!")
    print(tokenizer.decode(collected_out))


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
