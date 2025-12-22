# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict
import numpy as np
from mistral_common.protocol.instruct.chunk import RawAudio

from mistral_common.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset


def _get_engine(path: str) -> LLM:
    engine_args = EngineArgs(
        model=path,
        max_model_len=8192,
        max_num_seqs=1,
        limit_mm_per_prompt={"audio": 1},
        config_format="mistral",
        load_format="mistral",
        tokenizer_mode="mistral",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
    )
    return LLM(**asdict(engine_args))


def test_voxtral_streaming_forward():
    audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]

    model_name = "mistralai/voxstral-3b-streaming-new"
    tokenizer = MistralTokenizer.from_hf_hub(model_name)

    def from_file(file_path: str):
        audio = Audio.from_file(file_path, strict=False)
        req = TranscriptionRequest(
            audio=RawAudio.from_audio(audio),
            streaming=StreamingMode.OFFLINE,
            language=None,
        )

        tokenized = tokenizer.instruct_tokenizer.encode_transcription(req)

        return (tokenized.tokens, tokenized.audios[0].audio_array)

    tokenized_list = [
        from_file(audio_asset.get_local_path()) for audio_asset in audio_assets
    ]

    inputs = []
    sampling_params = []

    for tokens, audio_array in tokenized_list:
        num_samples = audio_array.shape[0]

        audio_tokens = int((num_samples / 16_000) / 0.08)

        max_tokens = audio_tokens - len(tokens) - 2
        sampling_params.append(SamplingParams(temperature=0.0, max_tokens=max_tokens))

        input_dict = {
            "multi_modal_data": {"audio": [(audio_array, None)]},
            "prompt_token_ids": tokens,
        }
        inputs.append(input_dict)

    llm = _get_engine(model_name)
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
    )

    texts = [out.outputs[0].text for out in outputs]
    print(texts)
