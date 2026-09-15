# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io

import numpy as np
import openai
import pytest
import soundfile as sf
import torch
from transformers import AutoConfig, AutoProcessor, Nemotron3_5AsrForRNNT

from tests.utils import RemoteOpenAIServer

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def test_nemotron_transcription_and_request_isolation(tmp_path):
    config = AutoConfig.from_pretrained(MODEL)
    config.encoder_config.hidden_size = 16
    config.encoder_config.intermediate_size = 32
    config.encoder_config.num_hidden_layers = 1
    config.encoder_config.num_attention_heads = 4
    config.encoder_config.num_key_value_heads = 4
    config.encoder_config.subsampling_conv_channels = 4
    config.encoder_config.max_position_embeddings = 512
    config.decoder_hidden_size = 8
    config.prompt_intermediate_size = 16
    config.max_symbols_per_step = 2

    processor = AutoProcessor.from_pretrained(MODEL)
    token_id = processor.tokenizer.encode("hello", add_special_tokens=False)[0]
    model = Nemotron3_5AsrForRNNT(config)
    with torch.no_grad():
        model.joint.head.weight.zero_()
        model.joint.head.bias.fill_(-10)
        model.joint.head.bias[token_id] = 10
    model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)

    expected = {}
    for seconds in (1.0, 0.5):
        inputs = processor(
            np.zeros(int(16_000 * seconds), dtype=np.float32),
            sampling_rate=16_000,
            language="en",
            return_tensors="pt",
        )
        inputs.pop("num_lookahead_tokens")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                decoder_start_token_id=config.blank_token_id,
                max_new_tokens=128,
            )
        expected[seconds] = processor.decode(
            output.sequences[0], skip_special_tokens=True
        )

    def audio_file(seconds: float) -> io.BytesIO:
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(int(16_000 * seconds)), 16_000, format="WAV")
        buffer.name = "speech.wav"
        buffer.seek(0)
        return buffer

    async def run_requests(server: RemoteOpenAIServer) -> None:
        async with server.get_async_client() as client:
            longer, shorter = await asyncio.gather(
                client.audio.transcriptions.create(
                    model=str(tmp_path), file=audio_file(1.0), language="en"
                ),
                client.audio.transcriptions.create(
                    model=str(tmp_path), file=audio_file(0.5), language="en"
                ),
            )
            assert longer.text == expected[1.0]
            assert shorter.text == expected[0.5]

            with pytest.raises(openai.BadRequestError):
                await client.audio.translations.create(
                    model=str(tmp_path), file=audio_file(0.5)
                )

            with pytest.raises(openai.BadRequestError):
                await client.audio.transcriptions.create(
                    model=str(tmp_path), file=audio_file(0.5), language="invalid"
                )

            with pytest.raises(openai.BadRequestError, match="max allowed: 0"):
                await client.audio.transcriptions.create(
                    model=str(tmp_path),
                    file=audio_file(0.5),
                    extra_body={"use_beam_search": True},
                )

    with RemoteOpenAIServer(
        str(tmp_path),
        [
            "--enforce-eager",
            "--dtype",
            "float16",
            "--max-model-len",
            "512",
            "--gpu-memory-utilization",
            "0.5",
        ],
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    ) as server:
        asyncio.run(run_requests(server))
