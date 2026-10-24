# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoModelForSeq2SeqLM

from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
from vllm.model_executor.models.vibevoice_asr import (
    VibeVoiceAsrForConditionalGeneration,
)
from vllm.multimodal.audio import AudioResampler

from ....conftest import AudioTestAssets, HfRunner, VllmRunner
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

MODEL_ID = "microsoft/VibeVoice-ASR-HF"
SAMPLE_RATE = 24_000


@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_ID)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    audio, sr = audio_assets[0].audio_and_sample_rate
    audio = AudioResampler(target_sr=SAMPLE_RATE).resample(audio, orig_sr=sr)
    prompt = VibeVoiceAsrForConditionalGeneration.get_generation_prompt(
        SpeechToTextParams(
            audio=audio,
            stt_config=SpeechToTextConfig(sample_rate=SAMPLE_RATE),
            model_config=None,
        )
    )["prompt"]

    with vllm_runner(
        MODEL_ID,
        dtype="half",
        max_model_len=2048,
        max_num_seqs=1,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [prompt],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[(audio, SAMPLE_RATE)],
        )

    with hf_runner(MODEL_ID, dtype="half", auto_cls=AutoModelForSeq2SeqLM) as hf_model:
        # vLLM omits the VAE sampling noise that HF adds to the acoustic latents.
        hf_model.model.config.acoustic_tokenizer_encoder_config.vae_std = 0.0
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            [prompt],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[(audio, SAMPLE_RATE)],
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
