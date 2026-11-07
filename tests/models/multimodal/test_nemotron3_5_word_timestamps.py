# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight offline alignment coverage; requires one full 80 GiB CUDA GPU."""

import hashlib
import os
from typing import cast

import numpy as np
import pytest
import soundfile as sf

from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.config import SpeechToTextConfig, SpeechToTextParams
from vllm.entrypoints.speech_to_text.offline import transcribe_with_word_timestamps
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.nemotron3_5_transcribe import (
    Nemotron3_5TranscribeForConditionalGeneration as Model,
)

from ...utils import create_new_process_for_each_test

MODEL = os.environ.get(
    "NEMOTRON_CTC_TEST_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
)


def _encoder_calls(llm) -> int:
    return sum(
        item["num_encoder_calls"]
        for worker in llm.collective_rpc("get_encoder_timing_stats")
        for item in worker.values()
    )


@pytest.mark.parametrize("num_speculative_tokens", [0, 2])
@create_new_process_for_each_test()
def test_offline_alignment_preserves_generation_and_reuses_audio(
    vllm_runner, num_speculative_tokens
):
    """Opt-in captures must survive completion and the next encoder-cache hit."""
    path = os.environ.get("NEMOTRON_CTC_TEST_AUDIO")
    if path:
        audio, sample_rate = sf.read(path, dtype="float32")
    else:
        audio, sample_rate = AudioAsset("mary_had_lamb").audio_and_sample_rate
    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / sample_rate
    spec = (
        {"method": "mtp", "num_speculative_tokens": num_speculative_tokens}
        if num_speculative_tokens
        else None
    )
    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=1,
        block_size=None,
        enable_chunked_prefill=True,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=2 * 1024**3,
        limit_mm_per_prompt={"audio": {"count": 1, "length": 480000}},
        hf_overrides={"enable_ctc_timestamps": True},
        speculative_config=spec,
        per_request_spec_decode_metrics=(
            "summary" if num_speculative_tokens else "none"
        ),
        enable_mm_processor_stats=True,
    ) as runner:
        llm = runner.llm
        prompt = cast(
            TokensPrompt,
            Model.get_generation_prompt(
                SpeechToTextParams(
                    audio=audio,
                    stt_config=SpeechToTextConfig(sample_rate=sample_rate),
                    model_config=llm.model_config,
                )
            ),
        )
        prompt["multi_modal_uuids"] = {
            "audio": hashlib.sha256(audio.tobytes()).hexdigest()
        }
        params = SamplingParams(temperature=0, max_tokens=256)
        params_before = params.clone()
        _encoder_calls(llm)
        baseline = llm.generate(prompt, params, use_tqdm=False)[0]
        assert _encoder_calls(llm) > 0

        original_kwargs = dict(prompt.get("mm_processor_kwargs", {}))
        first = transcribe_with_word_timestamps(llm, prompt, params)[0]
        assert prompt.get("mm_processor_kwargs", {}) == original_kwargs
        assert _encoder_calls(llm) > 0
        assert first.error is None
        assert first.words
        assert first.output.outputs[0].token_ids == baseline.outputs[0].token_ids
        assert params == params_before
        for word in first.words:
            assert word.word
            assert 0 <= word.start <= word.end <= duration + 1 / 16000
        stats = llm.collective_rpc("get_audio_alignment_stats")[0]
        assert stats["handles"] == 0
        assert stats["captures"] == 1
        assert stats["bytes"] > 0

        second = transcribe_with_word_timestamps(llm, prompt, params)[0]
        assert _encoder_calls(llm) == 0
        assert second.error is None
        assert second.words == first.words
        assert second.output.outputs[0].token_ids == first.output.outputs[0].token_ids
        if num_speculative_tokens:
            metrics = second.output.outputs[0].spec_decode_metrics
            assert metrics is not None
            summary = metrics.to_dict()
            assert summary["num_draft_tokens"] > 0
            assert summary["num_accepted_draft_tokens"] > 0

        truncated_params = SamplingParams(
            temperature=0, max_tokens=1, ignore_eos=True, skip_special_tokens=False
        )
        truncated_before = truncated_params.clone()
        mixed = transcribe_with_word_timestamps(
            llm, [prompt, prompt], [truncated_params, params]
        )
        assert len(mixed) == 2
        partial, complete = mixed
        partial_output = partial.output.outputs[0]
        assert partial_output.finish_reason == "length"
        assert len(partial_output.token_ids) == 1
        assert partial_output.text
        assert partial.error == "truncated_transcript"
        assert partial.words == []
        assert complete.error is None
        assert complete.words == first.words
        assert complete.output.outputs[0].text == first.output.outputs[0].text
        assert truncated_params == truncated_before
        assert params == params_before
        mixed_stats = llm.collective_rpc("get_audio_alignment_stats")[0]
        assert mixed_stats["handles"] == mixed_stats["workspace_bytes"] == 0

        with pytest.raises(ValueError, match="n=1"):
            transcribe_with_word_timestamps(
                llm, prompt, SamplingParams(n=2, temperature=1, seed=0, max_tokens=16)
            )
        llm.enqueue(prompt, params, use_tqdm=False)
        with pytest.raises(ValueError, match="queued requests"):
            transcribe_with_word_timestamps(llm, prompt, params)
        queued = llm.wait_for_completion(use_tqdm=False)
        assert len(queued) == 1 and queued[0].finished
        llm.llm_engine.reset_encoder_cache()
        cleared = llm.collective_rpc("get_audio_alignment_stats")[0]
        assert cleared["captures"] == cleared["handles"] == cleared["bytes"] == 0
