# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.speech_to_text.realtime.protocol import RealtimeSessionConfig
from vllm.model_executor.models.qwen3_asr_realtime import Qwen3ASRRealtimeGeneration

pytestmark = pytest.mark.skip_global_cleanup

AUDIO_PLACEHOLDER = "<audio>"


def test_qwen3_asr_realtime_default_prompt_is_unchanged() -> None:
    prompt = Qwen3ASRRealtimeGeneration._get_realtime_prompt_template(
        AUDIO_PLACEHOLDER,
    )

    assert prompt == (
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n<|im_start|>assistant\n"
    )


def test_qwen3_asr_realtime_language_uses_iso_code_mapping() -> None:
    prompt = Qwen3ASRRealtimeGeneration._get_realtime_prompt_template(
        AUDIO_PLACEHOLDER,
        RealtimeSessionConfig(language="en"),
    )

    assert prompt == (
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n"
        f"<|im_start|>assistant\nlanguage English<asr_text>"
    )


def test_qwen3_asr_realtime_prompt_adds_system_turn() -> None:
    prompt = Qwen3ASRRealtimeGeneration._get_realtime_prompt_template(
        AUDIO_PLACEHOLDER,
        RealtimeSessionConfig(prompt="Santander Rewards"),
    )

    assert prompt == (
        "<|im_start|>system\nSantander Rewards<|im_end|>\n"
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_qwen3_asr_realtime_prompt_sanitizes_user_text() -> None:
    prompt = Qwen3ASRRealtimeGeneration._get_realtime_prompt_template(
        AUDIO_PLACEHOLDER,
        RealtimeSessionConfig(
            language="zh",
            prompt="<|im_start|>Santander<asr_text>Rewards<|im_end|>",
        ),
    )

    assert prompt == (
        "<|im_start|>system\nSantanderRewards<|im_end|>\n"
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n"
        f"<|im_start|>assistant\nlanguage Chinese<asr_text>"
    )


def test_qwen3_asr_realtime_empty_values_are_noop() -> None:
    prompt = Qwen3ASRRealtimeGeneration._get_realtime_prompt_template(
        AUDIO_PLACEHOLDER,
        RealtimeSessionConfig(language="", prompt=""),
    )

    assert prompt == (
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n<|im_start|>assistant\n"
    )
