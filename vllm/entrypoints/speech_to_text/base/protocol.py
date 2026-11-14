# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Literal, TypeAlias

import torch

from vllm.config.speech_to_text import VADConfig
from vllm.exceptions import VLLMValidationError

## Protocols for Audio
AudioResponseFormat: TypeAlias = Literal["json", "text", "srt", "verbose_json", "vtt"]
TranscriptionResponseFormat: TypeAlias = AudioResponseFormat | Literal["diarized_json"]
_LONG_INFO = torch.iinfo(torch.long)

# Any VAD duration longer than the audio acts like the whole audio, and silero
# converts these to floats, which overflows for huge ints.
_VAD_MAX_DURATION_MS = 24 * 60 * 60 * 1000


def validate_vad_config(vad_config: VADConfig) -> None:
    if (
        vad_config.neg_threshold is not None
        and vad_config.neg_threshold > vad_config.threshold
    ):
        raise VLLMValidationError(
            "vad_config.neg_threshold must not exceed vad_config.threshold.",
            parameter="vad_config.neg_threshold",
            value=vad_config.neg_threshold,
        )
