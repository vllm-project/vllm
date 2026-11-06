# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only CrisperWhisper 2.0 model.

CrisperWhisper 2.0 is a Whisper fine-tune whose decoder prompt starts with
mode tags that select a verbatim or an intended (clean) transcript.
"""

from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, PromptType, TextPrompt
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY

from .utils import WeightsMapper
from .whisper import (
    WhisperDummyInputsBuilder,
    WhisperForConditionalGeneration,
    WhisperMultiModalProcessor,
    WhisperProcessingInfo,
)

logger = init_logger(__name__)

_MODES = ("verbatim", "intended")
_NUM_MODE_TAGS = 5


class CrisperWhisperMultiModalProcessor(WhisperMultiModalProcessor):
    # The mode tags come before <|startoftranscript|>.
    skip_decoder_start_token: bool = True


@MULTIMODAL_REGISTRY.register_processor(
    CrisperWhisperMultiModalProcessor,
    info=WhisperProcessingInfo,
    dummy_inputs=WhisperDummyInputsBuilder,
)
class CrisperWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    # The blank-detection head is not used for decoding.
    hf_to_vllm_mapper = WhisperForConditionalGeneration.hf_to_vllm_mapper | (
        WeightsMapper(orig_to_new_prefix={"encoder_blank_head.": None})
    )

    # The model does not emit timestamp tokens.
    supports_segment_timestamp = False

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        """Build the decoder prompt of the official CrisperWhisper prompt builder.

        `vllm_xargs={"mode": "intended"}` selects the clean transcript; the
        default mode is `"verbatim"`.
        """
        if stt_params.task_type != "transcribe":
            raise ValueError("CrisperWhisper does not support translation")
        language = stt_params.language
        if language is None:
            raise ValueError(
                "Language must be specified when creating the CrisperWhisper prompt"
            )
        mode = (stt_params.extra_args or {}).get("mode", "verbatim")
        if mode not in _MODES:
            raise ValueError(
                f"Unsupported CrisperWhisper mode {mode!r}; "
                f"expected one of {list(_MODES)}."
            )
        if stt_params.request_prompt or stt_params.hotwords:
            logger.warning_once(
                "CrisperWhisper ignores the prompt and hotwords of "
                "transcription requests."
            )

        mode_tags = "".join(f"[{mode}_{i}]" for i in range(1, _NUM_MODE_TAGS + 1))
        decoder_text = (
            f"{mode_tags}<|startoftranscript|><|{language}|>"
            "<|transcribe|><|notimestamps|>"
        )
        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={
                    "audio": (stt_params.audio, stt_params.stt_config.sample_rate)
                },
            ),
            decoder_prompt=TextPrompt(prompt=decoder_text),
        )
