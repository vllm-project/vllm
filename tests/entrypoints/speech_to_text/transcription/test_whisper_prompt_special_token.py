# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: the Whisper request prompt must be prefixed with the real
``<|startofprev|>`` special token.

``get_generation_prompt()`` used to build the prefix with ``<|prev|>``, which
is not in the Whisper vocabulary. The tokenizer split it into ordinary text
tokens (``< | pre v | >``) placed before ``<|startoftranscript|>``, corrupting
the decoder conditioning: any request carrying the OpenAI ``prompt`` field
degenerated into repetition loops, and on multi-chunk (>30 s) audio the
transcription collapsed to the final chunk's output only.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
from vllm.model_executor.models.whisper import WhisperForConditionalGeneration

# Any Whisper checkpoint works: the special-token table is shared across the
# family, so use the smallest one to keep the download negligible.
TOKENIZER_MODEL = "openai/whisper-tiny"


def _decoder_prompt(request_prompt: str) -> str:
    params = SpeechToTextParams(
        audio=np.zeros(16_000, dtype=np.float32),
        stt_config=SpeechToTextConfig(),
        model_config=MagicMock(),
        language="en",
        task_type="transcribe",
        request_prompt=request_prompt,
    )
    prompt = WhisperForConditionalGeneration.get_generation_prompt(params)
    return prompt["decoder_prompt"]["prompt"]


@pytest.fixture(scope="module")
def whisper_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def test_prompt_prefix_is_a_single_special_token(whisper_tokenizer) -> None:
    decoder_prompt = _decoder_prompt("style and vocabulary hint")

    prev_id = whisper_tokenizer.convert_tokens_to_ids("<|startofprev|>")
    assert prev_id is not None and prev_id != whisper_tokenizer.unk_token_id

    ids = whisper_tokenizer.encode(decoder_prompt, add_special_tokens=False)
    # The prefix must survive tokenization as one special token, not leak
    # literal "<", "|", ... text tokens in front of <|startoftranscript|>.
    assert ids[0] == prev_id


def test_prompt_prefix_token_exists_in_vocab(whisper_tokenizer) -> None:
    # Guards against regressing to a made-up token again: whatever prefix the
    # prompt builder emits must round-trip as exactly one vocabulary entry.
    decoder_prompt = _decoder_prompt("hint")
    prefix = decoder_prompt.split("<|startoftranscript|>")[0]
    prefix_token = prefix[: prefix.index(">") + 1]
    ids = whisper_tokenizer.encode(prefix_token, add_special_tokens=False)
    assert len(ids) == 1, (
        f"{prefix_token!r} is not a single Whisper vocab token; "
        f"it tokenizes to {len(ids)} tokens"
    )


def test_no_request_prompt_means_no_prefix() -> None:
    decoder_prompt = _decoder_prompt("")
    assert decoder_prompt.startswith("<|startoftranscript|>")
