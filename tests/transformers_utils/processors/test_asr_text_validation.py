# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.transformers_utils.processors.fireredasr2 import FireRedASR2Processor
from vllm.transformers_utils.processors.funasr import FunASRProcessor


@pytest.mark.parametrize("cls", [FireRedASR2Processor, FunASRProcessor])
@pytest.mark.parametrize("bad_text", [123, {"a": 1}, ["ok", 123]])
def test_processor_rejects_invalid_text(cls, bad_text):
    """`text` that is not a string or a list of strings must raise a clear
    ``ValueError`` rather than a ``TypeError``/``KeyError`` (non-list) or be
    silently accepted (list with non-string elements)."""
    processor = object.__new__(cls)
    processor._in_target_context_manager = False

    with pytest.raises(ValueError, match="Invalid input text"):
        processor(text=bad_text)
