# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.deepseek_ocr import (
    MAX_NGRAM_SIZE,
    MAX_NGRAM_WINDOW_SIZE,
    NGramPerReqLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.skip_global_cleanup


def test_ngram_processor_rejects_attacker_sized_scan_parameters():
    with pytest.raises(ValueError, match="ngram_size"):
        NGramPerReqLogitsProcessor.validate_params(
            SamplingParams(extra_args={"ngram_size": MAX_NGRAM_SIZE + 1})
        )

    with pytest.raises(ValueError, match="window_size"):
        NGramPerReqLogitsProcessor.validate_params(
            SamplingParams(
                extra_args={
                    "ngram_size": 2,
                    "window_size": MAX_NGRAM_WINDOW_SIZE + 1,
                }
            )
        )


def test_unigram_processor_only_scans_the_bounded_window():
    processor = NoRepeatNGramLogitsProcessor(ngram_size=1, window_size=2)
    logits = torch.zeros(5)

    result = processor([0, 1, 2], logits)

    assert result[0] == 0
    assert result[1] == -float("inf")
    assert result[2] == -float("inf")
