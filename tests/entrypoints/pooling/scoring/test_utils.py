# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.entrypoints.pooling.scoring.typing import ScoreInput
from vllm.entrypoints.pooling.scoring.utils import (
    truncate_text_to_tokens,
    validate_score_input,
)
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize(
    ("data_1", "data_2", "is_multimodal_model", "architecture", "message"),
    [
        pytest.param(
            {"content": []},
            "document",
            False,
            "TestModel",
            "MultiModalParam is not supported for TestModel",
            id="unsupported-multimodal-input",
        ),
        pytest.param(
            ["query 1", "query 2"],
            ["document"],
            False,
            "TestModel",
            "Input lengths must be either 1:1, 1:N or N:N",
            id="incompatible-input-lengths",
        ),
        pytest.param(
            [],
            ["document"],
            False,
            "TestModel",
            "At least one text element must be given",
            id="empty-first-input",
        ),
        pytest.param(
            ["query"],
            [],
            False,
            "TestModel",
            "At least one text_pair element must be given",
            id="empty-second-input",
        ),
    ],
)
def test_validate_score_input_rejects_invalid_inputs(
    data_1: ScoreInput | list[ScoreInput],
    data_2: ScoreInput | list[ScoreInput],
    is_multimodal_model: bool,
    architecture: str,
    message: str,
):
    with pytest.raises(VLLMValidationError) as exc_info:
        validate_score_input(
            data_1,
            data_2,
            is_multimodal_model=is_multimodal_model,
            architecture=architecture,
        )

    assert str(exc_info.value) == message
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_truncate_text_to_tokens_handles_shared_character_offsets():
    vocab = {
        token: i for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)

    truncated = truncate_text_to_tokens("aéx", tokenizer, max_tokens=2)

    assert truncated == "a"
    assert len(tokenizer(truncated, add_special_tokens=False)["input_ids"]) == 1
