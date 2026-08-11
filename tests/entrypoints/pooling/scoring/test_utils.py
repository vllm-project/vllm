# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.pooling.scoring.typing import ScoreInput
from vllm.entrypoints.pooling.scoring.utils import validate_score_input
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
