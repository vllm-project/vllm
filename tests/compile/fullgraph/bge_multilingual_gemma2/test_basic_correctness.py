# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from .._basic_correctness import (
    BGE_MULTILINGUAL_GEMMA2_SETTING,
    run_compile_correctness,
)


@pytest.mark.parametrize(
    "test_setting",
    [BGE_MULTILINGUAL_GEMMA2_SETTING],
    ids=["test_setting1"],
)
def test_compile_correctness(test_setting):
    run_compile_correctness(test_setting)
