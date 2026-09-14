# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from .._basic_correctness import GRANITE_SETTING, run_compile_correctness


@pytest.mark.parametrize(
    "test_setting",
    [GRANITE_SETTING],
    ids=["test_setting0"],
)
def test_compile_correctness(test_setting):
    run_compile_correctness(test_setting)
