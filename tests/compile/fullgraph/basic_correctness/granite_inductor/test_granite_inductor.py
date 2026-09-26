# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .._basic_correctness import GRANITE_SETTING, run_compile_correctness


def test_compile_correctness():
    run_compile_correctness(GRANITE_SETTING, "inductor")
