# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from ._basic_correctness import (
    BGE_MULTILINGUAL_GEMMA2_SETTING,
    GRANITE_SETTING,
    run_compile_correctness,
)


def test_compile_correctness_granite():
    run_compile_correctness(GRANITE_SETTING, "eager")


def test_compile_correctness_bge_multilingual_gemma2():
    run_compile_correctness(BGE_MULTILINGUAL_GEMMA2_SETTING, "eager")
