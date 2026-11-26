# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration checks for the Drex read strategy.

The strategy itself imports the engine config, which the local unit-test
interpreter cannot load, so this file skips unless the full package imports.
The end-to-end read is exercised on the GPU box against the checkpoint.
"""

import pytest

strategies = pytest.importorskip("vllm.entrypoints.openai.decisions.strategies")


def test_drex_architecture_selects_the_drex_strategy():
    assert strategies.READ_STRATEGIES["Qwen3_5ForDrex"] is strategies.DrexStrategy


def test_drex_strategy_reads_as_a_classify_task():
    assert strategies.DrexStrategy.task == "classify"
    assert strategies.DrexStrategy.MAX_OPTIONS == 255
    assert strategies.DrexStrategy.MAX_QUESTIONS == 64
