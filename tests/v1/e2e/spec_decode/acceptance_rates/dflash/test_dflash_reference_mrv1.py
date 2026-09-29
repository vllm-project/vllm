# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ._dflash import run_dflash_reference_acceptance_lengths


def test_dflash_reference_acceptance_lengths_mrv1(
    monkeypatch: pytest.MonkeyPatch, vllm_runner
):
    run_dflash_reference_acceptance_lengths(monkeypatch, False, vllm_runner)
