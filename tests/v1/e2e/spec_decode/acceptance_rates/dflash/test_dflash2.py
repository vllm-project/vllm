# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only

from ._dflash import QWEN3_8_DFLASH2_NVFP4, run_dflash_correctness


@single_gpu_only
def test_dflash2_correctness(monkeypatch: pytest.MonkeyPatch, vllm_runner):
    run_dflash_correctness(monkeypatch, QWEN3_8_DFLASH2_NVFP4, True, vllm_runner)
