# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only

from ._dflash import (
    LAGUNA_DFLASH_NVFP4,
    QWEN3_DFLASH,
    DFlashCorrectnessConfig,
    run_dflash_correctness,
)


@single_gpu_only
@pytest.mark.parametrize(
    ("config", "use_mrv2"),
    [
        pytest.param(
            QWEN3_DFLASH,
            False,
            id="qwen3-mrv1",
        ),
        pytest.param(
            QWEN3_DFLASH,
            True,
            id="qwen3-mrv2",
        ),
        pytest.param(
            LAGUNA_DFLASH_NVFP4,
            True,
            id="laguna-nvfp4-mrv2",
        ),
    ],
)
def test_dflash_correctness(
    monkeypatch: pytest.MonkeyPatch,
    config: DFlashCorrectnessConfig,
    use_mrv2: bool,
    vllm_runner,
):
    run_dflash_correctness(monkeypatch, config, use_mrv2, vllm_runner)
