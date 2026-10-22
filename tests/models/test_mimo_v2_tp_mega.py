# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.mimo_v2_mega_moe import MiMoV2TPMegaMoEMethod


def test_mimo_tp_mega_selects_padded_mega_or_fallback():
    method = object.__new__(MiMoV2TPMegaMoEMethod)
    method.capacity_rows = 8192
    method.max_rows = 256

    assert method.execution_path(0) == "empty"
    assert method.execution_path(1) == "mega_tp_ar"
    assert method.execution_path(256) == "mega_tp_ar"
    assert method.execution_path(257) == "aiter_a8w4"
    assert method.execution_path(8192) == "aiter_a8w4"
    with pytest.raises(ValueError, match="outside configured padded capacity"):
        method.execution_path(8193)
