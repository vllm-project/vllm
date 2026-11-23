# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL P/D accuracy with AITER unified attention on ROCm."""

import pytest

from tests.utils import spawn_new_process_for_each_test
from tests.v1.kv_connector.rocm_pd_accuracy_utils import (
    DEEPSEEK_MODEL,
    QWEN_MODEL,
    run_rocm_pd_accuracy,
)
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm P/D equivalents"),
    pytest.mark.distributed(num_gpus=4),
]


@pytest.mark.parametrize(
    "model,prefill_tp,decode_tp",
    [
        pytest.param(QWEN_MODEL, 2, 2, id="nixl-aiter-qwen-p2-d2"),
        pytest.param(QWEN_MODEL, 1, 2, id="nixl-aiter-qwen-p1-d2"),
        pytest.param(QWEN_MODEL, 2, 1, id="nixl-aiter-qwen-p2-d1"),
        pytest.param(DEEPSEEK_MODEL, 1, 1, id="nixl-aiter-deepseek-p1-d1"),
        pytest.param(DEEPSEEK_MODEL, 1, 2, id="nixl-aiter-deepseek-p1-d2"),
        pytest.param(DEEPSEEK_MODEL, 2, 1, id="nixl-aiter-deepseek-p2-d1"),
    ],
)
@spawn_new_process_for_each_test
def test_rocm_pd_accuracy(model, prefill_tp, decode_tp, monkeypatch):
    run_rocm_pd_accuracy("nixl", model, prefill_tp, decode_tp, monkeypatch)
