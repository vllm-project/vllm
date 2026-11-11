# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake P/D accuracy with same-host HIP transport on ROCm."""

import pytest

from tests.utils import spawn_new_process_for_each_test
from tests.v1.kv_connector.rocm_pd_accuracy_utils import (
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
        pytest.param(QWEN_MODEL, 1, 1, id="mooncake-hip-qwen-p1-d1"),
        pytest.param(QWEN_MODEL, 2, 2, id="mooncake-hip-qwen-p2-d2"),
    ],
)
@spawn_new_process_for_each_test
def test_rocm_pd_accuracy(model, prefill_tp, decode_tp, monkeypatch):
    run_rocm_pd_accuracy("mooncake", model, prefill_tp, decode_tp, monkeypatch)
