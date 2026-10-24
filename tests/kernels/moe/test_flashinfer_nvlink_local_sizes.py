# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`get_local_sizes` has to tolerate a forward context without DP metadata.

`ForwardContext.dp_metadata` is None when expert parallelism runs without data
parallelism (for example TP=1, PCP=4, EP). Both FlashInfer NVLink all-to-all
prepare/finalize modules then fall back to the local token count, so their
`get_local_sizes` must report the absence instead of asserting.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    flashinfer_nvlink_one_sided,
    flashinfer_nvlink_two_sided,
)

MODULES = pytest.mark.parametrize(
    "module",
    [flashinfer_nvlink_one_sided, flashinfer_nvlink_two_sided],
    ids=["one_sided", "two_sided"],
)


@pytest.mark.cpu_test
@MODULES
def test_get_local_sizes_without_dp_metadata(module):
    context = SimpleNamespace(dp_metadata=None)
    with patch.object(module, "get_forward_context", lambda: context):
        assert module.get_local_sizes() is None


@pytest.mark.cpu_test
@MODULES
def test_get_local_sizes_with_dp_metadata(module):
    dp_metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [4, 7])
    context = SimpleNamespace(dp_metadata=dp_metadata)
    with patch.object(module, "get_forward_context", lambda: context):
        assert module.get_local_sizes() == [4, 7]
