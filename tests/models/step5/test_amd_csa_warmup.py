# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm CSA layers reach the Step5 scored-prefill warmup."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Step5 CSA is the ROCm path"
)


def test_step5_csa_warmup_uses_first_indexed_layer_topk():
    from vllm.models.step5.amd import warmup

    sparse = MagicMock(indexer=SimpleNamespace(topk=512))
    dense = MagicMock(indexer=None)
    worker = object()

    with patch.object(warmup, "run_step5_scored_prefill_warmup") as run:
        warmup.step5_csa_warmup([dense, sparse], worker)
    run.assert_called_once_with(worker, 512)

    with patch.object(warmup, "run_step5_scored_prefill_warmup") as run:
        warmup.step5_csa_warmup([dense], worker)
    run.assert_not_called()
