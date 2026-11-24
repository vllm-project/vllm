# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 scored-prefill warmup on the NVIDIA path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.models.step5.nvidia import warmup
from vllm.models.step5.nvidia.sparse_attention import Step5DSAAttentionImpl

WORKER = SimpleNamespace(device="cuda")


def _layer(adapter=None, dsa=True):
    impl = MagicMock(
        adapter=adapter, sparse_topk=512, max_num_regions=4096, indexer_heads=4
    )
    if dsa:
        impl.__class__ = Step5DSAAttentionImpl
    return SimpleNamespace(attn=SimpleNamespace(impl=impl))


@pytest.mark.parametrize("ran", [True, False])
def test_scores_prefill_only_after_warmup_step(ran):
    adapter = MagicMock()
    with patch.object(
        warmup, "run_step5_scored_prefill_warmup", return_value=ran
    ) as run:
        warmup.step5_dsa_warmup([_layer(dsa=False), _layer(adapter)], WORKER)
    run.assert_called_once_with(WORKER, 512)
    assert adapter.warmup_prefill_scoring.called is ran
    if ran:
        adapter.warmup_prefill_scoring.assert_called_once_with(4096, 512, 4, "cuda")


@pytest.mark.parametrize("layer", [_layer(MagicMock(), dsa=False), _layer(None)])
def test_no_dsa_adapter_is_a_noop(layer):
    with patch.object(warmup, "run_step5_scored_prefill_warmup") as run:
        warmup.step5_dsa_warmup([layer], WORKER)
    run.assert_not_called()
