# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 scored-prefill warmup dispatch on the NVIDIA path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.model_executor.warmup import step5_dsa_warmup as warmup


def _worker(model_type: str = "step5"):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type=model_type)
            )
        ),
        device="cuda",
    )


def _impl(adapter):
    return SimpleNamespace(
        adapter=adapter, sparse_topk=512, max_num_regions=4096, indexer_heads=4
    )


@pytest.fixture(autouse=True)
def _nvidia():
    with patch.object(warmup.current_platform, "is_rocm", return_value=False):
        yield


def test_skips_other_models():
    with patch.object(warmup, "_step5_dsa_impl") as find:
        warmup.step5_dsa_warmup(_worker("deepseek_v4"))
    find.assert_not_called()


@pytest.mark.parametrize("ran", [True, False])
def test_scores_prefill_only_after_warmup_step(ran):
    adapter = MagicMock()
    worker = _worker()
    with (
        patch.object(warmup, "_step5_dsa_impl", return_value=_impl(adapter)),
        patch.object(
            warmup, "run_step5_scored_prefill_warmup", return_value=ran
        ) as run,
    ):
        warmup.step5_dsa_warmup(worker)
    run.assert_called_once_with(worker, 512)
    assert adapter.warmup_prefill_scoring.called is ran
    if ran:
        adapter.warmup_prefill_scoring.assert_called_once_with(4096, 512, 4, "cuda")


@pytest.mark.parametrize("impl", [None, _impl(None)])
def test_no_dsa_adapter_is_a_noop(impl):
    with (
        patch.object(warmup, "_step5_dsa_impl", return_value=impl),
        patch.object(warmup, "run_step5_scored_prefill_warmup") as run,
    ):
        warmup.step5_dsa_warmup(_worker())
    run.assert_not_called()
