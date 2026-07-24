# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-detection contract for the dedicated FlashInfer XQA decode API.

These guard the graceful-degradation behavior on FlashInfer builds that do not
expose ``xqa_batch_decode_with_kv_cache`` (e.g. the stock pinned wheel): the
probes must report ``False`` and the lazy wrapper must raise a clear
upgrade-guidance error instead of an opaque ``AttributeError``.
"""

import pytest

from vllm.utils import flashinfer as fi


def test_xqa_ragged_q_requires_xqa_decode(monkeypatch) -> None:
    # Ragged-Q can never be available unless the base XQA decode path is.
    monkeypatch.setattr(fi, "has_flashinfer_xqa_decode", lambda: False)
    fi.has_flashinfer_xqa_ragged_q.cache_clear()
    try:
        assert fi.has_flashinfer_xqa_ragged_q() is False
    finally:
        fi.has_flashinfer_xqa_ragged_q.cache_clear()


def test_xqa_decode_probe_false_without_flashinfer(monkeypatch) -> None:
    monkeypatch.setattr(fi, "has_flashinfer", lambda: False)
    fi.has_flashinfer_xqa_decode.cache_clear()
    try:
        assert fi.has_flashinfer_xqa_decode() is False
    finally:
        fi.has_flashinfer_xqa_decode.cache_clear()


def test_xqa_wrapper_raises_helpful_error_when_absent(monkeypatch) -> None:
    # Force the lazy wrapper's cached implementation to be missing.
    monkeypatch.setattr(fi, "has_flashinfer", lambda: False)
    wrapper = fi._lazy_import_wrapper(
        "flashinfer.decode",
        "xqa_batch_decode_with_kv_cache",
        fallback_fn=fi._missing_xqa,
    )
    with pytest.raises(RuntimeError, match="xqa_batch_decode_with_kv_cache"):
        wrapper()
