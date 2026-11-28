# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only parts of the TP sampling check that the 2-GPU test
(tests/distributed/test_tp_sampler_consistency.py) does not reach."""

import pytest

from vllm.v1.worker.gpu.sample import sampler as sampler_mod
from vllm.v1.worker.gpu.sample import tp_consistency


def test_no_tp_group_is_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tp_consistency, "_tp_cpu_group", lambda: None)
    supported = {"FlashInfer": True, "XPU": False}
    assert tp_consistency.agree_on_sampling({"seed": 0}, supported) == supported


def test_ineligible_sampler_is_not_queried():
    def not_asked() -> bool:
        raise AssertionError("backend queried for an ineligible sampler")

    assert sampler_mod._local_support(False, not_asked) == (False, None)


def test_local_error_is_returned_not_raised():
    # The rank must still take part in the collective before failing.
    def unavailable_but_requested() -> bool:
        raise RuntimeError("FlashInfer top-p/top-k sampling unavailable")

    local, error = sampler_mod._local_support(True, unavailable_but_requested)
    assert local is False and isinstance(error, RuntimeError)
