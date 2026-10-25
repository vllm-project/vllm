# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

from vllm.utils import nccl


@pytest.mark.parametrize("has_no_cache", [True, False])
def test_scoped_nccl_env_needs_no_cache(monkeypatch, has_no_cache):
    # Without NCCL_NO_CACHE, NCCL keeps the first value it reads, so the scope
    # must not set it at all (a joining rank would disagree with its peers).
    monkeypatch.setattr(nccl, "_nccl_has_no_cache", lambda: has_no_cache)
    monkeypatch.delenv("NCCL_RUNTIME_CONNECT", raising=False)
    with nccl.scoped_nccl_env({"NCCL_RUNTIME_CONNECT": "0"}):
        inside = os.environ.get("NCCL_RUNTIME_CONNECT")
    assert inside == ("0" if has_no_cache else None)
    assert "NCCL_RUNTIME_CONNECT" not in os.environ
