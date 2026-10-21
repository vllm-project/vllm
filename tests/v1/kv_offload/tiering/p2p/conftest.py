# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1.kv_offload.tiering.p2p.session import client as client_module


@pytest.fixture
def lookup_clock(monkeypatch):
    """Advance monotonic time for P2P deadlines without sleeping."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock.now)
    return clock
