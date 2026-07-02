# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Force `spawn` for EngineCore: tests here start an in-process gRPC
server, which makes vLLM's default fork() of EngineCore inherit dirty
gRPC state and intermittently SIGSEGV.
"""

import pytest


@pytest.fixture(autouse=True)
def use_spawn_for_v1_tracing(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
