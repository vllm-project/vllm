# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`/health_generate` against injected failures on a single GPU.

`/health` stays 200 for all of these; `/health_generate` must not.
"""

import pytest

from .utils import PROBE_TIMEOUT_S, ready_probe_server

MODEL = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server():
    with ready_probe_server(MODEL, []) as srv:
        yield srv


def test_no_false_positive(server):
    """A healthy engine stays ready when idle and under sustained load."""
    assert server.stays_ready(PROBE_TIMEOUT_S + 3)
    with server.busy():
        assert server.stays_ready(PROBE_TIMEOUT_S + 3)


@pytest.mark.parametrize("fault", ["fault_ima_now", "fault_ima_on_next_forward"])
def test_cuda_error_while_idle(fault: str):
    """An async CUDA fault that nothing has synchronized on yet, raised before
    or during the probe's own forward pass, fails that same `/health_generate` call."""
    with ready_probe_server(MODEL, []) as srv:
        assert srv.status() == 200
        srv.rpc(fault, "0")
        assert srv.status("/health") == 200
        assert srv.status() == 503
