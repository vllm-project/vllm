# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`/health_generate` against injected failures on a single GPU.

`/health` stays 200 for all of these; `/health_generate` must not.
"""

import pytest

from vllm.platforms import current_platform

from .utils import PROBE_TIMEOUT_S, ready_probe_server

# The fault injection launches CUDA kernels.
pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

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
    or during the probe's own forward pass, fails the next probe."""
    with ready_probe_server(MODEL, []) as srv:
        assert srv.status() == 200
        srv.rpc(fault, "0")
        assert srv.status("/health") == 200
        assert srv.wait_until_unready() is not None
