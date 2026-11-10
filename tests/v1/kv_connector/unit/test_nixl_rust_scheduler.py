# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native NIXL scheduler half (`vllm._rust_core.RustNixlConnectorScheduler`)
must build the same metadata and `kv_transfer_params` as the Python connector,
driven by the Python scheduler through a randomized P/D workload."""

import pytest

pytest.importorskip("vllm._rust_core")

from tests.v1.kv_connector.unit.nixl_rust_diff import (  # noqa: E402
    KINDS,
    PickleStats,
    run_one,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kind", KINDS)
def test_native_nixl_scheduler_matches_python(kind: str, seed: int):
    stats = PickleStats()
    ok, msg = run_one(kind, seed, stats)
    assert ok, msg
    assert stats.compared > 0


def test_rejects_subclassed_connectors():
    from vllm._rust_core import RustNixlConnectorScheduler

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlPullConnector

    class Custom(NixlPullConnector):
        pass

    connector = Custom.__new__(Custom)
    with pytest.raises(NotImplementedError):
        RustNixlConnectorScheduler(connector)
