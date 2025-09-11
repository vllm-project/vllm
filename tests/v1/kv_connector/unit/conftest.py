# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture(autouse=True)
def _mock_p2p_nccl_engine_send_sync(monkeypatch):
    """Global mock to prevent real send_sync creation during tests.

    - Mock `P2pNcclEngine.send_sync` to a no-op.
    - Provide dummy `zmq.Context` and `zmq.Poller` to virtual-bind without
      creating real ZMQ resources.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (  # noqa: E501
        P2pNcclEngine)

    def _fake_send_sync(self, item=None):
        return True

    monkeypatch.setattr(P2pNcclEngine,
                        "send_sync",
                        _fake_send_sync,
                        raising=True)

    # Dummy ZMQ context/socket/poller so that bind/register are no-ops.
    try:
        import zmq  # type: ignore

        class _DummySocket:

            def bind(self, *args, **kwargs):
                pass

            def connect(self, *args, **kwargs):
                pass

            def send_json(self, *args, **kwargs):
                pass

            def recv_json(self, *args, **kwargs):
                return {}

            def send_multipart(self, *args, **kwargs):
                pass

            def recv_multipart(self, *args, **kwargs):
                return [b"", b""]

            def close(self):
                pass

        class _DummyContext:

            def socket(self, *args, **kwargs):
                return _DummySocket()

            def term(self):
                pass

        class _DummyPoller:

            def register(self, *args, **kwargs):
                pass

            def poll(self, *args, **kwargs):
                return []

        monkeypatch.setattr(zmq, "Context", lambda *a, **k: _DummyContext())
        monkeypatch.setattr(zmq, "Poller", lambda *a, **k: _DummyPoller())
    except Exception:
        # If zmq is unavailable, safely ignore.
        pass
