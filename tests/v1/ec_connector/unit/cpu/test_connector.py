# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory


def _cfg():
    ec = MagicMock()
    ec.is_ec_producer = True
    ec.is_ec_consumer = True
    cfg = MagicMock()
    cfg.ec_transfer_config = ec
    return cfg


def test_scheduler_role_builds_only_scheduler(monkeypatch):
    fake_sched = MagicMock()
    monkeypatch.setattr(ECCPUConnector, "_make_scheduler", lambda self, cfg: fake_sched)
    c = ECCPUConnector(_cfg(), ECConnectorRole.SCHEDULER)
    assert c.connector_scheduler is fake_sched
    assert c.connector_worker is None
    c.has_cache_item("x")
    fake_sched.has_cache_item.assert_called_once_with("x")


def test_worker_role_builds_only_worker(monkeypatch):
    fake_worker = MagicMock()
    monkeypatch.setattr(ECCPUConnector, "_make_worker", lambda self, cfg: fake_worker)
    c = ECCPUConnector(_cfg(), ECConnectorRole.WORKER)
    assert c.connector_worker is fake_worker
    assert c.connector_scheduler is None


def test_factory_registered():
    cls = ECConnectorFactory._registry["ECCPUConnector"]()
    assert cls is ECCPUConnector


def test_request_finished_forwards_to_scheduler(monkeypatch):
    from unittest.mock import MagicMock

    import vllm.distributed.ec_transfer.ec_connector.cpu.connector as conn_mod  # noqa: F401
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
    from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector

    fake_sched = MagicMock()
    fake_sched.request_finished.return_value = (False, {"h1": {"peer_port": 1}})
    monkeypatch.setattr(ECCPUConnector, "_make_scheduler", lambda self, cfg: fake_sched)
    c = ECCPUConnector(_cfg(), ECConnectorRole.SCHEDULER)
    req = MagicMock()
    assert c.request_finished(req) == (False, {"h1": {"peer_port": 1}})
    fake_sched.request_finished.assert_called_once_with(req)
