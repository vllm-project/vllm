# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory


def _cfg(*, nixl_enabled=False, use_v2_model_runner=True):
    ec = MagicMock()
    ec.is_ec_producer = True
    ec.is_ec_consumer = True
    ec.get_from_extra_config.return_value = nixl_enabled
    cfg = MagicMock()
    cfg.ec_transfer_config = ec
    cfg.use_v2_model_runner = use_v2_model_runner
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


@pytest.mark.parametrize("role", [ECConnectorRole.SCHEDULER, ECConnectorRole.WORKER])
@pytest.mark.parametrize("nixl_enabled", [False, "false", "0", "no"])
def test_v1_model_runner_is_allowed_without_nixl(monkeypatch, role, nixl_enabled):
    monkeypatch.setattr(
        ECCPUConnector, "_make_scheduler", lambda self, cfg: MagicMock()
    )
    monkeypatch.setattr(ECCPUConnector, "_make_worker", lambda self, cfg: MagicMock())
    cfg = _cfg(
        nixl_enabled=nixl_enabled,
        use_v2_model_runner=False,
    )

    ECCPUConnector(cfg, role)


@pytest.mark.parametrize("role", [ECConnectorRole.SCHEDULER, ECConnectorRole.WORKER])
@pytest.mark.parametrize("nixl_enabled", [True, "true", "1", "yes"])
def test_v1_model_runner_is_rejected_with_nixl(monkeypatch, role, nixl_enabled):
    monkeypatch.setattr(
        ECCPUConnector, "_make_scheduler", lambda self, cfg: MagicMock()
    )
    monkeypatch.setattr(ECCPUConnector, "_make_worker", lambda self, cfg: MagicMock())
    cfg = _cfg(
        nixl_enabled=nixl_enabled,
        use_v2_model_runner=False,
    )

    with pytest.raises(ValueError, match="with NIXL requires the V2 model runner"):
        ECCPUConnector(cfg, role)


@pytest.mark.parametrize("role", [ECConnectorRole.SCHEDULER, ECConnectorRole.WORKER])
def test_v2_model_runner_is_allowed_with_nixl(monkeypatch, role):
    monkeypatch.setattr(
        ECCPUConnector, "_make_scheduler", lambda self, cfg: MagicMock()
    )
    monkeypatch.setattr(ECCPUConnector, "_make_worker", lambda self, cfg: MagicMock())

    ECCPUConnector(_cfg(nixl_enabled=True), role)


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
