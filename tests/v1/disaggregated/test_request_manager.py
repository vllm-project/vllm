# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest

from vllm.distributed.disaggregated.factory import (
    DisaggregatedRequestManagerFactory)
from vllm.distributed.disaggregated.request_manager import (
    DisaggregatedRequestManager)


@pytest.fixture(autouse=True)
def _isolate_factory_registry(monkeypatch):
    original = DisaggregatedRequestManagerFactory._registry.copy()
    monkeypatch.setattr(DisaggregatedRequestManagerFactory,
                        "_registry", {},
                        raising=False)
    try:
        yield
    finally:
        DisaggregatedRequestManagerFactory._registry = original


def _dummy_config(enabled: bool):
    cfg = types.SimpleNamespace()
    cfg.kv_transfer_config = object() if enabled else None
    return cfg


def test_register_request_manager():
    # Fresh registry
    assert DisaggregatedRequestManagerFactory._registry == {}

    @DisaggregatedRequestManagerFactory.register("low")
    class _DummyManagerLow(DisaggregatedRequestManager):
        priority = 1

        def dispatch_request(self, request, shared_http_clients):
            return False, None

    class _DummyManagerHigh(DisaggregatedRequestManager):
        priority = 0

        def dispatch_request(self, request, shared_http_clients):
            return False, None

    # Register the two managers
    DisaggregatedRequestManagerFactory.register_request_manager(
        "high", lambda cfg: _DummyManagerHigh(cfg))

    assert set(DisaggregatedRequestManagerFactory._registry.keys()) == {
        "low", "high"
    }

    # Duplicate registration should fail
    with pytest.raises(ValueError):
        DisaggregatedRequestManagerFactory.register_request_manager(
            "low", lambda cfg: _DummyManagerLow(cfg))

    # Creation should order by ascending priority (high comes before low)
    cfg = _dummy_config(enabled=True)
    managers = DisaggregatedRequestManagerFactory.create_request_managers(cfg)
    assert isinstance(managers[0], _DummyManagerHigh)
    assert isinstance(managers[1], _DummyManagerLow)
