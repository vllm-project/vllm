# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory


class DummyECTransferConfig:
    def __init__(self, ec_connector: str, ec_connector_module_path: str | None = None):
        self.ec_connector = ec_connector
        self.ec_connector_module_path = ec_connector_module_path


class ExternalECConnector:
    pass


class RegisteredECConnector:
    pass


def test_ec_connector_module_path_takes_priority(monkeypatch):
    module = types.ModuleType("tests.fake_external_ec_connector")
    module.TestECConnector = ExternalECConnector
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        ECConnectorFactory._registry,
        "TestECConnector",
        lambda: RegisteredECConnector,
    )

    config = DummyECTransferConfig(
        ec_connector="TestECConnector",
        ec_connector_module_path=module.__name__,
    )

    assert ECConnectorFactory.get_connector_class(config) is ExternalECConnector


def test_mooncake_store_ec_connector_is_registered():
    from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden import (
        MooncakeStoreECConnector,
    )

    config = DummyECTransferConfig(ec_connector="MooncakeStoreECConnector")

    assert ECConnectorFactory.get_connector_class(config) is MooncakeStoreECConnector
