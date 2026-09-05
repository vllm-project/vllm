# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory


class DummyECTransferConfig:
    def __init__(self, ec_connector: str):
        self.ec_connector = ec_connector


def test_mooncake_store_ec_connector_is_registered():
    from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding import (
        MooncakeStoreECConnector,
    )

    config = DummyECTransferConfig(ec_connector="MooncakeStoreECConnector")

    assert ECConnectorFactory.get_connector_class(config) is MooncakeStoreECConnector
