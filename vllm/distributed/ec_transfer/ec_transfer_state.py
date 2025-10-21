# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm import envs
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_EC_CONNECTOR_AGENT: ECConnectorBase | None = None


def get_ec_transfer() -> ECConnectorBase:
    assert _EC_CONNECTOR_AGENT is not None, "disaggregated EC cache is not initialized"
    return _EC_CONNECTOR_AGENT


def has_ec_transfer() -> bool:
    return _EC_CONNECTOR_AGENT is not None


def ensure_ec_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """
    Initialize EC cache connector.
    """

    global _EC_CONNECTOR_AGENT

    if vllm_config.ec_transfer_config is None:
        return

    if (
        vllm_config.ec_transfer_config.is_ec_transfer_instance
        and _EC_CONNECTOR_AGENT is None
    ):
        if envs.VLLM_USE_V1:
            _EC_CONNECTOR_AGENT = ECConnectorFactory.create_connector(
                config=vllm_config, role=ECConnectorRole.WORKER
            )
        else:
            raise ValueError("V0 is no longer supported")
