# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.proxy.register import ProxyRegistrar

_EC_CONNECTOR_AGENT: ECConnectorBase | None = None
_EC_REGISTRAR: "ProxyRegistrar | None" = None


def get_ec_transfer() -> ECConnectorBase:
    assert _EC_CONNECTOR_AGENT is not None, "disaggregated EC cache is not initialized"
    return _EC_CONNECTOR_AGENT


def has_ec_transfer() -> bool:
    return _EC_CONNECTOR_AGENT is not None


def ensure_ec_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """
    Initialize EC cache connector.
    """

    global _EC_CONNECTOR_AGENT, _EC_REGISTRAR

    if vllm_config.ec_transfer_config is None:
        return

    if (
        vllm_config.ec_transfer_config.is_ec_transfer_instance
        and _EC_CONNECTOR_AGENT is None
    ):
        _EC_CONNECTOR_AGENT = ECConnectorFactory.create_connector(
            config=vllm_config, role=ECConnectorRole.WORKER
        )
        if vllm_config.ec_transfer_config.get_from_extra_config(
            "proxy_registry_addr", None
        ):
            from vllm.distributed.ec_transfer.proxy.register import (
                start_worker_registration,
            )

            _EC_REGISTRAR = start_worker_registration(vllm_config)


def ensure_ec_transfer_shutdown() -> None:
    global _EC_CONNECTOR_AGENT, _EC_REGISTRAR
    if _EC_REGISTRAR is not None:
        _EC_REGISTRAR.close()
        _EC_REGISTRAR = None
    if _EC_CONNECTOR_AGENT is not None:
        _EC_CONNECTOR_AGENT.shutdown()
        _EC_CONNECTOR_AGENT = None
