# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable, Mapping
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.engine import (
    TransferBackend,
    TransferEngineKind,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.te import (
    TransferEngineBackend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent import (
    TentTransferBackend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent_config import (
    TENT_METRICS_BASE_PORT,
)

# Mooncake reads these inside the Transfer Engine constructor.
_TENT_ENV_VARS = ("MC_USE_TENT", "MC_USE_TEV1")
_DISABLED_VALUES = ("0", "false", "off", "no")


def resolve_transfer_engine(
    extra_config: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> TransferEngineKind:
    """Picks the engine from the connector config.

    Mooncake's own environment switch is honoured, and a config that contradicts
    it is an error rather than a silent preference for one of them.
    """
    requested = extra_config.get("transfer_engine")
    env_requests_tent = _tent_requested(environ if environ is not None else os.environ)
    if requested is None:
        if env_requests_tent:
            return TransferEngineKind.TENT
        return TransferEngineKind.TE
    valid = [kind.value for kind in TransferEngineKind]
    if requested not in valid:
        raise RuntimeError(
            f"transfer_engine must be one of {valid}, got {requested!r}."
        )
    kind = TransferEngineKind(requested)
    if env_requests_tent and kind is not TransferEngineKind.TENT:
        raise RuntimeError(
            "MC_USE_TENT or MC_USE_TEV1 asks Mooncake for its TENT implementation "
            f"while transfer_engine is {requested!r}; unset the variable or set "
            "transfer_engine to 'tent'."
        )
    return kind


def create_transfer_backend(
    kind: TransferEngineKind,
    *,
    engine_cls: Callable[[], Any] | None,
    hostname: str,
    protocol: str,
    device_name: str,
    device_id: int,
    overrides: Mapping[str, Any] | None = None,
) -> TransferBackend:
    if kind is TransferEngineKind.TE:
        if engine_cls is None:
            raise RuntimeError(
                "Mooncake is not available on this worker, so transfer_engine="
                "'te' cannot run; install Mooncake or select "
                "transfer_engine='tent'."
            )
        return TransferEngineBackend(
            engine_cls=engine_cls,
            hostname=hostname,
            protocol=protocol,
            device_name=device_name,
        )
    return TentTransferBackend(
        hostname=hostname,
        protocol=protocol,
        device_name=device_name,
        metrics_port=TENT_METRICS_BASE_PORT + device_id,
        overrides=overrides,
    )


def _tent_requested(environ: Mapping[str, str]) -> bool:
    for name in _TENT_ENV_VARS:
        value = environ.get(name)
        if value is not None and value.lower() not in _DISABLED_VALUES:
            return True
    return False
