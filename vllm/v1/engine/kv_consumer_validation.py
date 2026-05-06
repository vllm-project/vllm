# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import KVTransferConfig

_NIXL_REMOTE_PREFILL_REQUIRED_PARAMS = (
    "remote_engine_id",
    "remote_request_id",
    "remote_host",
    "remote_port",
)


class KVConsumerRequestError(ValueError):
    """Raised when a kv_consumer request would fall back to local prefill."""


def _kv_config_uses_nixl(kv_transfer_config: KVTransferConfig) -> bool:
    if kv_transfer_config.kv_connector == "NixlConnector":
        return True

    connector_configs = kv_transfer_config.kv_connector_extra_config.get(
        "connectors", []
    )
    return any(
        isinstance(conn_config, dict)
        and conn_config.get("kv_connector") == "NixlConnector"
        for conn_config in connector_configs
    )


def _missing_kv_param(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _get_kv_transfer_params(request: Any) -> Any:
    if hasattr(request, "kv_transfer_params"):
        return request.kv_transfer_params

    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is None or sampling_params.extra_args is None:
        return None
    return sampling_params.extra_args.get("kv_transfer_params")


def validate_kv_consumer_request(
    request: Any, kv_transfer_config: KVTransferConfig | None
) -> None:
    if (
        kv_transfer_config is None
        or kv_transfer_config.kv_role != "kv_consumer"
        or request.pooling_params is not None
    ):
        return

    params = _get_kv_transfer_params(request)
    if not isinstance(params, dict) or not params:
        raise KVConsumerRequestError(
            "kv_consumer instance requires non-empty kv_transfer_params "
            f"for request {request.request_id!r}; refusing local prefill."
        )

    if params.get("do_remote_prefill") is not True:
        raise KVConsumerRequestError(
            "kv_consumer instance requires "
            "kv_transfer_params.do_remote_prefill=true "
            f"for request {request.request_id!r}; refusing local prefill."
        )

    if params.get("do_remote_decode") is True:
        raise KVConsumerRequestError(
            "kv_consumer instance received prefill-side "
            f"kv_transfer_params for request {request.request_id!r}; "
            "refusing local prefill."
        )

    if not _kv_config_uses_nixl(kv_transfer_config):
        return

    missing = [
        param
        for param in _NIXL_REMOTE_PREFILL_REQUIRED_PARAMS
        if _missing_kv_param(params.get(param))
    ]
    if missing:
        raise KVConsumerRequestError(
            "NixlConnector kv_consumer instance requires remote prefill "
            f"metadata {missing} for request {request.request_id!r}; "
            "refusing local prefill."
        )

    if "remote_block_ids" not in params or params.get("remote_block_ids") is None:
        raise KVConsumerRequestError(
            "NixlConnector kv_consumer instance requires "
            f"kv_transfer_params.remote_block_ids for request "
            f"{request.request_id!r}; refusing local prefill."
        )
