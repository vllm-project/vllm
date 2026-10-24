# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ConnexConnector: P/D disaggregation over Connex.

The implementation lives in the ``connex-ai`` package
(``connex.adapters.vllm.pd_connector``). It pulls KV blocks from the prefill
instance with one-sided reads (connex-rs over UCX with GPUDirect RDMA, or
Mooncake Transfer Engine) and keeps them on the prefill instance as a
renewable hold until the decode instance releases them, so a request whose
decode instance fails can be resumed on another decode instance without a
second prefill. See docs/features/connex_connector_usage.md.
"""

try:
    from connex.adapters.vllm.pd_connector import (
        ConnexConnector,
        ConnexConnectorMetadata,
    )
except ImportError as e:
    raise ImportError(
        "ConnexConnector requires the connex-ai package: "
        "pip install 'connex-ai[rs]' (connex-rs data plane) or "
        "'connex-ai[mooncake]' (Mooncake Transfer Engine)."
    ) from e

__all__ = ["ConnexConnector", "ConnexConnectorMetadata"]
