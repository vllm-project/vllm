# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store key helpers for the hidden-state Mooncake connector."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
        HiddenPoolKey,
    )


def escape_key_part(value: str) -> str:
    """Escape one key component while keeping simple values readable."""
    return quote(str(value), safe="-_.~")


def make_hidden_data_key(pool_key: "HiddenPoolKey") -> str:
    return pool_key.to_string()
