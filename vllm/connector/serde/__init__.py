# SPDX-License-Identifier: Apache-2.0

# inspired by LMCache
from typing import Optional

from vllm.connector.serde.safe_serde import SafeDeserializer, SafeSerializer
from vllm.connector.serde.serde import Deserializer, Serializer


def create_serde(serde_type: str) -> tuple[Serializer, Deserializer]:
    s: Optional[Serializer] = None
    d: Optional[Deserializer] = None

    if serde_type == "safe":
        s = SafeSerializer()
        d = SafeDeserializer()
    else:
        raise ValueError(f"Unknown serde type: {serde_type}")

    return s, d


__all__ = [
    "Serializer",
    "Deserializer",
    "SafeSerializer",
    "SafeDeserializer",
    "create_serde",
]
