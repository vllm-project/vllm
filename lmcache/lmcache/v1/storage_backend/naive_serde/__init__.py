# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Tuple

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.naive_serde.cachegen_decoder import CacheGenDeserializer
from lmcache.v1.storage_backend.naive_serde.cachegen_encoder import CacheGenSerializer
from lmcache.v1.storage_backend.naive_serde.kivi_serde import (
    KIVIDeserializer,
    KIVISerializer,
)
from lmcache.v1.storage_backend.naive_serde.naive_serde import (
    NaiveDeserializer,
    NaiveSerializer,
)
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


def CreateSerde(
    serde_type: str,
    metadata: LMCacheMetadata,
    config: LMCacheEngineConfig,
) -> Tuple[Serializer, Deserializer]:
    s: Optional[Serializer] = None
    d: Optional[Deserializer] = None

    if serde_type == "naive":
        s, d = NaiveSerializer(), NaiveDeserializer()
    elif serde_type == "kivi":
        s, d = KIVISerializer(), KIVIDeserializer()
    elif serde_type == "cachegen":
        s, d = (
            CacheGenSerializer(config, metadata),
            CacheGenDeserializer(config, metadata),
        )
    else:
        raise ValueError(f"Invalid type: {serde_type}")

    return s, d


__all__ = [
    "Serializer",
    "Deserializer",
    "KIVISerializer",
    "KIVIDeserializer",
    "CreateSerde",
]
