# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorage,
    ExOffloadingStorageKVCacheConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.factory import (  # noqa: E501
    ExOffloadingStorageFactory,
)


def get_uri_scheme(uri: str) -> str:
    uri = uri.strip()
    if not uri:
        raise ValueError("empty URI string")

    if "://" not in uri:
        raise ValueError("invalid URI format")

    scheme, _ = uri.split("://", 1)
    if not scheme:
        raise ValueError("invalid URI format")

    return scheme.lower()


class ExOffloadingStorageManager:
    _local = threading.local()
    _lock = threading.Lock()

    @classmethod
    def _get_thread_cache(
        cls,
    ) -> dict[tuple[str, frozenset], ExOffloadingStorage]:
        if not hasattr(cls._local, "cache"):
            cls._local.cache = {}
        return cls._local.cache

    @classmethod
    def get_storage_by_uri(
        cls, uri: str, kvcache_config: ExOffloadingStorageKVCacheConfig
    ) -> tuple[ExOffloadingStorage, str]:
        cache = cls._get_thread_cache()

        storage_name = get_uri_scheme(uri)
        storage_cls = ExOffloadingStorageFactory.get_storage_class(storage_name)
        extra_config, filepath = storage_cls.parse_uri(uri)

        cache_key = (
            storage_name,
            frozenset(extra_config.items()) if extra_config else frozenset(),
        )

        with cls._lock:
            if cache_key in cache:
                return cache[cache_key], filepath

            storage = storage_cls(extra_config)
            storage.register_kvcache(kvcache_config)
            cache[cache_key] = storage

            return storage, filepath
