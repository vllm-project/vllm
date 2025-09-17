# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import uuid
from dataclasses import field
from typing import Any, Literal, Optional, get_args

from pydantic.dataclasses import dataclass

from vllm.config.utils import config

KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]


@config
@dataclass
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    kv_connector: Optional[str] = None
    """The KV connector for vLLM to transmit KV caches between vLLM instances.
    """

    engine_id: Optional[str] = None
    """The engine id for KV transfers."""

    kv_buffer_device: Optional[str] = "cuda"
    """The device used by kv connector to buffer the KV cache.
    Currently only support 'cuda'."""

    kv_buffer_size: float = 1e9
    """The buffer size for TorchDistributedConnector. Measured in number of
    bytes. Recommended value: 1e9 (about 1GB)."""

    kv_role: Optional[KVRole] = None
    """Whether this vLLM instance produces, consumes KV cache, or both. Choices
    are 'kv_producer', 'kv_consumer', and 'kv_both'."""

    kv_rank: Optional[int] = None
    """The rank of this vLLM instance in the KV cache transfer. Typical value:
    0 for prefill instance, 1 for decode instance.
    Currently only 1P1D is supported."""

    kv_parallel_size: int = 1
    """The number of parallel instances for KV cache transfer. For
    P2pNcclConnector, this should be 2."""

    kv_ip: str = "127.0.0.1"
    """The KV connector ip, used to build distributed connection."""

    kv_port: int = 14579
    """The KV connector port, used to build distributed connection."""

    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    kv_connector_module_path: Optional[str] = None
    """The Python module path to dynamically load the KV connector from.
    Only supported in V1."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(f"Unsupported kv_role: {self.kv_role}. "
                             f"Supported roles are {get_args(KVRole)}")

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError("Please specify kv_disagg_role when kv_connector "
                             f"is set, supported roles are {get_args(KVRole)}")

    @property
    def is_kv_transfer_instance(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in get_args(KVRole)

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in get_args(KVProducer)

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in get_args(KVConsumer)

    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)
