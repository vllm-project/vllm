# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import uuid
from dataclasses import field
from enum import Enum
from typing import Any

from vllm.config.utils import config


class ECRole(str, Enum):
    EC_PRODUCER = "ec_producer"
    EC_CONSUMER = "ec_consumer"
    EC_BOTH = "ec_both"


@config
class ECTransferConfig:
    """Configuration for distributed EC cache transfer."""

    ec_connector: str | None = None
    """The EC connector for vLLM to transmit EC caches between vLLM instances.
    """

    engine_id: str | None = None
    """The engine id for EC transfers."""

    ec_buffer_device: str | None = "cuda"
    """The device used by ec connector to buffer the EC cache.
    Currently only support 'cuda'."""

    ec_buffer_size: float = 1e9
    """The buffer size for TorchDistributedConnector. Measured in number of
    bytes. Recommended value: 1e9 (about 1GB)."""

    ec_role: ECRole | None = None
    """Whether this vLLM instance produces, consumes EC cache, or both. Choices
    are 'ec_producer', 'ec_consumer', 'ec_both'."""

    ec_rank: int | None = None
    """The rank of this vLLM instance in the EC cache transfer. Typical value:
    0 for encoder, 1 for pd instance.
    Currently only 1P1D is supported."""

    ec_parallel_size: int = 1
    """The number of parallel instances for EC cache transfer. For
    PyNcclConnector, this should be 2."""

    ec_ip: str = "127.0.0.1"
    """The EC connector ip, used to build distributed connection."""

    ec_port: int = 14579
    """The EC connector port, used to build distributed connection."""

    ec_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    ec_connector_module_path: str | None = None
    """The Python module path to dynamically load the EC connector from.
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
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if isinstance(self.ec_role, str):
            try:
                self.ec_role = ECRole(self.ec_role)
            except ValueError as e:
                supported_roles = [r.value for r in ECRole]
                raise ValueError(
                    f"Unsupported ec_role: {self.ec_role}. "
                    f"Supported roles are {supported_roles}"
                ) from e

        if self.ec_role is not None and not isinstance(self.ec_role, ECRole):
            supported_roles = [r.value for r in ECRole]
            raise ValueError(
                f"Unsupported ec_role: {self.ec_role}. "
                f"Supported roles are {supported_roles}"
            )

        if self.ec_connector is not None and self.ec_role is None:
            supported_roles = [r.value for r in ECRole]
            raise ValueError(
                "Please specify ec_role when ec_connector "
                f"is set, supported roles are {supported_roles}"
            )

    @property
    def is_ec_transfer_instance(self) -> bool:
        return self.ec_connector is not None and self.ec_role is not None

    @property
    def is_ec_producer(self) -> bool:
        return self.ec_connector is not None and self.ec_role in (
            ECRole.EC_PRODUCER,
            ECRole.EC_BOTH,
        )

    @property
    def is_ec_consumer(self) -> bool:
        return self.ec_connector is not None and self.ec_role in (
            ECRole.EC_CONSUMER,
            ECRole.EC_BOTH,
        )

    def get_from_extra_config(self, key, default) -> Any:
        return self.ec_connector_extra_config.get(key, default)
