# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from dataclasses import field
from typing import Any, Literal, get_args

from vllm.config.utils import CompileFactors, config

IOSender = Literal["io_sender", "io_both"]
IOReceiver = Literal["io_receiver", "io_both"]
IORole = Literal[IOSender, IOReceiver]


@config
class IOTransferConfig:
    """Configuration for distributed IO transfer of IO data."""

    io_connector: str | None = None
    """The IO connector used to transmit IO data between
    vLLM instances (preprocessor -> worker -> postprocessor)."""

    io_role: IORole | None = None
    """Whether this vLLM instance is a sender, a receiver, or both.
    Choices are 'io_sender', 'io_receiver', 'io_both'."""

    engine_id: str | None = None
    """The engine id for IO transfers."""

    io_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """Any extra config that the IO connector may need."""

    io_connector_module_path: str | None = None
    """The Python module path to dynamically load a custom IO connector
    from. Used when io_connector is not a registered connector name."""

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.io_role is not None and self.io_role not in get_args(IORole):
            raise ValueError(
                f"Unsupported io_role: {self.io_role}. "
                f"Supported roles are {get_args(IORole)}"
            )

        if self.io_connector is not None and self.io_role is None:
            raise ValueError(
                "Please specify io_role when io_connector "
                f"is set, supported roles are {get_args(IORole)}"
            )

    @property
    def is_io_transfer_instance(self) -> bool:
        return self.io_connector is not None and self.io_role in get_args(IORole)

    @property
    def is_io_sender(self) -> bool:
        return self.io_connector is not None and self.io_role in get_args(IOSender)

    @property
    def is_io_receiver(self) -> bool:
        return self.io_connector is not None and self.io_role in get_args(IOReceiver)

    def get_from_extra_config(self, key: str, default: Any) -> Any:
        return self.io_connector_extra_config.get(key, default)

    def compile_factors(self) -> CompileFactors:
        """This config does not affect the compiled graph.

        If a field starts affecting it, return
        `get_compile_factors(self, ignored_factors)` with every other field in
        `ignored_factors`.
        """
        return {}
