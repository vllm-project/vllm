# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Any, Literal

import torch
from pydantic import ConfigDict, SkipValidation

from vllm.config.device_diagnostics import (
    build_failed_device_type_message,
    environment_looks_like_rocm,
)
from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]


def _get_current_platform_device_type() -> str:
    from vllm.platforms import current_platform

    return current_platform.device_type


def _get_failed_device_type_message() -> str:
    return build_failed_device_type_message(
        getattr(torch.version, "cuda", None),
        getattr(torch.version, "hip", None),
        environment_looks_like_rocm=environment_looks_like_rocm(),
    )


@config(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    """Configuration for the device to use for vLLM execution."""

    device: SkipValidation[Device | torch.device | None] = "auto"
    """Device type for vLLM execution.
    This parameter is deprecated and will be
    removed in a future release.
    It will now be set automatically based
    on the current platform."""
    device_type: str = field(init=False)
    """Device type from the current platform. This is set in
    `__post_init__`."""

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
        # the device/platform information will be summarized
        # by torch/vllm automatically.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.device == "auto":
            # Automated device type detection
            self.device_type = _get_current_platform_device_type()
            if not self.device_type:
                raise RuntimeError(_get_failed_device_type_message())
        else:
            # Device type is assigned explicitly
            if isinstance(self.device, str):
                self.device_type = self.device
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type

        # Some device types require processing inputs on CPU
        if self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)
