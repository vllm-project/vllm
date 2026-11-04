# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Literal

import torch
from pydantic import ConfigDict, SkipValidation

from vllm.config.utils import CompileFactors, config, get_compile_factors

Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]


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

    def compile_factors(self) -> CompileFactors:
        """WARNING: Whenever a new field is added to this config, add it to
        `ignored_factors` if it does not affect the computation graph.
        """
        # Device/platform info is captured by the torch and vLLM factors.
        ignored_factors = {
            "device",
            "device_type",
        }
        return get_compile_factors(self, ignored_factors)

    def __post_init__(self):
        if self.device == "auto":
            # Automated device type detection
            from vllm.platforms import current_platform

            self.device_type = current_platform.device_type
            if not self.device_type:
                raise RuntimeError(
                    "Failed to infer device type, please set "
                    "the environment variable `VLLM_LOGGING_LEVEL=DEBUG` "
                    "to turn on verbose logging to help debug the issue."
                )
        else:
            # Device type is assigned explicitly
            if isinstance(self.device, str):
                self.device_type = self.device
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type

        # Some platforms require processing inputs on CPU.
        from vllm.platforms import current_platform

        if (
            current_platform.uses_host_device_handling()
            and self.device_type == current_platform.device_type
        ):
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)
