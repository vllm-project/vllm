# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .base import (
    IUMBPRuntime,
    UMBPRuntimeCapabilities,
    UMBPSchedulerHandle,
    UMBPWorkerHandle,
)
from .distributed import DistributedRuntime
from .embedded import EmbeddedRuntime
from .factory import UMBPRuntimeConfig, UMBPRuntimeFactory
from .standalone import StandaloneRuntime

UMBPRuntimeFactory.register("embedded", EmbeddedRuntime.from_config)
UMBPRuntimeFactory.register("standalone", StandaloneRuntime.from_config)
UMBPRuntimeFactory.register("distributed", DistributedRuntime.from_config)

__all__ = [
    "IUMBPRuntime",
    "UMBPRuntimeCapabilities",
    "UMBPSchedulerHandle",
    "UMBPWorkerHandle",
    "UMBPRuntimeConfig",
    "UMBPRuntimeFactory",
    "DistributedRuntime",
    "EmbeddedRuntime",
    "StandaloneRuntime",
]
