# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .base import (
    IUMBPRuntime,
    UMBPRuntimeCapabilities,
    UMBPSchedulerHandle,
    UMBPWorkerHandle,
)
from .embedded import EmbeddedRuntime
from .factory import UMBPRuntimeConfig, UMBPRuntimeFactory

UMBPRuntimeFactory.register("embedded", EmbeddedRuntime.from_config)

__all__ = [
    "IUMBPRuntime",
    "UMBPRuntimeCapabilities",
    "UMBPSchedulerHandle",
    "UMBPWorkerHandle",
    "UMBPRuntimeConfig",
    "UMBPRuntimeFactory",
    "EmbeddedRuntime",
]
