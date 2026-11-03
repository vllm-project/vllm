# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .base import IUMBPRuntime, UMBPSchedulerHandle, UMBPWorkerHandle
from .embedded import EmbeddedRuntime
from .factory import UMBPRuntimeConfig, UMBPRuntimeFactory
from .standalone import StandaloneRuntime

UMBPRuntimeFactory.register("embedded", EmbeddedRuntime.from_config)
UMBPRuntimeFactory.register("standalone", StandaloneRuntime.from_config)

__all__ = [
    "IUMBPRuntime",
    "UMBPSchedulerHandle",
    "UMBPWorkerHandle",
    "UMBPRuntimeConfig",
    "UMBPRuntimeFactory",
    "EmbeddedRuntime",
    "StandaloneRuntime",
]
