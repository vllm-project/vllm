# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Weave integration for vLLM.

Keep this package import lightweight.

Some environments import parts of vLLM without having the compiled CUDA
extension available/compatible (e.g. when only using CPU utilities). Eagerly
importing connector code can trigger platform resolution and attempt to load
`vllm._C`, which can fail even when users only need pure-Python pieces.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "WeaveConnector",
    "WeaveConnectorMetadata",
    "WeaveConnectorScheduler",
    "WeaveConnectorWorker",
    "WeaveOffloadingSpec",
]


def __getattr__(name: str):
    if name == "WeaveConnector":
        return import_module(".connector", __name__).WeaveConnector
    if name == "WeaveConnectorMetadata":
        return import_module(".metadata", __name__).WeaveConnectorMetadata
    if name == "WeaveConnectorScheduler":
        return import_module(".scheduler", __name__).WeaveConnectorScheduler
    if name == "WeaveConnectorWorker":
        return import_module(".worker", __name__).WeaveConnectorWorker
    if name == "WeaveOffloadingSpec":
        return import_module(".offloading_spec", __name__).WeaveOffloadingSpec
    raise AttributeError(name)


if TYPE_CHECKING:
    from .connector import WeaveConnector as WeaveConnector
    from .metadata import WeaveConnectorMetadata as WeaveConnectorMetadata
    from .offloading_spec import WeaveOffloadingSpec as WeaveOffloadingSpec
    from .scheduler import WeaveConnectorScheduler as WeaveConnectorScheduler
    from .worker import WeaveConnectorWorker as WeaveConnectorWorker
