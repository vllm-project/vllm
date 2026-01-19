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
    "LoomConnector",
    "WeaveConnectorMetadata",
    "WeaveConnectorScheduler",
    "WeaveConnectorWorker",
    "LoomOffloadingSpec",
]


def __getattr__(name: str):
    if name == "LoomConnector":
        return import_module(".connector.connector", __name__).LoomConnector
    if name == "WeaveConnectorMetadata":
        return import_module(".connector.metadata", __name__).WeaveConnectorMetadata
    if name == "WeaveConnectorScheduler":
        return import_module(".connector.scheduler", __name__).WeaveConnectorScheduler
    if name == "WeaveConnectorWorker":
        return import_module(".connector.worker", __name__).WeaveConnectorWorker
    if name == "LoomOffloadingSpec":
        return import_module(".kv_offload.spec", __name__).LoomOffloadingSpec
    raise AttributeError(name)


if TYPE_CHECKING:
    from .connector.connector import LoomConnector as LoomConnector
    from .connector.metadata import WeaveConnectorMetadata as WeaveConnectorMetadata
    from .connector.scheduler import WeaveConnectorScheduler as WeaveConnectorScheduler
    from .connector.worker import WeaveConnectorWorker as WeaveConnectorWorker
    from .kv_offload.spec import LoomOffloadingSpec as LoomOffloadingSpec
