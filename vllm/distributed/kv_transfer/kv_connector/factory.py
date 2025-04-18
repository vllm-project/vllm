# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from typing import TYPE_CHECKING, Callable, Dict, Type

from .base import KVConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class KVConnectorFactory:
    _registry: Dict[str, Callable[[], Type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> Type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, rank: int, local_rank: int,
                         config: "VllmConfig", world_group) -> KVConnectorBase:
        connector_name = config.kv_transfer_config.kv_connector
        if connector_name not in cls._registry:
            raise ValueError(f"Unsupported connector type: {connector_name}")

        connector_cls = cls._registry[connector_name]()
        return connector_cls(rank, local_rank, config, world_group)


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.
KVConnectorFactory.register_connector(
    "PyNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "vllm.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnector",
    "vllm.distributed.kv_transfer.kv_connector.lmcache_connector",
    "LMCacheConnector")

KVConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    "vllm.distributed.kv_transfer.kv_connector.mooncake_store_connector",
    "MooncakeStoreConnector")