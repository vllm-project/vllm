# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from lmcache.integration.vllm.vllm_ec_adapter import LMCacheECConnectorImpl

if TYPE_CHECKING:
    from vllm.v1.request import Request


class LMCacheECConnector(ECConnectorBase):
    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._impl = LMCacheECConnectorImpl(
            vllm_config=vllm_config,
            role=role,
            parent=self,
        )

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        return self._impl.start_load_caches(encoder_cache, **kwargs)

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:
        return self._impl.save_caches(encoder_cache, mm_hash, **kwargs)

    def has_cache_item(self, identifier: str) -> bool:
        return self._impl.has_cache_item(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        return self._impl.update_state_after_alloc(request, index)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        return self._impl.build_connector_meta(scheduler_output)
