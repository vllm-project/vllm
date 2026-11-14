# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public connector shell for tiered encoder-cache offloading."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.connector import (
    ECCPUConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


class ECTieringConnector(ECCPUConnector):
    """CPU-primary connector with optional asynchronous secondary tiers."""

    _encoder_cache_lookup: Callable[[Request, int], int | None]

    def on_new_request(self, request: Request) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def bind_encoder_cache_lookup(
        self, lookup: Callable[[Request, int], int | None]
    ) -> None:
        self._encoder_cache_lookup = lookup

    def ensure_cache_available(
        self, request: Request, num_computed_tokens: int
    ) -> bool:
        assert self.connector_scheduler is not None
        local_cache_hashes = {
            feature.identifier
            for i, feature in enumerate(request.mm_features)
            if self._encoder_cache_lookup(request, i)
            == request.get_num_encoder_embeds(i)
        }
        return self.connector_scheduler.ensure_cache_available(
            request, num_computed_tokens, local_cache_hashes
        )

    def _make_scheduler(self, vllm_config: VllmConfig):
        from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
            ECTieringScheduler,
        )

        self._encoder_cache_lookup = lambda request, input_id: None
        return ECTieringScheduler(vllm_config)
