# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public connector shell for tiered encoder-cache offloading."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.connector import (
    ECCPUConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


class ECTieringConnector(ECCPUConnector):
    """CPU-primary connector with optional asynchronous secondary tiers."""

    def on_new_request(self, request: Request) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def on_new_request_with_cache(
        self, request: Request, local_cache_hashes: Collection[str]
    ) -> None:
        from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
            ECTieringScheduler,
        )

        if isinstance(self.connector_scheduler, ECTieringScheduler):
            self.connector_scheduler.on_new_request(request, local_cache_hashes)
        else:
            self.on_new_request(request)

    def ensure_cache_available_with_cache(
        self,
        request: Request,
        num_computed_tokens: int,
        local_cache_hashes: Collection[str],
    ) -> bool:
        from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
            ECTieringScheduler,
        )

        if isinstance(self.connector_scheduler, ECTieringScheduler):
            return self.connector_scheduler.ensure_cache_available(
                request, num_computed_tokens, local_cache_hashes
            )
        return self.ensure_cache_available(request, num_computed_tokens)

    def _make_scheduler(self, vllm_config: VllmConfig):
        from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
            ECTieringScheduler,
        )

        return ECTieringScheduler(vllm_config)
