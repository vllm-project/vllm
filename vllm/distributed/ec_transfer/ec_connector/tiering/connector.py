# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public connector shell for tiered encoder-cache offloading."""

from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.connector import (
    ECCPUConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ECTieringConnector(ECCPUConnector):
    """CPU-primary connector with optional asynchronous secondary tiers."""

    def _make_scheduler(self, vllm_config: VllmConfig):
        from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
            ECTieringScheduler,
        )

        return ECTieringScheduler(vllm_config)
