# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public connector shell for tiered encoder-cache offloading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.distributed.ec_transfer.ec_connector.cpu.connector import (
    ECCPUConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.metrics import (
        ECConnectorPromMetrics,
        ECConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.v1.request import Request


class ECTieringConnector(ECCPUConnector):
    """CPU-primary connector with optional asynchronous secondary tiers."""

    def on_new_request(self, request: Request) -> None:
        assert self.connector_scheduler is not None
        callback = getattr(self.connector_scheduler, "on_new_request", None)
        if callback is not None:
            callback(request)

    def _make_scheduler(self, vllm_config: VllmConfig):
        from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
            ECTieringScheduler,
        )

        return ECTieringScheduler(vllm_config)

    def _make_worker(self, vllm_config: VllmConfig):
        from vllm.distributed.ec_transfer.ec_connector.tiering.worker import (
            ECTieringWorker,
        )

        return ECTieringWorker(vllm_config)

    def get_ec_connector_stats(self) -> ECConnectorStats | None:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.get_stats()
        if self.connector_worker is not None:
            return self.connector_worker.get_stats()
        return None

    @classmethod
    def build_ec_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> ECConnectorStats:
        from vllm.distributed.ec_transfer.ec_connector.tiering.metrics import (
            ECTieringStats,
        )

        return ECTieringStats(data=data) if data is not None else ECTieringStats()

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> ECConnectorPromMetrics:
        from vllm.distributed.ec_transfer.ec_connector.tiering.metrics import (
            ECTieringPromMetrics,
        )

        return ECTieringPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )
