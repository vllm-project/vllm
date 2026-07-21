# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import Any

from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

logger = init_logger(__name__)


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            kv_bytes_per_block = config.worker_kv_bytes_per_block * world_size
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // world_size

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

        # ---- Compact layout setup ----
        # Scheduler authority is the transported per-group aggregate charge.
        # Worker transfer geometry is a separate worker-only field.
        self._compact_slice_accounting = config.compact_slice_accounting
        compact_group_payloads = tuple(
            group.compact_bytes_per_native_block_per_worker for group in config.groups
        )
        any_compact = any(payload is not None for payload in compact_group_payloads)
        all_compact = all(payload is not None for payload in compact_group_payloads)
        if any_compact and not all_compact:
            raise ValueError(
                "compact payload charges must be present for every offloading group"
            )
        self._enable_compact_layout = all_compact and bool(compact_group_payloads)

        self._compact_per_rank_budget: int | None = None
        self._compact_group_payload_map: dict[int, int] | None = None
        self._compact_policy_capacity: int | None = None

        if self._enable_compact_layout:
            cpu_bytes = int(cpu_bytes_to_use)
            if cpu_bytes % world_size != 0:
                raise ValueError(
                    f"cpu_bytes_to_use ({cpu_bytes}) must be divisible by "
                    f"world_size ({world_size}) for compact layout "
                    f"(remainder: {cpu_bytes % world_size})"
                )
            per_rank_budget = cpu_bytes // world_size

            payload_map: dict[int, int] = {}
            for group_idx, payload in enumerate(compact_group_payloads):
                assert payload is not None
                if payload <= 0:
                    raise ValueError(
                        f"compact group {group_idx} has non-positive "
                        f"compact_real_bytes_per_rank ({payload})"
                    )
                payload_map[group_idx] = payload

            expected_group_ids = list(range(len(config.groups)))
            if sorted(payload_map) != expected_group_ids:
                raise ValueError(
                    "compact group indices must be contiguous from zero: "
                    f"expected {expected_group_ids}, got {sorted(payload_map)}"
                )

            policy_capacity = per_rank_budget // min(payload_map.values())
            if policy_capacity <= 0:
                raise ValueError(
                    "compact per-rank budget cannot hold the smallest group payload"
                )

            self._compact_per_rank_budget = per_rank_budget
            self._compact_group_payload_map = payload_map
            self._compact_policy_capacity = policy_capacity
            logger.info(
                "Compact layout enabled: per_rank_budget=%d, "
                "policy_capacity=%d, world_size=%d, payload_map=%s",
                per_rank_budget,
                policy_capacity,
                world_size,
                payload_map,
            )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            if self._enable_compact_layout:
                assert self._compact_per_rank_budget is not None
                assert self._compact_policy_capacity is not None
                assert self._compact_group_payload_map is not None
                # Use a stable 64 KiB fixed-page target reduced by GCD, ensuring
                # the page size always evenly divides the per-rank budget regardless
                # of the model-dependent group payload (which may not divide the
                # budget evenly).  This matches the final product's fixed-page
                # scatter geometry without shared/bounded-tail features.
                compact_page_size = math.gcd(64 * 1024, self._compact_per_rank_budget)
                if compact_page_size <= 0:
                    raise ValueError(
                        "compact page size must be positive; "
                        f"got {compact_page_size} from "
                        f"gcd(65536, {self._compact_per_rank_budget})"
                    )
                self._manager = CPUOffloadingManager(
                    num_blocks=self._compact_policy_capacity,
                    cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                    compact_group_payload_map=self._compact_group_payload_map,
                    blocks_per_chunk=self.blocks_per_chunk,
                    compact_cpu_budget_bytes=self._compact_per_rank_budget,
                    compact_page_size=compact_page_size,
                )
            else:
                self._manager = CPUOffloadingManager(
                    num_blocks=self.num_blocks,
                    cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        if self._enable_compact_layout:
            if self._compact_slice_accounting is None:
                raise ValueError(
                    "compact worker requires worker-local physical slice accounting"
                )
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_blocks=self.num_blocks,
                compact_slice_accounting=self._compact_slice_accounting,
                compact_cpu_budget_bytes_per_rank=self._compact_per_rank_budget,
            )
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
