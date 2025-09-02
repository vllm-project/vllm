# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import logging
import math
import os
import queue
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import numpy as np
import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm.distributed.kv_transfer.kv_connector.v1.p2p_xfer_connector import (
    P2PXferConnector,
    P2PXferConnectorScheduler,
    P2PXferConnectorWorker,
)

EngineId = str
logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None

try:
    from nixl._api import nixl_agent_config
except ImportError:
    nixl_agent_config = None
    logger.warning("NIXL agent config is not available")

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": ("cpu",),
    "cpu": ("cpu",),
}

# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._bindings import nixlXferTelemetry
    logger.info("NIXL Telemetry is available")
except ImportError:
    logger.warning("NIXL Telemetry is not available")
    nixlXferTelemetry = None

class NixlConnector(P2PXferConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: NixlConnectorScheduler | None = (
                NixlConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: NixlConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(vllm_config, self.engine_id)

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            NixlKVConnectorStats(data=data)
            if data is not None
            else NixlKVConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ) -> KVConnectorPromMetrics:
        return NixlPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

class NixlConnectorScheduler(P2PXferConnectorScheduler):
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        super().__init__(vllm_config, engine_id)

        # Legacy variable support; super reads from VLLM_P2PXFER* and override here if needed
        override_nixl_host = os.getenv("VLLM_NIXL_SIDE_CHANNEL_HOST", None)
        override_nixl_port = os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", None)
        override_nixl_timeout = os.getenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", None)
        if (override_nixl_host is not None):
            self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        if (override_nixl_port is not None):
            self.side_channel_port = (
                int(override_nixl_port)
                + vllm_config.parallel_config.data_parallel_rank
            )
        if (override_nixl_timeout is not None):
            self.abort_request_timeout = int(override_nixl_timeout)


class NixlConnectorWorker(P2PXferConnectorWorker):
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        super().__init__(vllm_config, engine_id)
        override_nixl_timeout = os.getenv("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", None)
        logger.info("[DBG] VLLM_NIXL_ABORT_REQUEST_TIMEOUT=%s", override_nixl_timeout)
        if (override_nixl_timeout is not None):
            self.abort_request_timeout = int(override_nixl_timeout)

        self.xfer_impl = NixlP2PWrapper(vllm_config, engine_id)

        self.xfer_stats = NixlKVConnectorStats()


class NixlP2PWrapper:
    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set for NixlConnector")

        self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
            "backends", ["UCX"]
        )

        # TODO temporary, once nixl allows for telemetry flag in config
        # (next release), we can remove this env var.
        os.environ["NIXL_TELEMETRY_ENABLE"] = "1"

        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
        # initialization with "mlx5dv_devx_alloc_uar" errors.
        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
            "num_threads", 4
        )
        if nixl_agent_config is None:
            config = None
        else:
            config = (
                nixl_agent_config(backends=self.nixl_backends)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads)
            )

        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), config)

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = \
            vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            if self.kv_buffer_device == "cuda":
                nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                nixl_memory_type = "DRAM"
        if nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.nixl_memory_type = nixl_memory_type

        # nixl_prepped_dlist_handle.
        self.src_xfer_side_handle: int = 0
        # Map of engine_id -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: dict[EngineId, int] = {}

        self._registered_descs: list[Any] = []

    def release_xfer_handle(self, handle: int):
        self.nixl_wrapper.release_xfer_handle(handle)

    def send_notif(self, remote_agent_name: str, notif_msg: bytes):
        self.nixl_wrapper.send_notif(remote_agent_name, notif_msg)

    def get_new_notifs(self):
        return self.nixl_wrapper.get_new_notifs()

    def get_agent_metadata(self) -> bytes:
        return self.nixl_wrapper.get_agent_metadata()

    # (base_addr, region_len, self.tp_rank, "")
    def register_memory(self, caches_data: list):
        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)

    def register_xfer_list(self, remote_engine_id: EngineId, remote_agent_name: str, blocks_data: list):
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[remote_engine_id] = self.nixl_wrapper.prep_xfer_dlist(
            remote_agent_name, descs
        )

    # List of tuples (addr, self.block_len, self.tp_rank)
    def register_local_blocks(self, blocks_data: list):
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs
        )

    def read_kv_blocks(self, dst_engine_id: EngineId, local_block_descs_ids: np.ndarray, remote_block_descs_ids: np.ndarray, notif_id):
        # Get side handles.
        local_xfer_side_handle = self.src_xfer_side_handle
        remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]
        # Prepare transfer with Nixl.
        handle = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            local_xfer_side_handle,
            local_block_descs_ids,
            remote_xfer_side_handle,
            remote_block_descs_ids,
            notif_msg=notif_id,
        )

        # Begin async xfer.
        self.nixl_wrapper.transfer(handle)
        return handle

    def add_remote_agent(self, agent_metadata: bytes):
        remote_agent_name = self.nixl_wrapper.add_remote_agent(agent_metadata)
        return remote_agent_name

    def check_xfer_state(self, handle):
        return self.nixl_wrapper.check_xfer_state(handle)

    def get_xfer_telemetry(self, handle):
        return self.nixl_wrapper.get_xfer_telemetry(handle)

    def shutdown(self, remote_agents : dict[EngineId, dict[int, str]]):
        if self.src_xfer_side_handle:
            self.nixl_wrapper.release_dlist_handle(self.src_xfer_side_handle)
            self.src_xfer_side_handle = 0

        for dst_xfer_side_handle in self.dst_xfer_side_handles.values():
            self.nixl_wrapper.release_dlist_handle(dst_xfer_side_handle)
        self.dst_xfer_side_handles.clear()

        for remote_agents_dict in remote_agents.values():
            for agent_name in remote_agents_dict.values():
                self.nixl_wrapper.remove_remote_agent(agent_name)

        for desc in self._registered_descs:
            self.nixl_wrapper.deregister_memory(desc)
        self._registered_descs.clear()


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
        }

    def record_transfer(self, res: nixlXferTelemetry):
        # Keep metrics units consistent with rest of the code: time us->s
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_failed_transfer(self):
        """Record a failed NIXL transfer operation."""
        self.data["num_failed_transfers"].append(1.0)

    def record_failed_notification(self):
        """Record a failed NIXL notification (send_notif)."""
        self.data["num_failed_notifications"].append(1.0)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return self.num_successful_transfers == 0

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging
        if self.is_empty():
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        post_time = np.asarray(self.data["post_duration"])
        # Convert to MB for CLI logging.
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n

        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90) * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90) * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean(), 1),
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class NixlPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        nixl_histogram_xfer_time = self._histogram_cls(
            name="vllm:nixl_xfer_time_seconds",
            documentation="Histogram of transfer duration for NIXL KV Cache transfers.",
            buckets=buckets[1:],
            labelnames=labelnames,
        )
        self.nixl_histogram_xfer_time = self.make_per_engine(nixl_histogram_xfer_time)
        nixl_histogram_post_time = self._histogram_cls(
            name="vllm:nixl_post_time_seconds",
            documentation="Histogram of transfer post time for NIXL KV"
            " Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_post_time = self.make_per_engine(nixl_histogram_post_time)
        # uniform 2kb to 16gb range
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
        nixl_histogram_bytes_transferred = self._histogram_cls(
            name="vllm:nixl_bytes_transferred",
            documentation="Histogram of bytes transferred per NIXL KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_bytes_transferred = self.make_per_engine(
            nixl_histogram_bytes_transferred
        )
        buckets = [
            10,
            20,
            30,
            50,
            75,
            100,
            200,
            400,
            1000,
            2000,
            4000,
            10000,
            20000,
            50000,
        ]
        nixl_histogram_num_descriptors = self._histogram_cls(
            name="vllm:nixl_num_descriptors",
            documentation="Histogram of number of descriptors per NIXL"
            "  KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_num_descriptors = self.make_per_engine(
            nixl_histogram_num_descriptors
        )
        counter_nixl_num_failed_transfers = self._counter_cls(
            name="vllm:nixl_num_failed_transfers",
            documentation="Number of failed NIXL KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_transfers = self.make_per_engine(
            counter_nixl_num_failed_transfers
        )
        counter_nixl_num_failed_notifications = self._counter_cls(
            name="vllm:nixl_num_failed_notifications",
            documentation="Number of failed NIXL KV Cache notifications.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_notifications = self.make_per_engine(
            counter_nixl_num_failed_notifications
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.nixl_histogram_xfer_time,
                self.nixl_histogram_post_time,
                self.nixl_histogram_bytes_transferred,
                self.nixl_histogram_num_descriptors,
            ],
            [
                "transfer_duration",
                "post_duration",
                "bytes_transferred",
                "num_descriptors",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)
        for counter_obj, counter_item_key in zip(
            [
                self.counter_nixl_num_failed_transfers,
                self.counter_nixl_num_failed_notifications,
            ],
            ["num_failed_transfers", "num_failed_notifications"],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)

