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
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
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

try:
    from mooncake.engine import TransferEngine as MooncakeWrapper
    logger.info("Mooncake xfer engine is available")
except ImportError:
    logger.warning("Mooncake xfer engine is not available")
    MooncakeWrapper = None

class MooncakeConnector(P2PXferConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, self.engine_id)

class MooncakeConnectorScheduler(P2PXferConnectorScheduler):
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        super().__init__(vllm_config, engine_id)

class MooncakeConnectorWorker(P2PXferConnectorWorker):
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        super().__init__(vllm_config, engine_id)
        self.xfer_impl = MooncakeP2PWrapper(vllm_config, engine_id)

class MooncakeP2PWrapper:
    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        tp_rank = get_tensor_model_parallel_rank()

        self.mooncake_instance_name=f"mooncake-{engine_id}-tp{tp_rank}"
        logger.debug(f"[XFER-MC] instance name {self.mooncake_instance_name} etcd_server={envs.VLLM_MOONCAKE_METADATA_SERVER} protocol={envs.VLLM_MOONCAKE_TRANSPORT_PROTOCOL}")

        # Current limitation in mooncake, the `protocol` argument to `initialize` is ignored.
        # Mooncake will use the protocol it was build for. However, we can force a RDMA-based
        # mooncake build to switch to TCP using the `MC_FORCE_TCP` env variable.
        if envs.VLLM_MOONCAKE_TRANSPORT_PROTOCOL == "tcp" and os.getenv("MC_FORCE_TCP", None) is None:
            os.environ["MC_FORCE_TCP"]="1"

        self.mooncake_wrapper = MooncakeWrapper()
        ret = self.mooncake_wrapper.initialize(self.mooncake_instance_name,
                                                envs.VLLM_MOONCAKE_METADATA_SERVER,
                                                envs.VLLM_MOONCAKE_TRANSPORT_PROTOCOL,
                                                envs.VLLM_MOONCAKE_DEVICE_NAME)
        if ret < 0:
            logger.error("Cannot initialize the mooncake xfer engine")
            raise RuntimeError("Cannot initialize the mooncake xfer engine")

        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = \
            vllm_config.kv_transfer_config.kv_buffer_device

        _MOONCAKE_SUPPORTED_XPUS = {
            "cuda": ("cuda", ),
            "tpu": ("cpu", ),
        }

        if self.device_type not in _MOONCAKE_SUPPORTED_XPUS:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _MOONCAKE_SUPPORTED_XPUS[
                self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported.")

        # tuple type is (base_addr, region_len, self.tp_rank)
        self.local_block_descs: list[tuple] = None
        # tuple is (string, list[tuple]) where string is the remote segment name
        # and list[tuple] is (addr, self.block_len, remote_tp_rank)
        self.remote_block_descs: dict[EngineId, tuple] = {}

    def release_xfer_handle(self, handle: int):
        logger.debug("[XFER-MC] release_xfer_handle called!")
        # mooncake automatically frees handles on completion or error.
        # attempting to release a handle already released is undefined behavior.
        pass

    def send_notif(self, remote_agent_name: str, notif_msg: bytes):
        logger.debug("[XFER-MC] send_notif called!")
        # mooncake doesn't have notifications
        pass

    def get_new_notifs(self):
        # mooncake doesn't have notifications
        return {}

    #xfer-agent specific metadata we want to expose
    def get_agent_metadata(self) -> bytes:
        logger.debug("[XFER-MC] get_agent_metadata called!")
        # This is the name to be used for mooncake agent-to-agent communications
        return self.mooncake_instance_name.encode('utf-8')

    def add_remote_agent(self, agent_metadata: bytes):
        logger.debug("[XFER-MC] add_remote_agent called!")
        # return the name of the targeted segment
        remote_agent_name = agent_metadata.decode('utf-8')
        return remote_agent_name

    # (base_addr, region_len, self.tp_rank, "")
    def register_memory(self, caches_data: list):
        logger.debug("[XFER-MC] register_memory called!")
        buffer_addresses = []
        capacities = []
        for kv_cache in caches_data:
            buffer_addresses.append(kv_cache[0])
            capacities.append(kv_cache[1])
            # Do not need to save the tp_rank here

        err = self.mooncake_wrapper.batch_register_memory(buffer_addresses, capacities)
        if err:
            raise RuntimeError("Cannot register kv-cache memory with mooncake xfer engine!")

    def register_xfer_list(self, remote_engine_id: EngineId, remote_agent_name: str, blocks_data: list):
        logger.debug("[XFER-MC] register_xfer_list called!")
        # Store blocks data (addresses & lengths) to be read/write at a later time
        self.remote_block_descs[remote_engine_id] = (remote_agent_name, blocks_data);

    # List of tuples (addr, self.block_len, self.tp_rank)
    def register_local_blocks(self, blocks_data: list):
        logger.debug("[XFER-MC] register_local_blocks called!")
        self.local_block_descs = blocks_data;

    def read_kv_blocks(self, dst_engine_id: EngineId, local_block_descs_ids: np.ndarray, remote_block_descs_ids: np.ndarray, notif_id):
        logger.debug("[XFER-MC] read_kv_blocks called!")
        #TODO: check using numpy
        src_buffers = []
        dst_buffers = []
        lengths = []

        # build list of local addresses
        for idx in local_block_descs_ids:
            block_desc = self.local_block_descs[idx]
            src_buffers.append(block_desc[0])
            lengths.append(block_desc[1])

        # build list of remote addresses
        remote_segment, rmte_block_descs = self.remote_block_descs[dst_engine_id]
        i=0
        for idx in remote_block_descs_ids:
            block_desc = rmte_block_descs[idx]
            dst_buffers.append(block_desc[0])
            # Lengths must match as the caller is responsible for generating
            assert lengths[i] == block_desc[1]
            i=i+1

        handle = self.mooncake_wrapper.batch_transfer_async_read(
            remote_segment, src_buffers, dst_buffers, lengths)

        if handle == 0:
            raise RuntimeError("batch transfer error encountered!")

        return handle

    def check_xfer_state(self, handle):
        logger.debug("[XFER-MC] check_xfer_state called!")
        xfer_state = self.mooncake_wrapper.transfer_check_status(handle)
        if xfer_state == 1:
            return "DONE"
        elif xfer_state == 0:
            return "PROC"
        elif xfer_state == -1:
            logger.debug("[XFER-MC] check_xfer_state FAILED!")
            return "ERR"
        elif xfer_state == -2:
            logger.debug("[XFER-MC] check_xfer_state TIMEOUT!")
            return "ERR"
        else:
            return "ERR"

    def shutdown(self, remote_agents : dict[EngineId, dict[int, str]]):
        #TODO: shutdown support
        pass
