# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/distributed/ec_transfer/ec_connector/example_connector.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed.rpc as rpc
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

_LOCAL_CONNECTOR: Optional["SHMConnector"] = None

def _rpc_receive_handle(feat_key: str, handle_data: Any) -> str:
    """Synchronously receive RPC data and write directly to cache.

    Args:
        feat_key (str): Feature key for the cached data (corresponds to mm_hash).
        handle_data (Any): Data to be stored in the cache (encoder cache tensor).
    """
    if (_LOCAL_CONNECTOR is not None and hasattr(_LOCAL_CONNECTOR, 'handle_caches')):
        _LOCAL_CONNECTOR.handle_caches[feat_key] = handle_data
        logger.debug("RPC received and cached key: %s", feat_key)
        return "ACK"
    
    return "NOT_READY"


@dataclass
class MMMeta:
    mm_hash: str
    num_token: int

    @staticmethod
    def make_meta(mm_hash, num_token) -> "MMMeta":
        return MMMeta(mm_hash=mm_hash, num_token=num_token)


@dataclass
class SHMConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas.append(mm_data)


class SHMConnector(ECConnectorBase):
    # NOTE: This is An implementation of the EC connector using Shared Memory (SHM).
    # It transfers the EC cache between processes (Producer/Consumer)
    # by sharing memory handles.

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
                
        self.handle_caches: dict[str, Any] = {}
        self._mm_datas_need_loads: dict[str, int] = {}

        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is not None:
            self._storage_path = transfer_config.get_from_extra_config(
                "shared_storage_path", "/tmp"
            )
            logger.debug(transfer_config)
            logger.debug("Shared storage path is %s", self._storage_path)
        else:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

        if role == ECConnectorRole.SCHEDULER:
            return

        ec_extra_config = getattr(transfer_config, "ec_connector_extra_config", {})
        self.ec_ip = transfer_config.ec_ip
        self.listen_ports = ec_extra_config.get("listen_ports", None)
        if not self.listen_ports:
            raise ValueError("Must have 'listen_ports' in ec_connector_extra_config.")
        self.max_workers = ec_extra_config.get("max_workers", 16)

        engine_id = ec_extra_config.get("engine_id", 0)
        produce_num = ec_extra_config.get("produce_instances", 1)
        consumer_num = ec_extra_config.get("consumer_instances", 1)

        producer_config: dict[str, Any] = ec_extra_config.get("producer", {})
        consumer_config: dict[str, Any] = ec_extra_config.get("consumer", {})
        producer_tp = producer_config.get("tp_size", 1)
        producer_dp = producer_config.get("dp_size", 1)
        producer_single_size = producer_tp * producer_dp
        consumer_tp = consumer_config.get("tp_size", 1)
        consumer_dp = consumer_config.get("dp_size", 1)
        consumer_single_size = consumer_tp * consumer_dp
        producer_size = produce_num * producer_single_size
        consumer_size = consumer_num * consumer_single_size
        self.rpc_world_size = producer_size + consumer_size

        vllm_local_rank = get_world_group().rank
        if transfer_config.ec_role == "ec_producer":
            self.rpc_rank = engine_id * producer_single_size + vllm_local_rank
            self.is_producer_node = True
        else:
            self.rpc_rank = (
                producer_size + engine_id * consumer_single_size + vllm_local_rank
            )
            self.is_producer_node = False
        self.rpc_name = f"worker_{self.rpc_rank}"
        master_port = str(self.listen_ports[0])

        if not rpc.api._is_current_rpc_agent_set():
            options = rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{self.ec_ip}:{master_port}", rpc_timeout=30.0
            )
            rpc.init_rpc(
                self.rpc_name,
                rank=self.rpc_rank,
                world_size=self.rpc_world_size,
                rpc_backend_options=options
            )

        if transfer_config.ec_role == "ec_producer":
            self.send_queue: queue.Queue[tuple[str, torch.Tensor]] = queue.Queue()
            self.consumer_names = [
                f"worker_{i}" for i in range(producer_size, self.rpc_world_size)
            ]
            self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self.thread_executor.submit(self.producer_run)

        # 4. RPC 初始化
        global _LOCAL_CONNECTOR
        _LOCAL_CONNECTOR = self
        logger.info("SHMConnector %s initialized successfully.", self.rpc_name)

    def producer_run(self):
        while True:
            try:
                feat_key, tensor = self.send_queue.get()
                shared_handle = reduce_tensor(tensor.detach().clone())
                futs = []
                for worker_name in self.consumer_names:
                    fut = rpc.rpc_async(
                        to=worker_name,
                        func=_rpc_receive_handle,
                        args=(feat_key, shared_handle),
                        timeout=20.0,
                    )
                    futs.append((worker_name, fut))

                all_received = True
                for worker_name, fut in futs:
                    try:
                        result = fut.wait()
                        if result != "ACK":
                            logger.warning(
                                "Worker %s did not ACK %s, got: %s",
                                worker_name,
                                feat_key,
                                result,
                            )
                            all_received = False
                    except Exception as e:
                        logger.error(
                            "Critical: Worker %s failed to receive %s. Error: %s",
                            worker_name,
                            feat_key,
                            e,
                        )
                        all_received = False

                if all_received:
                    self._generate_filename_debug(feat_key)
                    logger.info(
                        "Broadcast Success: %s received by all %d workers.",
                        feat_key,
                        len(self.consumer_names),
                    )
                else:
                    logger.error(
                        "Broadcast Incomplete: %s might be missing on some workers.",
                        feat_key,
                    )

                self.send_queue.task_done()
                
            except Exception as e:
                logger.error("Producer thread fatal error: %s", e)

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, SHMConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.start_load_caches, ",
                    "but the connector metadata is None",
                )
            )
            return
        # Load the EC for each mm data
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            try:
                item = self.handle_caches.get(mm_data.mm_hash)
                if item is None:
                    logger.warning("Cache miss for hash %s", mm_data.mm_hash)
                    continue
                func, args = item
                list_args = list(args)
                list_args[6] = get_world_group().local_rank
                encoder_cache[mm_data.mm_hash] = func(*list_args)
                logger.debug("recv tensor for hash %s", mm_data.mm_hash)
            except Exception as e:
                logger.error(
                    "Unhandled Cache Miss %s, error code: %s", mm_data.mm_hash, str(e)
                )

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        """
        Queue the encoder cache to consumers.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return
        self.send_queue.put((mm_hash, encoder_cache[mm_hash]))
        logger.debug("Save cache successful for mm_hash %s", mm_hash)

    def has_cache_item(
        self,
        identifier: str,
    ) -> bool:
        """
        Check if cache exist externally for the media

        Args:
            identifier (str): the identifier of the media.

        Returns:
            Bool indicate that media exists in cache or not
        """
        if self.is_producer:
            return False
        else:
            return self._found_match_for_mm_data(identifier)
    
    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        num_encoder_token = request.get_num_encoder_embeds(index)
        # Insert mm_hash only if this block has not been recorded yet.
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only build for load mm_data only
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = SHMConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_mm_data(self, mm_hash) -> bool:
        """Check if the cache is hit for the request."""
        # filename = self._generate_filename_debug(mm_hash)
        if not self.is_producer:
            foldername = os.path.join(self._storage_path, mm_hash)
            return os.path.exists(foldername)
        else:
            filename = self._generate_filename_debug(mm_hash)
            return os.path.exists(filename)

    def _generate_foldername_debug(
        self,
        mm_hash: str,
        create_folder: bool = True,  # <- now defaults to True
    ) -> str:
        """
        Return the folder in which the cache for this mm_hash lives.
        If `create_folder` is True (default) the directory is created
        recursively the first time it is needed.
        """
        foldername = os.path.join(self._storage_path, mm_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(self, mm_hash: str) -> str:
        """
        Return the full path of the safetensors file for this mm_hash.
        Ensures the parent directory exists because
        `_generate_foldername_debug` is called with its default
        (`create_folder=True`).
        """
        foldername = self._generate_foldername_debug(mm_hash)  # <- folder auto-created
        return foldername

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        if not self.is_producer:
            for request_id in finished_req_ids:
                gc.collect()
                torch.cuda.empty_cache()
        return None, None
