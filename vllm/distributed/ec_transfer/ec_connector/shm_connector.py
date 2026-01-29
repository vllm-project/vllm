# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/distributed/ec_transfer/ec_connector/example_connector.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
import time, queue, torch, zmq, contextlib, pickle
import gc, shutil
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.parallel_state import get_world_group
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.multiprocessing.reductions import reduce_tensor
from collections.abc import Iterator

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


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
    # It transfers the EC cache between processes (Producer/Consumer) by sharing memory handles.

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._mm_datas_need_loads: dict[str, int] = {}
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is not None:
            self._storage_path = \
                transfer_config.get_from_extra_config(
                "shared_storage_path", "/tmp"
            )
            logger.debug(transfer_config)
            logger.debug("Shared storage path is %s", self._storage_path)
        else:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

        if role == ECConnectorRole.SCHEDULER:
            return
        ec_extra_config = getattr(transfer_config, "ec_connector_extra_config", {})
        self.listen_ports = ec_extra_config.get("listen_ports", None)
        if not self.listen_ports:
            raise ValueError(
                "Producer/Consumer must have 'listen_ports' in ec_connector_extra_config."
                )
        self.consumer_sock_addrs = [
            (transfer_config.ec_ip, addr_port) for addr_port in self.listen_ports
            ]
        logger.debug("Consumer addrs is: %s", self.consumer_sock_addrs)

        self.thread_executor = \
            ThreadPoolExecutor(
            max_workers=getattr(transfer_config, "max_workers", 8) or 8
        )
        if transfer_config.ec_role == "ec_producer":
            self.send_queue = queue.Queue[tuple[str, torch.Tensor]]()
            self.zmq_paths = [
                make_zmq_path("tcp", host, port) for host, port in self.consumer_sock_addrs
                ]
            self.thread_executor.submit(self.producer_run)
            logger.debug("============ Producer Mode ===============")
        elif transfer_config.ec_role == "ec_consumer":
            self.handle_caches = {}
            self.recv_queue = queue.Queue()
            self.thread_executor.submit(self.consumer_run)
            self.thread_executor.submit(self.recv_feat_async)
            logger.debug("============= Consumer Start =============")

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

       This method maps the metadata provided by the scheduler to the internal handle storage and registers the tensors in the `encoder_cache`.
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
                func, args = self.handle_caches.get(mm_data.mm_hash, None)
                list_args = list(args)
                list_args[6] = get_world_group().local_rank
                encoder_cache[mm_data.mm_hash] = func(*list_args)
                logger.debug("recv tensor for hash %s", mm_data.mm_hash)
            except Exception as e:
                logger.error(
                    f"Unhandled Cache Miss {mm_data.mm_hash}, error code: {str(e)}"
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
                self.handle_caches.pop(request_id + "-image-0", None)
        else:
            for request_id in finished_req_ids:
                gc.collect()
                torch.cuda.empty_cache()
        return None, None

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

    def shared_handle_send(self, path, send_data):
        """Send shared memory handle to a specific ZMQ address and wait for ACK."""
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            ensure_zmq_send(sock, pickle.dumps(send_data))
            ack = sock.recv()
            if ack != b"ACK":
                raise ValueError(f"Unexpected ACK response: {ack}")
            return ack

    def producer_run(self):
        """
        Background worker for the Producer. 
        Detaches tensors, reduces them to shared handles, and broadcasts to all consumers.
        """
        while True:
            try:
                feat_key, tensor = self.send_queue.get()
                shared_handle = reduce_tensor(tensor.detach().clone())
                send_data = {"key": feat_key,"value": shared_handle}
                future_list = []
                for path in self.zmq_paths:
                    future = self.thread_executor.submit(
                        self.shared_handle_send, path, send_data
                        )
                    future_list.append(future)
                ack_count = 0
                for future in as_completed(future_list):
                    try:
                        task_result = future.result()
                        if task_result == b"ACK":
                            ack_count += 1
                    except Exception:
                        raise ValueError(f"Unexpected ACK response: {task_result}")
                if len(self.zmq_paths) == ack_count:
                    filename = self._generate_filename_debug(feat_key)
                    logger.debug(
                        "rank %s send the feat key %s, filename %s", 
                        get_world_group().local_rank, 
                        feat_key, 
                        filename)
                self.send_queue.task_done()
            except Exception as e:
                logger.error(
                    f"put key: {feat_key} into store fail, error code: {str(e)}"
                    )
                if 'feat_key' in locals():
                    self.send_queue.task_done()
                continue

    def recv_feat_async(self):
        """Background worker to process received payloads from the consumer queue."""
        while True:
            try:
                payload = self.recv_queue.get()
                data = pickle.loads(payload)
                feat_key = data["key"]
                share_handle = data["value"]
                self.handle_caches[feat_key] = share_handle
                self.recv_queue.task_done()
            except Exception as e:
                logger.error(
                    f"get key: {feat_key} into store fail, error code: {str(e)}"
                    )
                if 'feat_key' in locals():
                    self.recv_queue.task_done()
                continue

    def consumer_run(self):
        """
        Background listener for the Consumer role.
        Opens a ZMQ ROUTER socket to receive handles from the Producer.
        """
        local_rank = get_world_group().local_rank
        tp_size = len(self.consumer_sock_addrs)
        side_channel_host, handshake_port = self.consumer_sock_addrs[
            local_rank % tp_size
            ]
        path = make_zmq_path("tcp", side_channel_host, handshake_port)
        logger.debug("Starting listening on path: %s", path)

        with zmq_ctx(zmq.ROUTER, path) as sock:
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error("Invalid message format: %s", frames)
                        continue
                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid message format: %s", frames)
                        continue
                    sock.send_multipart((identity, b"", b"ACK"))
                    self.recv_queue.put(payload[0])
                except Exception as e:
                    logger.error("Failed to decode message: %s", e)


def ensure_zmq_send(socket: zmq.Socket, data: list, max_retries: int = 3):
    """Reliably send data over a ZMQ socket with a retry mechanism."""
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:
            retries_left -= 1
            if retries_left > 0:
                logger.warning(
                    "Send failed: %s, retrying... (%s attempts left)", e, 
                    retries_left
                )
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries: %s", e)
                raise RuntimeError(
                    f"Failed to send data after {max_retries} retries: {e}")

@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """
    Context manager for managing ZMQ life-cycle.
    Handles context creation, socket binding/connection, and clean destruction.
    """
    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):
        raise ValueError(f"Unexpected socket type: {socket_type}")
    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
            )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
            