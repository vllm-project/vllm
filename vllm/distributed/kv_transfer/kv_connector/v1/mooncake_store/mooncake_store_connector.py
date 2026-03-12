import threading
from typing import Any

import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
#from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.pool_scheduler import (
    KVPoolScheduler,
    get_zmq_rpc_path_lookup,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.pool_worker import KVPoolWorker


class MooncakeStoreConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )

        connector_name = vllm_config.kv_transfer_config.kv_connector
        if connector_name == "MooncakeConnectorStoreV1":
            logger.info(
                "Using the MooncakeConnectorStoreV1 from mooncake.store."
            )

        self.kv_caches: dict[str, torch.Tensor] = {}

        self.sent_but_unfinished_reqs: set[str] = set()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = KVPoolScheduler(vllm_config, self.use_layerwise)
        else:
            self.connector_worker = KVPoolWorker(
                vllm_config,
                self.use_layerwise,
            )

            assert self.connector_worker is not None
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config, self.use_layerwise)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise:
            return
        self.connector_worker.wait_for_layer_load()

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        self.connector_worker.save_kv_layer(self._get_connector_metadata())

    def wait_for_save(self):
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            return

        self.connector_worker.wait_for_save(self._get_connector_metadata())

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """

        # Should only be called while set to valid metadata.
        return self._connector_metadata

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        if self._get_connector_metadata() is None:
            return None, None

        done_sending, done_recving = self.connector_worker.get_finished(
            finished_req_ids, self._get_connector_metadata()
        )
        return done_sending, done_recving


class LookupKeyServer:
    def __init__(
        self,
        pool_worker: KVPoolWorker,
        vllm_config: "VllmConfig",
        use_layerwise: bool,
    ):
        self.decoder = MsgpackDecoder()
        self.decoder_tensor = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.pool_worker = pool_worker
        self.running = True
        self.use_layerwise = use_layerwise

        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                token_len = int.from_bytes(all_frames[0], byteorder="big")
                hash_frames = all_frames[1:]
                hashes_str = self.decoder.decode(hash_frames)
                result = self.pool_worker.lookup_scheduler(token_len, hashes_str, self.use_layerwise)
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
