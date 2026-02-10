from typing import Any

import vllm.envs as envs
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackEncoder

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.config_data import (
    MooncakeStoreConnectorMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
)


class KVPoolScheduler:
    def __init__(self, vllm_config: "VllmConfig", use_layerwise):
        self.use_layerwise = use_layerwise
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.consumer_is_to_load = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_load", False
        )
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self.client = LookupKeyClient(vllm_config)
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)

        self._block_size = vllm_config.cache_config.block_size
        if self.pcp_size > 1:
            self._block_size *= self.pcp_size
        if self.dcp_size > 1:
            self._block_size *= self.dcp_size
        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
        self._preempted_req_ids: set[str] = set()
        # Whether to discard partial chunks
        self._discard_partial_chunks = vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", True
        )
        self._unfinished_requests: dict[str, tuple[Request, list[int]]] = {}
        self._unfinished_request_ids: set[str] = set()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_load:
            return 0, False

        if self._discard_partial_chunks:
            token_len = len(request.prompt_token_ids) // self._block_size * self._block_size
        else:
            token_len = len(request.prompt_token_ids)

        num_external_hit_tokens = self.client.lookup(token_len, request.block_hashes)

        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.info(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        return need_to_allocate, self.load_async and not self.use_layerwise

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        local_block_ids = []
        if num_external_tokens > 0:
            local_block_ids = blocks.get_block_ids()[0]

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" and not self.consumer_is_to_put

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)

        meta = MooncakeStoreConnectorMetadata(self._unfinished_request_ids, scheduler_output.preempted_req_ids)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request.req_id]
            request_tuple = self._unfinished_requests.get(request.req_id)
            request_real = request_tuple[0]  # type: ignore[index]
            if not isinstance(request.block_ids[0], list):
                unfolded_block_ids = request.block_ids.copy()
            else:
                unfolded_block_ids = request.block_ids[0].copy()
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=0,
            )
            self._request_trackers[request.req_id] = request_tracker
            last_chunk_tokens_num = (
                (len(request.prompt_token_ids) // self._block_size * self._block_size)
                if self._discard_partial_chunks
                else len(request.prompt_token_ids)
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                block_hashes=request_real.block_hashes,
                is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                discard_partial_chunks=self._discard_partial_chunks,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                # resumed request
                new_block_ids = cached_reqs.new_block_ids[i]
                if not new_block_ids:
                    continue
                if req_id in self._preempted_req_ids:
                    if isinstance(new_block_ids, tuple):
                        new_block_ids = new_block_ids[0].copy()
                    else:
                        new_block_ids = new_block_ids.copy()
                    self._preempted_req_ids.discard(req_id)
                    load_spec = self.load_specs.pop(req_id, None)
                    request_tuple = self._unfinished_requests.get(req_id)
                    request_real = request_tuple[0]  # type: ignore[index]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids=new_block_ids,
                        num_saved_tokens=0,
                    )
                    self._request_trackers[req_id] = request_tracker
                    last_chunk_tokens_num = (
                        (len(request_real.prompt_token_ids) // self._block_size * self._block_size)
                        if self._discard_partial_chunks
                        else len(request_real.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request_real.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                    )

                # decode/chunked request
                else:
                    request_tracker = self._request_trackers[req_id]
                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        request = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = request.all_token_ids[num_current_tokens : num_current_tokens + num_new_tokens]
                        request_tracker.token_len += len(new_token_ids)
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached"
                        )
                    num_computed_token = cached_reqs.num_computed_tokens[i]
                    if num_computed_token >= len(request.prompt_token_ids):
                        continue
                    request_tracker.update(new_block_ids)

                    last_chunk_tokens_num = (
                        (len(request.prompt_token_ids) // self._block_size * self._block_size)
                        if self._discard_partial_chunks
                        else len(request.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=None,
                        skip_save=force_skip_save,
                        block_hashes=request.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                    )
                if req_meta is not None:
                    meta.add_request(req_meta)

        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (request, block_ids) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                if not load_spec:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                if (num_tokens_to_compute % self._block_size != 0) and (
                    num_tokens_to_compute == len(request.prompt_token_ids) - 1
                ):
                    num_tokens_to_compute = num_tokens_to_compute + 1
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=block_ids,
                    num_saved_tokens=0,
                )

                self._request_trackers[request_id] = request_tracker

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=request.block_hashes,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            return False, None
        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", len(block_ids), request.request_id)
        return delay_free_blocks, None


class LookupKeyClient:
    def __init__(self, vllm_config: "VllmConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(self, token_len: int, block_hashes: list[BlockHash]) -> int:
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        token_len_bytes = token_len.to_bytes(4, byteorder="big")
        all_frames = [token_len_bytes] + list(hash_frames)
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config: "VllmConfig") -> str:
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if "lookup_rpc_port" in extra_config:
            rpc_port = extra_config["lookup_rpc_port"]
        elif "mooncake_rpc_port" in extra_config:
            rpc_port = extra_config["mooncake_rpc_port"]
            logger.warning(
                "It is recommended to use the lookup_rpc_port, as the mooncake_rpc_port will be removed in the future."
            )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}"
