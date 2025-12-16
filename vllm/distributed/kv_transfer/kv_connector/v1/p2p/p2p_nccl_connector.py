# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import regex as re
import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request block ids
    block_ids: torch.Tensor
    # Request num tokens
    num_tokens: int

    @staticmethod
    def make_meta(
        request_id: str, token_ids: list[int], block_ids: list[int], block_size: int
    ) -> "ReqMeta":
        block_ids_tensor = torch.tensor(block_ids)
        return ReqMeta(
            request_id=request_id,
            block_ids=block_ids_tensor,
            num_tokens=len(token_ids),
        )


@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, token_ids, block_ids, block_size)
        )


class P2pNcclConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.is_producer = self._kv_transfer_config.is_kv_producer
        self.chunked_prefill: dict[str, tuple[list[int], list[int] | None]] = {}

        self._rank = get_world_group().rank if role == KVConnectorRole.WORKER else 0
        self._local_rank = (
            get_world_group().local_rank if role == KVConnectorRole.WORKER else 0
        )

        self.p2p_nccl_engine = (
            P2pNcclEngine(
                local_rank=self._local_rank,
                config=self._kv_transfer_config,
                hostname="",
                port_offset=self._rank,
            )
            if role == KVConnectorRole.WORKER
            else None
        )

        # Worker-side bookkeeping for async KV loads.
        # Track which (request_id, block_ids) has already been injected into the
        # paged KV buffer. Requests can be preempted and later resumed with new
        # blocks, so we must allow re-loading when block_ids change.
        self._loaded_req_block_ids: dict[str, tuple[int, ...]] = {}
        self._finished_recving_req_ids: set[str] = set()
        self._invalid_block_ids: set[int] = set()

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        # Only consumer/decode loads KV Cache
        if self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        attn_metadata = forward_context.attn_metadata

        def inject_kv_into_layer(
            layer: torch.Tensor,
            kv_cache: torch.Tensor,
            block_ids: torch.Tensor,
            request_id: str,
        ) -> None:
            """
            Inject KV cache data into a given attention layer tensor.

            This function updates `layer` in-place with values from `kv_cache`,
            handling different backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            If the number of provided block IDs does not match the number of KV
            blocks, only the overlapping portion is updated, and a warning is
            logged.

            Args:
                layer (torch.Tensor): The attention layer KV tensor to update.
                kv_cache (torch.Tensor): The KV cache tensor to inject.
                block_ids (torch.Tensor): Indices of the blocks to update.
                request_id (str): Request identifier used for logging.

            Returns:
                None. The function modifies `layer` in-place.
            """
            # NOTE: forward_context.attn_metadata can be None in kv_connector_no_forward
            # (used to drive background transfers). Fall back to shape-based
            # heuristics in that case.
            if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:  # type: ignore[arg-type]
                num_block = kv_cache.shape[0]
                self.check_tensors_except_dim(layer, kv_cache, 0)
                if len(block_ids) == num_block:
                    layer[block_ids, ...] = kv_cache
                else:
                    overlap = min(len(block_ids), num_block)
                    if overlap == 0:
                        logger.warning(
                            "🚧kv_cache does not match (no overlap), block_ids:%d, "
                            "num_block:%d, request_id:%s",
                            len(block_ids),
                            num_block,
                            request_id,
                        )
                        return
                    layer[block_ids[:overlap], ...] = kv_cache[:overlap, ...]
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, overlap:%d, request_id:%s",
                        len(block_ids),
                        num_block,
                        overlap,
                        request_id,
                    )

            elif layer.shape[0] == 2:  # FlashAttention
                num_block = kv_cache.shape[1]
                self.check_tensors_except_dim(layer, kv_cache, 1)
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    overlap = min(len(block_ids), num_block)
                    if overlap == 0:
                        logger.warning(
                            "🚧kv_cache does not match (no overlap), block_ids:%d, "
                            "num_block:%d, request_id:%s",
                            len(block_ids),
                            num_block,
                            request_id,
                        )
                        return
                    layer[:, block_ids[:overlap], ...] = kv_cache[:, :overlap, ...]
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, overlap:%d, request_id:%s",
                        len(block_ids),
                        num_block,
                        overlap,
                        request_id,
                    )

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)

        if metadata is None:
            return

        # Load the KV for each request each layer (non-blocking).
        for request in metadata.requests:
            request_id = request.request_id
            block_ids_key = tuple(int(b) for b in request.block_ids.tolist())
            if self._loaded_req_block_ids.get(request_id) == block_ids_key:
                continue

            ip, port = self.parse_request_id(request_id, False)
            remote_address = ip + ":" + str(port + self._rank)

            # Collect attention-layer KV buffers to inject into.
            layers_to_load: list[tuple[str, torch.Tensor]] = []
            for layer_name, layer_mod in forward_context.no_compile_layers.items():
                kv_cache = getattr(layer_mod, "kv_cache", None)
                if kv_cache is None:
                    continue
                layers_to_load.append(
                    (layer_name, kv_cache[forward_context.virtual_engine])
                )

            ok = True
            tensor_ids = [
                f"{request_id}#{layer_name}" for layer_name, _ in layers_to_load
            ]

            if self.p2p_nccl_engine.send_type == "GET":
                # Sync pull-based loading: fetch tensors from the remote producer
                # as needed.
                for layer_name, layer in layers_to_load:
                    kv_cache = self.p2p_nccl_engine.recv_tensor(
                        f"{request_id}#{layer_name}",
                        remote_address,
                    )
                    if kv_cache is None:
                        logger.warning(
                            "🚧kv_cache is None (GET load failure), "
                            "request_id=%s layer=%s",
                            request_id,
                            layer_name,
                        )
                        ok = False
                        break
                    inject_kv_into_layer(layer, kv_cache, request.block_ids, request_id)
            else:
                # Async push-based loading: only inject when all layers have
                # arrived to avoid blocking worker threads.
                if tensor_ids and not self.p2p_nccl_engine.has_all_recv_tensors(
                    tensor_ids
                ):
                    # Not ready yet (prefill may not have sent KV caches).
                    continue

                for layer_name, layer in layers_to_load:
                    kv_cache = self.p2p_nccl_engine.get_recv_tensor(
                        f"{request_id}#{layer_name}"
                    )
                    if kv_cache is None:
                        logger.warning(
                            "🚧kv_cache is None (load failure), request_id=%s layer=%s",
                            request_id,
                            layer_name,
                        )
                        ok = False
                        break

                    inject_kv_into_layer(layer, kv_cache, request.block_ids, request_id)

            if not ok:
                # Best-effort failure reporting: mark all blocks invalid so the
                # scheduler can recompute them (kv_load_failure_policy=recompute).
                self._invalid_block_ids.update(
                    int(b) for b in request.block_ids.tolist()
                )
                # For push-based async loads, mark the request as finished_recving
                # so the scheduler can handle invalid blocks. For GET (sync),
                # we do not use the finished_recving path.
                if self.p2p_nccl_engine.send_type != "GET":
                    self._finished_recving_req_ids.add(request_id)
                self._loaded_req_block_ids[request_id] = block_ids_key
                continue

            if self.p2p_nccl_engine.send_type != "GET":
                self._finished_recving_req_ids.add(request_id)
            self._loaded_req_block_ids[request_id] = block_ids_key

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        def extract_kv_from_layer(
            layer: torch.Tensor,
            block_ids: torch.Tensor,
        ) -> torch.Tensor:
            """
            Extract KV cache slices from a given attention layer tensor.

            This function handles multiple backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            Args:
                layer (torch.Tensor): The KV cache from the attention layer.
                block_ids (torch.Tensor): Indices of blocks to extract.

            Returns:
                torch.Tensor: A tensor containing the extracted KV slices.
                Returns None if the layout is unsupported.
            """
            if (
                isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2
            ):  # MLA or FlashInfer
                return layer[block_ids, ...]

            if layer.shape[0] == 2:  # FlashAttention
                return layer[:, block_ids, ...]

            return None

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
        for request in connector_metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, True)
            remote_address = ip + ":" + str(port + self._rank)

            kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
            self.p2p_nccl_engine.send_tensor(
                request_id + "#" + layer_name, kv_cache, remote_address
            )

    def wait_for_save(self):
        if self.is_producer:
            assert self.p2p_nccl_engine is not None
            self.p2p_nccl_engine.wait_for_sent()

    def get_finished(
        self, finished_req_ids: set[str], **kwargs: Any
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        assert self.p2p_nccl_engine is not None

        no_compile_layers = self._vllm_config.compilation_config.static_forward_context
        finished_sending, _ = self.p2p_nccl_engine.get_finished(
            finished_req_ids, no_compile_layers
        )

        # Clear worker-side loaded bookkeeping for finished requests.
        for req_id in finished_req_ids:
            self._loaded_req_block_ids.pop(req_id, None)

        finished_recving: set[str] | None = None
        if not self.is_producer and self._finished_recving_req_ids:
            finished_recving = set(self._finished_recving_req_ids)
            self._finished_recving_req_ids.clear()

        return finished_sending, finished_recving

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            return 0, False

        prompt_token_ids = request.prompt_token_ids or []
        num_external_tokens = len(prompt_token_ids) - 1 - num_computed_tokens

        if num_external_tokens < 0:
            num_external_tokens = 0

        send_type = self._kv_transfer_config.get_from_extra_config(
            "send_type", "PUT_ASYNC"
        )

        # GET mode is pull-based (no recv_store), so our async readiness-based
        # flow (WAITING_FOR_REMOTE_KVS + kv_connector_no_forward) is not
        # supported. Fall back to sync loading for correctness.
        if send_type == "GET":
            return num_external_tokens, False

        # PUT/PUT_ASYNC: async remote KV load. Let the scheduler park the request
        # in WAITING_FOR_REMOTE_KVS and drive KV loading via kv_connector_no_forward.
        return num_external_tokens, num_external_tokens > 0

    def update_connector_output(self, connector_output: KVConnectorOutput):
        # Scheduler-side: drop requests whose async KV load has finished.
        if self.is_producer:
            return
        for req_id in connector_output.finished_recving or ():
            self._requests_need_load.pop(req_id, None)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request,
                blocks.get_block_ids()[0],
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        meta = P2pNcclConnectorMetadata()

        if not self.is_producer:
            send_type = self._kv_transfer_config.get_from_extra_config(
                "send_type", "PUT_ASYNC"
            )
            if send_type != "GET":
                # Consumer: drive async KV loads even when the request isn't
                # scheduled for compute (WAITING_FOR_REMOTE_KVS).
                #
                # NOTE: Do NOT pop `_requests_need_load` here. The scheduler-side
                # connector can only safely drop entries after the worker reports
                # `finished_recving` (via update_connector_output()).
                for req_id, (req, block_ids) in self._requests_need_load.items():
                    meta.add_request(
                        request_id=req_id,
                        token_ids=req.prompt_token_ids or [],
                        block_ids=block_ids,
                        block_size=self._block_size,
                    )
                return meta

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = (scheduler_output.num_scheduled_tokens)[
                    new_req.req_id
                ]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                # the request's prompt is chunked prefill
                if num_tokens < len(new_req.prompt_token_ids or []):
                    # 'CachedRequestData' has no attribute 'prompt_token_ids'
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0],
                        new_req.prompt_token_ids,
                    )
                    continue
                # the request's prompt is not chunked prefill
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids or [],
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
                continue
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids or [],
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
                self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids

            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
                assert req_id in self.chunked_prefill
                assert new_block_ids is not None
                block_ids = new_block_ids[0]
                if not resumed_from_preemption:
                    block_ids = self.chunked_prefill[req_id][0] + block_ids
                prompt_token_ids = self.chunked_prefill[req_id][1]
                assert prompt_token_ids is not None
                # the request's prompt is chunked prefill again
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
                    continue
                # the request's prompt is all prefilled finally
                meta.add_request(
                    request_id=req_id,
                    token_ids=prompt_token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )
                self.chunked_prefill.pop(req_id, None)
                continue

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                assert new_block_ids is not None
                block_ids = new_block_ids[0]

                meta.add_request(
                    request_id=req_id,
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )

        self._requests_need_load.clear()
        return meta

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.is_producer or not self._invalid_block_ids:
            return set()
        invalid = set(self._invalid_block_ids)
        self._invalid_block_ids.clear()
        return invalid

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        self.chunked_prefill.pop(request.request_id, None)

        return False, None

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(f"Request id {request_id} does not contain hostname and port")

    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
            s1 == s2 for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim
        ):
            raise NotImplementedError(
                "Currently, only symmetric TP is supported. Asymmetric TP, PP,"
                "and others will be supported in future PRs."
            )
