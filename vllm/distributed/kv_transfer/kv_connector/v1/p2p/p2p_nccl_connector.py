# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_world_group,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _normalize_req_id(request_id: str) -> str:
    parts = request_id.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8:
        return parts[0]
    return request_id


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
    ssm_requests: list[ReqMeta]

    def __init__(self):
        self.requests = []
        self.ssm_requests = []

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

    def add_ssm_request(self, request_id: str) -> None:
        self.ssm_requests.append(
            ReqMeta(request_id=request_id, block_ids=torch.tensor([]), num_tokens=0)
        )


class P2pNcclConnector(KVConnectorBase_V1, SupportsHMA):
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # Mamba does not support cross-layer block layout
        if self._has_mamba:
            return False
        # P2P NCCL does not benefit from cross-layer blocks
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
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

        self._has_mamba = False
        self._conv_decomp: Any = None
        self._mamba_states_sent: set[str] = set()
        if role == KVConnectorRole.WORKER:
            self._detect_mamba(kv_cache_config)

    def _detect_mamba(self, kv_cache_config: "KVCacheConfig") -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        self._has_mamba = any(
            not isinstance(g.kv_cache_spec, FullAttentionSpec)
            for g in kv_cache_config.kv_cache_groups
        )
        if not self._has_mamba:
            return

        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        assert is_conv_state_dim_first(), (
            "P2pNccl mamba state transfer requires DS conv state layout. "
            "Set VLLM_SSM_CONV_STATE_LAYOUT=DS"
        )
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            ssm_conv_transfer_utils,
        )

        mamba_spec = next(
            spec
            for g in kv_cache_config.kv_cache_groups
            for spec in [g.kv_cache_spec]
            if isinstance(spec, MambaSpec)
        )
        self._conv_decomp = ssm_conv_transfer_utils.derive_mamba_conv_split(
            mamba_spec,
            get_tensor_model_parallel_world_size(),
        )
        logger.info(
            "P2pNcclConnector detected mamba model, conv_decomp=%s",
            self._conv_decomp,
        )

    # ==============================
    # Worker-side methods
    # ==============================

    def _is_attention_layer(self, layer: Any) -> bool:
        cache = getattr(layer, "kv_cache", None)
        return isinstance(cache, torch.Tensor)

    def _load_mamba_states(
        self,
        forward_context: "ForwardContext",
        request_id: str,
        remote_address: str,
        mamba_req_idx: int = 0,
    ) -> None:
        fc_attn_meta = forward_context.attn_metadata
        if not isinstance(fc_attn_meta, dict):
            return
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadata,
        )
        from vllm.v1.attention.backends.mamba_attn import (
            BaseMambaAttentionMetadata,
        )

        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            meta = fc_attn_meta.get(layer_name)
            if not isinstance(meta, (BaseMambaAttentionMetadata, GDNAttentionMetadata)):
                continue
            kv_cache = getattr(layer, "kv_cache", None)
            if isinstance(kv_cache, (tuple, list)) and len(kv_cache) > 0:
                kv_cache = kv_cache[0]
            if not isinstance(kv_cache, torch.Tensor) or kv_cache.shape[0] == 0:
                continue
            norm_id = _normalize_req_id(request_id)
            received = self.p2p_nccl_engine.recv_tensor(
                norm_id + "#" + layer_name + "_conv", remote_address
            )
            if received is None:
                continue
            slot = 0
            if isinstance(meta, BaseMambaAttentionMetadata):
                si = meta.state_indices_tensor_d
                if si is not None and mamba_req_idx < len(si):
                    slot = si[mamba_req_idx].item()
            if 0 <= slot < kv_cache.shape[0]:
                kv_cache[slot] = received.to(kv_cache.dtype)

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
        if attn_metadata is None:
            return

        def inject_kv_into_layer(
            layer: torch.Tensor,
            kv_cache: torch.Tensor,
            block_ids: torch.Tensor,
            request_id: str,
        ) -> None:
            """
            Inject KV cache data into a given attention layer tensor.

            This function updates `layer` in-place with values from `kv_cache`.
            All backends (MLA, FlashAttention, FlashInfer, TritonAttention)
            are indexed along the first dimension (block index).

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
            num_block = kv_cache.shape[0]
            self.check_tensors_except_dim(layer, kv_cache, 0)
            if len(block_ids) == num_block:
                layer[block_ids, ...] = kv_cache
            else:
                layer[block_ids[:num_block], ...] = kv_cache
                logger.warning(
                    "kv_cache does not match, block_ids:%d, "
                    "num_block:%d, request_id:%s",
                    len(block_ids),
                    num_block,
                    request_id,
                )

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)
        if metadata is None:
            return

        # Load the KV for each request each layer
        mamba_req_idx = 0
        for request in metadata.requests:
            request_id = request.request_id
            norm_id = _normalize_req_id(request_id)
            try:
                ip, port = self.parse_request_id(request_id, False)
            except ValueError:
                continue
            remote_address = ip + ":" + str(port + self._rank)
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoE
                if not self._is_attention_layer(layer):
                    continue
                kv_cache = layer.kv_cache
                received = self.p2p_nccl_engine.recv_tensor(
                    norm_id + "#" + layer_name, remote_address
                )
                if received is None:
                    continue
                inject_kv_into_layer(
                    kv_cache, received, request.block_ids, request.request_id
                )

            if self._has_mamba:
                self._load_mamba_states(
                    forward_context, request_id, remote_address, mamba_req_idx
                )
                mamba_req_idx += 1

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def _save_mamba_states(
        self,
        forward_context: "ForwardContext",
        connector_metadata: "P2pNcclConnectorMetadata",
    ) -> None:
        fc_attn_meta = forward_context.attn_metadata
        if not isinstance(fc_attn_meta, dict):
            return
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadata,
        )
        from vllm.v1.attention.backends.mamba_attn import (
            BaseMambaAttentionMetadata,
        )

        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            meta = fc_attn_meta.get(layer_name)
            if not isinstance(meta, (BaseMambaAttentionMetadata, GDNAttentionMetadata)):
                continue
            kv_cache = getattr(layer, "kv_cache", None)
            if isinstance(kv_cache, (tuple, list)) and len(kv_cache) > 0:
                kv_cache = kv_cache[0]
            if not isinstance(kv_cache, torch.Tensor) or kv_cache.shape[0] == 0:
                continue
            logger.debug(
                "_save_mamba_states: SENDING layer=%s shape=%s",
                layer_name,
                kv_cache.shape,
            )
            slot = 0
            if isinstance(meta, BaseMambaAttentionMetadata):
                si = meta.state_indices_tensor_p
                if si is not None and len(si) > 0:
                    slot = si[0].item()
            if slot >= kv_cache.shape[0]:
                continue
            ssm_requests = connector_metadata.requests
            for req_idx, request in enumerate(ssm_requests):
                norm_id = _normalize_req_id(request.request_id)
                tensor_id = norm_id + "#" + layer_name + "_conv"
                if tensor_id in self._mamba_states_sent:
                    continue
                try:
                    ip, port = self.parse_request_id(request.request_id, True)
                except ValueError:
                    continue
                remote_address = ip + ":" + str(port + self._rank)
                if isinstance(meta, BaseMambaAttentionMetadata):
                    si = meta.state_indices_tensor_p
                    if si is not None and req_idx < len(si):
                        slot = si[req_idx].item()
                if slot >= kv_cache.shape[0]:
                    continue
                state_slice = kv_cache[slot].clone()
                self.p2p_nccl_engine.send_tensor(
                    tensor_id,
                    state_slice,
                    remote_address,
                )
                self._mamba_states_sent.add(tensor_id)

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
        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
        for request in connector_metadata.requests:
            request_id = request.request_id
            norm_id = _normalize_req_id(request_id)
            try:
                ip, port = self.parse_request_id(request_id, True)
            except ValueError:
                continue
            remote_address = ip + ":" + str(port + self._rank)

            kv_cache = kv_layer[request.block_ids, ...]
            self.p2p_nccl_engine.send_tensor(
                norm_id + "#" + layer_name, kv_cache, remote_address
            )

        if self._has_mamba:
            from vllm.forward_context import get_forward_context

            fc = get_forward_context()
            if fc is not None:
                self._save_mamba_states(fc, connector_metadata)

    def clear_connector_metadata(self) -> None:
        self._mamba_states_sent.clear()
        super().clear_connector_metadata()

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
        # Normalize request IDs to match tensor_id keys in the engine
        norm_finished_req_ids = {_normalize_req_id(rid) for rid in finished_req_ids}
        no_compile_layers = self._vllm_config.compilation_config.static_forward_context
        sent, recvd = self.p2p_nccl_engine.get_finished(
            norm_finished_req_ids, no_compile_layers
        )

        if self._has_mamba:
            for request_id in finished_req_ids:
                norm_id = _normalize_req_id(request_id)
                for layer_name in no_compile_layers:
                    tensor_id = norm_id + "#" + layer_name + "_conv"
                    with self.p2p_nccl_engine.recv_store_cv:
                        self.p2p_nccl_engine.recv_store.pop(tensor_id, None)

        return sent, recvd

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
        # The last prompt token is always computed locally by the decoder
        num_external_tokens = len(prompt_token_ids) - 1 - num_computed_tokens
        if num_external_tokens < 0:
            num_external_tokens = 0
        return num_external_tokens, False

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

    def _truncate_mamba_token_ids(self, token_ids: list[int]) -> list[int]:
        """P-side: drop the last prompt token so the prefiller sends h(N-1).
        The decoder recomputes the last token to derive h(N).
        Only effective when self._has_mamba and token_ids has > 1 element."""
        if self._has_mamba and len(token_ids) > 1:
            return token_ids[:-1]
        return token_ids

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
                    token_ids=self._truncate_mamba_token_ids(
                        new_req.prompt_token_ids or []
                    ),
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
                # Skip requests that completed prefill in a single step
                if req_id not in self.chunked_prefill:
                    continue
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
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
                    token_ids=self._truncate_mamba_token_ids(prompt_token_ids),
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

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
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
