# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific scheduler-side logic for the NIXL connector."""

import time
import uuid
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
    PeerEndpoint,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    LookupPinReply,
    LookupPinRequest,
    PeerKVIdentity,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_pins import (
    hashes_not_reproducible,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    kv_lease_extension,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# The client enforces its own timeout; this only bounds a request that would
# otherwise wait for a lookup whose result never shows up.
_LOOKUP_GRACE_S = 5.0


class NixlPullConnectorScheduler(NixlBaseConnectorScheduler):
    """Pull-specific scheduler logic (READ-based KV transfer)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).

        """
        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and (
            params.get("peer_lookup")
            or (params.get("_peer_hit") and params.get("do_remote_prefill"))
        ):
            return self._get_num_peer_matched_tokens(
                request, params, num_computed_tokens
            )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._get_remote_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        if (
            params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and all(
                p in params
                for p in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
                )
            )
        ):
            # Decode node has kv blocks for part of prefill request, so, provide them
            # as an external token count to scheduler.
            # The tokens will be loaded if not already present
            # in the prefill node local cache
            remote_num_tokens = params.get("remote_num_tokens") or 0
            count = (
                min(remote_num_tokens, request.num_prompt_tokens) - num_computed_tokens
            )
            if count > 0:
                # Check kv_recompute_threshold: skip pull if
                # remote tokens are below the threshold.
                if (
                    self.kv_recompute_threshold > 0
                    and count < self.kv_recompute_threshold
                ):
                    logger.debug(
                        "Skipping remote pull for %s: %d remote tokens < threshold %d",
                        request.request_id,
                        count,
                        self.kv_recompute_threshold,
                    )
                    return 0, False
                return count, True

        # No remote prefill for this request.
        return 0, False

    # ---- peer prefix lookup (consumer side) ------------------------------

    def _peer_lookup_precondition_failure(
        self, request: "Request", params: dict[str, Any]
    ) -> str | None:
        """Why this request cannot do a peer lookup, or None. No network."""
        if not params.get("do_remote_prefill"):
            # A hit is loaded as a remote prefill; without the flag the load
            # would never be queued and the request would wait on it forever.
            return "do_remote_prefill not set"
        if self._compatibility_hash is None:
            return "compatibility hash not known"
        hash_block_size = self.kv_cache_config.hash_block_size
        if hash_block_size != self.block_size:
            return "hash block size differs from block size"
        if len(self.kv_cache_config.kv_cache_groups) != 1:
            return "more than one KV cache group"
        if (
            params.get("remote_engine_id") is None
            or not params.get("remote_host")
            or params.get("remote_port") is None
        ):
            return "peer address missing"
        if request.skip_reading_external_kv:
            return "request does not read external KV"
        if request.num_prompt_tokens < self.block_size:
            return "no full prompt block"
        if hashes_not_reproducible(
            self.vllm_config.cache_config.prefix_caching_hash_algo
        ):
            return "block hashes are not reproducible across processes"
        return None

    def _peer_miss(
        self, request: "Request", params: dict[str, Any], reason: str
    ) -> tuple[int, bool]:
        """The peer has nothing for us: compute the whole prompt locally."""
        logger.debug("Peer lookup for %s missed: %s", request.request_id, reason)
        self.peer_lookup_stats.record_miss(reason)
        params["peer_lookup"] = False
        params["do_remote_prefill"] = False
        params["peer_miss_reason"] = reason
        return 0, False

    def _submit_peer_lookup(self, request: "Request", params: dict[str, Any]) -> None:
        num_blocks = request.num_prompt_tokens // self.block_size
        lookup = LookupPinRequest(
            block_hashes=[bytes(h) for h in request.block_hashes[:num_blocks]],
            identity=PeerKVIdentity(
                compatibility_hash=self._compatibility_hash or "",
                block_size=self.block_size,
                hash_algo=self.vllm_config.cache_config.prefix_caching_hash_algo,
                cache_salt=request.cache_salt,
            ),
            lease_duration=self._kv_lease_duration,
            lookup_id=uuid.uuid4().hex,
        )
        host, port = params["remote_host"], params["remote_port"]
        self._peer_endpoints[(host, port)] = PeerEndpoint(
            engine_id=params["remote_engine_id"],
            tp_size=params.get("tp_size"),
            dcp_size=params.get("dcp_size", 1),
            pp_size=params.get("pp_size", 1),
        )
        self.peer_lookup_stats.lookups += 1
        client = self._get_lookup_client()
        # The client probes the peer before the lookup, so allow for both.
        params["_peer_lookup_deadline"] = (
            time.monotonic() + client.max_latency_s + _LOOKUP_GRACE_S
        )
        client.submit(
            request.request_id,
            host,
            port,
            lookup,
            engine_id=params["remote_engine_id"],
        )

    def _peer_reply_is_consistent(
        self, request: "Request", reply: LookupPinReply
    ) -> bool:
        """A hit must be whole blocks of one group that we asked about."""
        if (
            reply.hit_tokens <= 0
            or not reply.remote_request_id
            or len(reply.remote_block_ids) != 1
        ):
            return False
        requested_blocks = request.num_prompt_tokens // self.block_size
        num_blocks = len(reply.remote_block_ids[0])
        return (
            reply.hit_tokens % self.block_size == 0
            and num_blocks == reply.hit_tokens // self.block_size
            and 0 < num_blocks <= requested_blocks
        )

    def _get_num_peer_matched_tokens(
        self,
        request: "Request",
        params: dict[str, Any],
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Peer lookup-and-pin, driven by the scheduler's repeated calls.

        Returns `(None, False)` while the lookup is out (the scheduler skips
        the request this step). Once it is answered, either the request becomes
        an ordinary remote prefill (`_peer_hit`) or it computes locally.
        """
        if params.get("peer_lookup"):
            if not params.get("_peer_lookup_submitted"):
                reason = self._peer_lookup_precondition_failure(request, params)
                if reason is not None:
                    logger.debug(
                        "No peer lookup for %s: %s", request.request_id, reason
                    )
                    return self._peer_miss(request, params, "local_precondition")
                self._submit_peer_lookup(request, params)
                params["_peer_lookup_submitted"] = True
                return None, False

            client = self._get_lookup_client()
            result = client.poll(request.request_id)
            if result is None:
                if time.monotonic() > params.get("_peer_lookup_deadline", 0.0):
                    client.cancel(request.request_id)
                    return self._peer_miss(request, params, "timeout")
                return None, False

            reply = result.reply
            if reply is None:
                return self._peer_miss(request, params, result.error or "bad_reply")
            if reply.hit_tokens <= 0 or not reply.remote_block_ids:
                return self._peer_miss(request, params, reply.miss_reason.value)
            if not self._peer_reply_is_consistent(request, reply):
                # Do not load blocks that do not line up with our prompt. The
                # pin, if any, is left to expire with its lease.
                if reply.remote_request_id:
                    self.peer_lookup_stats.orphans_unreleased += 1
                logger.warning(
                    "Ignoring inconsistent peer lookup reply for %s", request.request_id
                )
                return self._peer_miss(request, params, "bad_reply")

            logger.debug(
                "Peer lookup_pin hit for %s: %d tokens under pin %s",
                request.request_id,
                reply.hit_tokens,
                reply.remote_request_id,
            )
            stats = self.peer_lookup_stats
            if reply.hit_tokens == (request.num_prompt_tokens // self.block_size) * (
                self.block_size
            ):
                stats.full_hits += 1
            else:
                stats.partial_hits += 1
            # From here on the request is an ordinary remote prefill from the
            # pin: the stock update_state_after_alloc / request_finished /
            # heartbeat machinery frees the pin on A.
            params["remote_block_ids"] = reply.remote_block_ids
            params["remote_request_id"] = reply.remote_request_id
            params["remote_num_tokens"] = reply.hit_tokens
            params["remote_blocks_expiry_time"] = reply.remote_blocks_expiry_time
            params["remote_lease_extension"] = reply.remote_lease_extension
            params["peer_lookup"] = False
            params["_peer_hit"] = True
            self._register_heartbeat(request.request_id, params)

        load_end = min(
            params["remote_num_tokens"],
            self._get_remote_prefill_token_count(request.num_prompt_tokens),
        )
        count = load_end - num_computed_tokens
        if count <= 0 or (
            self.kv_recompute_threshold > 0 and count < self.kv_recompute_threshold
        ):
            # Nothing worth loading. do_remote_prefill stays set so that
            # update_state_after_alloc sends the notify-only recv that frees
            # the pin on A.
            return 0, False
        return count, True

    def _align_peer_blocks(
        self, params: dict[str, Any], blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """Cut A's pinned blocks down to what this request loads.

        The worker pairs the request's unhashed local blocks with the *tail* of
        `remote_block_ids`. A pinned more blocks than are loaded when the
        trailing prompt tokens are recomputed or the prompt is shorter than the
        hit, so the list must end exactly where the load ends.
        """
        local_tokens = (
            sum(
                block.block_hash is not None and not block.is_null
                for block in blocks.blocks[0]
            )
            * self.block_size
        )
        load_end = local_tokens + num_external_tokens
        num_blocks = cdiv(load_end, self.block_size)
        remote_block_ids = params["remote_block_ids"]
        assert all(len(group) >= num_blocks for group in remote_block_ids), (
            f"peer pin has {[len(g) for g in remote_block_ids]} blocks, "
            f"load needs {num_blocks}"
        )
        params["remote_block_ids"] = [group[:num_blocks] for group in remote_block_ids]
        params["remote_num_tokens"] = load_end

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("_peer_hit") and params.get("do_remote_prefill"):
            if num_external_tokens > 0:
                self._align_peer_blocks(params, blocks, num_external_tokens)
            else:
                # The local prefix covers the hit: the recv below only frees
                # the pin, so there is nothing left to keep alive.
                self.peer_lookup_stats.notify_only_releases += 1
                self._stop_heartbeat(request.request_id)

        if params.get("do_remote_decode") or (
            params.get("do_remote_prefill") and self.is_bidirectional_kv_xfer_enabled
        ):
            self._reqs_in_batch.add(request.request_id)
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.
            self._reqs_need_save[request.request_id] = request
        elif params.get("do_remote_prefill") or (
            params.get("do_remote_decode")
            and self.is_bidirectional_kv_xfer_enabled
            and not params.get("_remote_blocks_processed")
        ):
            if params.get("remote_block_ids"):
                if all(
                    p in params
                    for p in (
                        "remote_engine_id",
                        "remote_request_id",
                        "remote_host",
                        "remote_port",
                    )
                ):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the local node. We need to call
                    # send_notif in _read_blocks to free the memory on the remote node.

                    unhashed_local_block_ids: BlockIds = (
                        blocks.get_unhashed_block_ids_all_groups()
                        if num_external_tokens > 0
                        else ()
                    )
                    local_block_ids = self.get_exchange_clipped_blocks(
                        unhashed_local_block_ids
                    )
                    # Blocks covered by the local prefix cache, per KV cache group.
                    # Each count fixes where that group's DCP slice starts, which the
                    # worker needs to line up with the remote's slice.
                    local_num_computed_blocks = tuple(
                        len(
                            [
                                block
                                for block in group
                                if block.block_hash is not None and not block.is_null
                            ]
                        )
                        for group in blocks.blocks
                    )

                    # Get unhashed blocks to pull from remote. Mind that a full prefix
                    # cache hit is indicated with an empty list.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                        local_num_computed_blocks,
                        # Parked in WAITING_FOR_REMOTE_KVS only when there is
                        # something to pull; a full local hit stays RUNNING.
                        num_external_tokens > 0,
                    )

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer",
                        params,
                    )
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    def request_finished(
        self,
        request: "Request",
        block_ids: "BlockIds",
    ) -> tuple[bool, dict[str, Any] | None]:
        """Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node

        # Stop heartbeating for aborted requests that never reached finished_recving:
        # normal path cleans up in update_connector_output.
        self._stop_heartbeat(request.request_id)

        if params.get("peer_lookup"):
            # The lookup is pending or was never polled: there is no pin to
            # free here. One that arrives later is released as an orphan.
            if self._lookup_client is not None:
                self._lookup_client.cancel(request.request_id)
            self.peer_lookup_stats.cancelled += 1
            params["peer_lookup"] = False
            params["do_remote_prefill"] = False
            return False, None

        if params.get("do_remote_prefill"):
            if params.get("_peer_hit"):
                self.peer_lookup_stats.notify_only_releases += 1
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled, e.g. via the
            # abort_immediately path used to clean up KV-transfer requests
            # rejected at the D-side serving layer).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [], (), False)
            params["do_remote_prefill"] = False
            return False, None

        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None

        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            # Clear _reqs_need_save if a request is aborted as partial prefill.
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        blocks_expiry_time = None
        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            request_kv_blocks_ttl = self._kv_lease_duration
            if is_d_node:
                # For blocks pinned on D, use a simpler timeout for now instead of a
                # lease mechanism as turn2 request is client-driven.
                request_kv_blocks_ttl = self.decoder_kv_blocks_ttl
            logger.debug(
                "NIXLConnector request_finished(%s) waiting for %d seconds "
                "before releasing blocks",
                request.request_id,
                request_kv_blocks_ttl,
            )
            # Lets the consumer decline reads of recycled blocks. P blocks: the
            # initial lease, extended by heartbeats; D blocks: the turn-2 TTL.
            blocks_expiry_time = time.perf_counter() + request_kv_blocks_ttl
            self._reqs_need_send[request.request_id] = blocks_expiry_time
            # NOTE HMA will "mark" empty/null blocks in groups with 0s (eg SWA ones),
            # trimming down after allocating for the whole sequence length. Empty
            # blocks are always at the start of the list.
            # Here we "unpad" blocks to send the actual remote blocks to be read.
            block_ids = self.get_exchange_clipped_blocks(block_ids)

            remote_num_tokens = request.num_computed_tokens

        return delay_free_blocks, dict(
            do_remote_prefill=is_p_node,
            do_remote_decode=is_d_node,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.transfer_tp_size,
            dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            pp_size=self.vllm_config.parallel_config.pipeline_parallel_size,
            remote_num_tokens=remote_num_tokens,
            remote_blocks_expiry_time=blocks_expiry_time,
            # Heartbeat renewal amount; turn-2 D blocks have a fixed TTL.
            remote_lease_extension=(
                kv_lease_extension(self._kv_lease_duration) if is_p_node else None
            ),
            remote_registration_generation=self._registration_generation,
            transfer_mode=self._TRANSFER_MODE,
        )
