# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific scheduler-side logic for the NIXL connector."""

import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqId,
    ReqMeta,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NixlPullConnectorScheduler(NixlBaseConnectorScheduler):
    """Pull-specific scheduler logic (READ-based KV transfer)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.seed_first_token = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "seed_first_token", False
            )
        )

        # Concurrent P/D dispatch: max time a decode "prepare" request may
        # stay parked/armed waiting for the KV-ready notification; on expiry
        # it falls back to a local prefill.
        self.prepare_ready_timeout_s = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "prepare_ready_timeout_s", 10.0
            )
        )
        # Early-arm: allocate a prepare request's blocks on the first
        # schedule pass and let the KV-ready notification start the pull
        # directly from the bridge thread (blocks pinned during prefill).
        self.pd_early_arm_enabled = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "pd_early_arm", False
            )
        )
        # Early-arm state, shared between the engine thread and the bridge
        # notify thread; dict/list ops are GIL-atomic and ownership is
        # handed over via dict.pop.
        # Armed prepare requests: blocks allocated, pull deferred.
        self._pd_armed: dict[ReqId, tuple[Request, BlockIds]] = {}
        # KV-ready params that arrived before their request was armed.
        self._pd_pending_ready: dict[str, tuple[dict[str, Any], float]] = {}
        # Armed pulls riding the next step's metadata as slow-path backup.
        self._pd_ready_backup: list[tuple[ReqId, ReqMeta]] = []

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed
              (None to have the scheduler query again later).
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

        if params is not None and params.get("do_remote_prefill"):
            if not params.get("remote_block_ids") and (
                params.get("remote_ready") is False
            ):
                # Concurrent P/D dispatch: prepare request, remote KV
                # metadata not arrived yet.
                pending = self._pd_take_pending_ready(request.request_id)
                if pending is not None:
                    # Ready notification raced ahead; merge and pull now.
                    self._pd_merge_ready_params(params, pending)
                elif self.pd_early_arm_enabled:
                    # Early-arm: allocate blocks via the normal async-load
                    # path; update_state_after_alloc arms the request.
                    params.setdefault("_pd_parked_ts", time.monotonic())
                else:
                    # Late-alloc parking: None skips the request in
                    # schedule() without allocating blocks.
                    now = time.monotonic()
                    parked_ts = params.setdefault("_pd_parked_ts", now)
                    if now - parked_ts > self.prepare_ready_timeout_s:
                        logger.warning(
                            "PD prepare request %s waited %.1fs for the "
                            "KV-ready notification; falling back to local "
                            "prefill.",
                            request.request_id,
                            now - parked_ts,
                        )
                        params["do_remote_prefill"] = False
                        return 0, False
                    return None, False
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._get_remote_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

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

                    # Get unhashed blocks to pull from remote. Mind that a full prefix
                    # cache hit is indicated with an empty list.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                    )

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer",
                        params,
                    )
            elif self.pd_early_arm_enabled and params.get("remote_ready") is False:
                # Early-arm: stash the allocated blocks for on_pd_kv_ready.
                unhashed_local_block_ids = (
                    blocks.get_unhashed_block_ids_all_groups()
                    if num_external_tokens > 0
                    else ()
                )
                local_block_ids = self.get_exchange_clipped_blocks(
                    unhashed_local_block_ids
                )
                pending = self._pd_take_pending_ready(request.request_id)
                if pending is not None:
                    # Ready already landed: pull with this very step.
                    self._pd_merge_ready_params(params, pending)
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                    )
                else:
                    self._pd_armed[request.request_id] = (request, local_block_ids)
                    params["_pd_armed_ts"] = time.monotonic()
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    @staticmethod
    def _pd_merge_ready_params(
        params: dict[str, Any], ready_params: dict[str, Any]
    ) -> None:
        """Merge the prefill-side kv_transfer_params and mark remote ready."""
        params.update(ready_params)
        params["remote_ready"] = True

    def _pd_take_pending_ready(self, req_id: ReqId) -> dict[str, Any] | None:
        """Pop a stashed KV-ready notification matching this internal
        request id (internal ids embed the router-side raw id, hence the
        substring match). Engine thread only."""
        if not self._pd_pending_ready:
            return None
        for raw_id in list(self._pd_pending_ready):
            if raw_id in req_id:
                params, _ = self._pd_pending_ready.pop(raw_id)
                return params
        return None

    def pd_cancel_armed(self, req_id: ReqId) -> bool:
        """Atomically claim an armed entry (timeout/abort path); False if
        on_pd_kv_ready already claimed it and the pull is going ahead."""
        return self._pd_armed.pop(req_id, None) is not None

    def on_pd_kv_ready(
        self, raw_request_id: str, kv_transfer_params: dict[str, Any]
    ) -> tuple[ReqId, ReqMeta] | None:
        """Deliver the prefill's kv_transfer_params for an armed prepare
        request and build its pull metadata immediately.

        Called from the bridge notify thread (or the UTILITY RPC fallback).
        The armed entry is claimed with an atomic dict.pop that arbitrates
        against the armed-timeout fallback. Returns (req_id, ReqMeta) when
        the pull was armed (caller may fast-publish it; the next step's
        metadata re-delivers it as backup), else None (params stashed for
        the engine thread).
        """
        if not self.pd_early_arm_enabled:
            return None
        armed_req_id = None
        for req_id in list(self._pd_armed):
            if raw_request_id in req_id:
                armed_req_id = req_id
                break
        entry = (
            self._pd_armed.pop(armed_req_id, None) if armed_req_id is not None else None
        )
        if entry is None:
            # Not armed yet or claimed by the timeout path: stash and prune.
            now = time.monotonic()
            self._pd_pending_ready[raw_request_id] = (kv_transfer_params, now)
            for raw_id, (_, ts) in list(self._pd_pending_ready.items()):
                if now - ts > 60.0:
                    self._pd_pending_ready.pop(raw_id, None)
            return None
        request, local_block_ids = entry
        params = request.kv_transfer_params
        assert params is not None
        self._pd_merge_ready_params(params, kv_transfer_params)
        # The pull is triggered right here; keep the flag False or
        # request_finished would queue a spurious empty recv.
        params["do_remote_prefill"] = False
        params.pop("_pd_armed_ts", None)
        if not all(
            params.get(p) is not None
            for p in (
                "remote_block_ids",
                "remote_engine_id",
                "remote_request_id",
                "remote_host",
                "remote_port",
            )
        ):
            logger.warning(
                "Got invalid KV-ready params for %s: %s; re-arming for the "
                "timeout fallback (local prefill).",
                armed_req_id,
                kv_transfer_params,
            )
            params["_pd_armed_ts"] = time.monotonic()
            self._pd_armed[armed_req_id] = (request, local_block_ids)
            return None
        helper = NixlConnectorMetadata()
        helper.add_new_req_to_recv(
            request_id=armed_req_id,
            local_block_ids=local_block_ids,
            kv_transfer_params=params,
        )
        req_meta = helper.reqs_to_recv[armed_req_id]
        self._pd_ready_backup.append((armed_req_id, req_meta))
        return armed_req_id, req_meta

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        # Early-arm: slow-path backup of the bridge's fast publish.
        if self._pd_ready_backup:
            assert isinstance(meta, NixlConnectorMetadata)
            while self._pd_ready_backup:
                req_id, req_meta = self._pd_ready_backup.pop()
                meta.reqs_to_recv.setdefault(req_id, req_meta)
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: "BlockIds",
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
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

        # Early-arm: a late KV-ready notification must not start a pull
        # into freed blocks.
        self._pd_armed.pop(request.request_id, None)

        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node

        # Stop heartbeating for aborted requests that never reached finished_recving:
        # normal path cleans up in update_connector_output.
        self._stop_heartbeat(request.request_id)

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled, e.g. via the
            # abort_immediately path used to clean up KV-transfer requests
            # rejected at the D-side serving layer).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
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
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + request_kv_blocks_ttl
            )
            if is_d_node:
                # D blocks expiry time exported for the turn-2 readback.
                blocks_expiry_time = self._reqs_need_send[request.request_id]
            # NOTE HMA will "mark" empty/null blocks in groups with 0s (eg SWA ones),
            # trimming down after allocating for the whole sequence length. Empty
            # blocks are always at the start of the list.
            # Here we "unpad" blocks to send the actual remote blocks to be read.
            block_ids = self.get_exchange_clipped_blocks(block_ids)

            remote_num_tokens = request.num_computed_tokens

        params = dict(
            do_remote_prefill=is_p_node,
            do_remote_decode=is_d_node,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_num_tokens=remote_num_tokens,
            remote_blocks_expiry_time=blocks_expiry_time,
        )
        if self.seed_first_token and is_p_node and request.num_output_tokens > 0:
            params["first_token_ids"] = list(request.output_token_ids)
        return delay_free_blocks, params
