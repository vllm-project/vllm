# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific scheduler-side logic for the NIXL connector."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqId,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NixlPushConnectorScheduler(NixlBaseConnectorScheduler):
    """Push-specific scheduler logic (WRITE-based KV transfer).

    All P2P communication is deferred to the worker level via NIXL
    notifications. The scheduler communicates with workers only through
    the standard ``build_connector_meta`` / ``update_connector_output``
    hooks.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # D-side: registration data to pass to D workers via metadata.
        self._push_pending_registrations: dict[str, dict[str, Any]] = {}

        # P-side: block IDs for finished requests, for tracking/cleanup.
        self._finished_request_blocks: dict[ReqId, BlockIds] = {}
        # P-side: newly finished blocks to send to P workers on next step.
        self._newly_finished_push_blocks: dict[ReqId, BlockIds] = {}

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """In push mode, D doesn't pull — it registers blocks and waits.

        However, we still need to handle the do_remote_prefill case
        where D needs to know how many tokens will be pushed.
        """
        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            actual = self._mamba_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        """In push mode, D stores registration data for the worker to send
        to P via NIXL notification (deferred to build_connector_meta)."""
        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector update_state_after_alloc: "
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
                    unhashed_local_block_ids: BlockIds = (
                        blocks.get_unhashed_block_ids_all_groups()
                        if num_external_tokens > 0
                        else ()
                    )
                    local_block_ids = self.get_sw_clipped_blocks(
                        unhashed_local_block_ids
                    )

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
            elif num_external_tokens > 0:
                logger.info(
                    "KV PUSH mode: D node storing registration for request %s",
                    request.request_id,
                )
                local_block_ids = blocks.get_unhashed_block_ids_all_groups()
                local_block_ids = self.get_sw_clipped_blocks(local_block_ids)

                # Store registration data for the worker to send via NIXL.
                self._push_pending_registrations[request.request_id] = {
                    "request_id": params["remote_request_id"],
                    "decode_request_id": request.request_id,
                    "decode_engine_id": self.engine_id,
                    "decode_tp_size": (
                        self.vllm_config.parallel_config.tensor_parallel_size
                    ),
                    "local_block_ids": local_block_ids,
                    "remote_engine_id": params["remote_engine_id"],
                    "remote_host": params["remote_host"],
                    "remote_port": params["remote_port"],
                    "remote_tp_size": params["tp_size"],
                }

                # Track the request as needing recv (waiting for push).
                # Set remote_block_ids to empty — in push mode D doesn't
                # know P's blocks; P determines them from the registration.
                params["remote_block_ids"] = ()
                self._reqs_need_recv[request.request_id] = (
                    request,
                    local_block_ids,
                )
            else:
                assert num_external_tokens == 0
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    def request_finished(
        self,
        request: Request,
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Push-mode request_finished: stores blocks for workers."""
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NixlPushConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node

        self._stop_heartbeat(request.request_id)

        if params.get("do_remote_prefill"):
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None

        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            self._reqs_not_processed.add(request.request_id)
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        if delay_free_blocks:
            request_kv_blocks_ttl = self._kv_lease_duration
            if is_d_node:
                request_kv_blocks_ttl = self.decoder_kv_blocks_ttl
            logger.debug(
                "NixlPushConnector request_finished(%s) waiting for %d seconds "
                "before releasing blocks",
                request.request_id,
                request_kv_blocks_ttl,
            )
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + request_kv_blocks_ttl
            )

            block_ids = self.get_sw_clipped_blocks(block_ids)
            remote_num_tokens = request.num_computed_tokens

            # Store finished blocks for worker-level matching with D
            # registrations (via NIXL notifications).
            self._finished_request_blocks[request.request_id] = block_ids
            self._newly_finished_push_blocks[request.request_id] = block_ids

        return delay_free_blocks, dict(
            do_remote_prefill=is_p_node,
            do_remote_decode=is_d_node,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_num_tokens=remote_num_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, NixlConnectorMetadata)

        # D side: package registration data for D workers.
        meta.push_registrations = dict(self._push_pending_registrations)
        self._push_pending_registrations.clear()

        # P side: package newly finished blocks for P workers.
        meta.push_finished_blocks = dict(self._newly_finished_push_blocks)
        self._newly_finished_push_blocks.clear()

        return meta

    def has_pending_push_work(self) -> bool:
        return bool(self._finished_request_blocks)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Clean up finished request blocks after push completes."""
        super().update_connector_output(connector_output)
        for req_id in connector_output.finished_sending or ():
            self._finished_request_blocks.pop(req_id, None)
