# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake pull connector: the decoder reads the prefill KV cache itself.

`MooncakeConnector` has the prefiller write into the decoder's registered
memory, so every decode batch carries the decoder's KV region table to the
prefiller. This connector turns the transfer around: the decoder asks each
paired prefiller worker for its registered regions (once per peer, then cached)
and for the block ids of the requests it is about to read, then reads the bytes
with `batch_transfer_sync_read` at its own scheduling pace. The prefiller holds
the blocks until every reader reports the read as finished.

Both connectors share the scheduler, the request metadata, the registered
regions and the response envelope; only the transfer direction and the
side-channel messages differ.
"""

import asyncio
import time
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

import msgspec
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    BlockTransferRequest,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    ReqId,
    SendBlockMeta,
    TransferId,
    TransferRegion,
    _compute_sender_transfer_plan,
    _validate_asymmetric_region_lengths,
    plan_block_descriptors,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.pull_protocol import (
    MooncakePullDone,
    MooncakePullReading,
    MooncakePullRequest,
    PullMessage,
    ReaderKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.read_leases import (
    ReadLeases,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def _consume_task_result(task: asyncio.Task) -> None:
    """Read the result of a task nobody awaits, so it is not reported twice."""
    if task.done() and not task.cancelled():
        task.exception()


class MooncakePullConnector(MooncakeConnector):
    """Mooncake P2P connector in READ mode.

    Same scheduler, metadata and registered regions as `MooncakeConnector`;
    only the worker differs, because the decoder reads the prefiller's KV cache
    instead of the prefiller writing into the decoder's memory.
    """

    def _build_worker(
        self, vllm_config: VllmConfig, kv_cache_config: "KVCacheConfig"
    ) -> "MooncakePullConnectorWorker":
        assert self.engine_id is not None
        return MooncakePullConnectorWorker(vllm_config, self.engine_id, kv_cache_config)


class MooncakePullConnectorWorker(MooncakeConnectorWorker):
    """Worker-side implementation of the pull connector.

    On the prefiller the worker holds KV blocks and hands out its registered
    regions and block ids. On the decoder it reads those blocks into its own
    cache and reports completion, which releases the prefiller's blocks.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        self._pull_message_decoder = msgspec.msgpack.Decoder(PullMessage)

        # Prefiller side: the leases of the readers that were handed block ids.
        self._leases = ReadLeases()

        # Decoder side: peer worker address -> the regions and layout it
        # reported, kept until that peer reports another layout.
        self._peer_layouts: dict[str, MooncakeXferMetadata] = {}

        if self.is_kv_producer:
            return
        # The read blocks its calling thread, so it runs on a pool. Each pool
        # thread must be bound to the correct device, because that binding is
        # thread-local.
        self._reader_executor = ThreadPoolExecutor(
            max_workers=self.num_sender_workers,
            thread_name_prefix="vllm-mooncake-reader",
            initializer=self._bind_sender_thread_device,
        )

    def shutdown(self):
        reader_executor = getattr(self, "_reader_executor", None)
        if reader_executor is not None:
            reader_executor.shutdown(wait=False)
        # A socket parked in a receive never lets the context terminate, and the
        # reader holds one open, so close every socket before the base class
        # stops the loops.
        self.async_zmq_ctx.destroy(linger=0)
        super().shutdown()

    ############################################################
    # Prefiller side: hand out blocks and registered regions
    ############################################################

    def _resolve_expected_readers(
        self, send_meta: SendBlockMeta, reader_tp_size: int, reader_pp_size: int
    ) -> None:
        """How many reader workers a set of blocks is served to."""
        expected = len(self.transfer_topo.handshake_target_ranks(reader_tp_size))
        if reader_pp_size > 1 and reader_pp_size != self.pp_size:
            # Every reader PP stage reads every producer stage, including peers
            # with no shared layers.
            expected *= reader_pp_size
        if send_meta.need_send and send_meta.need_send != expected:
            raise RuntimeError(
                f"Transfer {send_meta.transfer_id} is read by "
                f"{send_meta.need_send} workers, but the reader topology "
                f"(tp={reader_tp_size}, pp={reader_pp_size}) implies {expected}."
            )
        send_meta.need_send = expected

    def _complete_transfer_if_read_by_all(self, send_meta: SendBlockMeta) -> None:
        if not send_meta.p_req_id:
            # The prefiller's own scheduler has not registered this transfer
            # yet; `record_send_reqs` completes it once it does.
            return
        if not send_meta.need_send:
            return
        if self._leases.reported(send_meta.transfer_id) < send_meta.need_send:
            return
        transfer_id = send_meta.transfer_id
        if self.reqs_need_send.pop(transfer_id, None) is None:
            return
        self._leases.forget(transfer_id)
        self.finished_sending_reqs.add(send_meta.p_req_id)

    def _mark_reader_done(
        self, send_meta: SendBlockMeta, readers_key: ReaderKey
    ) -> None:
        self._leases.report(send_meta.transfer_id, readers_key)
        self._complete_transfer_if_read_by_all(send_meta)

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        """Report released transfers, and forget the ones already released.

        A transfer can leave `reqs_need_send` through completion or through the
        prefiller's expiry. The lease bookkeeping has to follow both, or a
        released transfer would keep looking like one a reader still holds.
        """
        finished = await super().fetch_finished_sending_reqs()
        for transfer_id in self._leases:
            if transfer_id not in self.reqs_need_send:
                self._leases.forget(transfer_id)
        return finished

    def _send_idle(self, transfer_id: TransferId) -> bool:
        """Whether the blocks of a transfer are free to be released."""
        return self._leases.may_release(
            transfer_id, envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
        )

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata) -> None:
        """Register the prefiller's finished requests, then re-check completion.

        A reader with a full local prefix hit reports completion before the
        prefiller's own scheduler registers the transfer, so completion has to
        be re-checked once the blocks show up.
        """
        await super().record_send_reqs(metadata)
        for send_meta in list(self.reqs_need_send.values()):
            self._complete_transfer_if_read_by_all(send_meta)

    async def _handle_side_channel_message(
        self, identity: bytes, payload: bytes, sock: zmq.asyncio.Socket
    ) -> None:
        try:
            message = self._pull_message_decoder.decode(payload)
        except msgspec.DecodeError as e:
            # Name the mismatch for the peer instead of handing it a raw
            # decoding failure.
            raise RuntimeError(
                "The peer sent a message that is not a Mooncake pull message, "
                "so the push and pull connectors are mixed on this P/D pair: "
                f"{e}"
            ) from e
        if isinstance(message, MooncakePullRequest):
            await self._handle_pull_request(identity, sock, message)
        elif isinstance(message, MooncakePullReading):
            self._handle_read_lease(message)
        else:
            self._handle_read_done(message)

    async def _wait_for_blocks(
        self, d_req_id: ReqId, send_meta: SendBlockMeta
    ) -> tuple[ReqId, SendBlockMeta]:
        """Wait for the prefiller's blocks, or for the request being dropped."""
        wait_tasks = {
            asyncio.create_task(send_meta.ready.wait()),
            asyncio.create_task(send_meta.aborted.wait()),
        }
        try:
            await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in wait_tasks:
                task.cancel()
        return d_req_id, send_meta

    async def _handle_pull_request(
        self,
        identity: bytes,
        sock: zmq.asyncio.Socket,
        request: MooncakePullRequest,
    ) -> None:
        readers_key: ReaderKey = (request.engine_id, request.tp_rank, request.pp_rank)

        # Handing out block ids lets the reader slice this worker's cache by KV
        # head, so the reader's TP degree has to be a layout this cache can
        # serve. Only this side holds the registered layer specs the check
        # needs, so it runs here.
        head_err = self._validate_head_resharding_layout(
            request.tp_size, self._local_transfer_regions()
        )
        if head_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR, err_msg=head_err
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return

        resolved: dict[ReqId, tuple[TransferId, list[list[int]]]] = {}
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        pending: list[tuple[ReqId, SendBlockMeta]] = []

        for d_req_id, (transfer_id, needs_data) in request.reqs.items():
            send_meta = self.reqs_need_send.get(transfer_id)
            if send_meta is None:
                # This reader may pair with the prefiller rank before its
                # scheduler registered the request; hold it until the blocks
                # show up.
                send_meta = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
                self.reqs_need_send[transfer_id] = send_meta
            self._resolve_expected_readers(send_meta, request.tp_size, request.pp_size)

            if not needs_data:
                # A reader with a full local prefix hit only needs the
                # prefiller to release the request, which happens as soon as
                # it is done.
                resolved[d_req_id] = (transfer_id, [])
                self._mark_reader_done(send_meta, readers_key)
                continue
            pending.append((d_req_id, send_meta))

        # One waiter per request that still needs its blocks, reused across
        # responses, so a batch that becomes ready piecemeal never piles up
        # waiters for requests it is already waiting on.
        waiters: dict[asyncio.Task, tuple[ReqId, SendBlockMeta]] = {
            asyncio.create_task(self._wait_for_blocks(d_req_id, send_meta)): (
                d_req_id,
                send_meta,
            )
            for d_req_id, send_meta in pending
        }
        include_regions = request.want_regions
        try:
            while True:
                status = (
                    MooncakeXferResponseStatus.FINISH
                    if not waiters
                    else MooncakeXferResponseStatus.CONTINUE
                )
                response = MooncakeXferResponse(
                    status=status,
                    layout_id=self._kv_layout_id,
                    metadata=self._build_xfer_metadata(
                        resolved, include_regions=include_regions
                    ),
                    err_reqs=err_reqs or None,
                    err_msg=err_msg,
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))

                # The registered regions travel with the first response only.
                include_regions = False
                resolved = {}
                err_reqs = []
                err_msg = None
                if not waiters:
                    break

                done, _ = await asyncio.wait(
                    waiters,
                    timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    err_reqs.extend(d_req_id for d_req_id, _ in waiters.values())
                    if err_msg is None:
                        err_msg = "Timeout waiting for P side ready."
                    for task in waiters:
                        task.cancel()
                    waiters.clear()
                    continue

                for task in done:
                    d_req_id, send_meta = waiters.pop(task)
                    if send_meta.aborted.is_set():
                        err_reqs.append(d_req_id)
                        if err_msg is None:
                            err_msg = (
                                "The prefiller released this request before "
                                "handing out its blocks."
                            )
                        continue
                    resolved[d_req_id] = (
                        send_meta.transfer_id,
                        send_meta.local_block_ids,
                    )
                    self._leases.hand_out(send_meta.transfer_id, readers_key)
        finally:
            for task in waiters:
                task.cancel()

    def _handle_read_lease(self, message: MooncakePullReading) -> None:
        """Extend the lease of transfers this reader is still reading.

        The prefiller reclaims blocks from a reader it has not heard from within
        the abort timeout, so a reader whose engine call outlives that timeout
        reports in while it waits. Nothing is answered here: the reader's socket
        only carries requests that expect a response.
        """
        readers_key: ReaderKey = (message.engine_id, message.tp_rank, message.pp_rank)
        for transfer_id in message.transfer_ids:
            self._leases.renew(transfer_id, readers_key)

    def _handle_read_done(self, message: MooncakePullDone) -> None:
        """Release the blocks a reader reported, and forget the transfers.

        Nothing is answered: the reader needs no acknowledgement to credit its
        own requests, and a reply would share its socket with the responses of
        the next pull of the same batch.
        """
        readers_key: ReaderKey = (message.engine_id, message.tp_rank, message.pp_rank)
        for transfer_id in message.transfer_ids:
            send_meta = self.reqs_need_send.get(transfer_id)
            if send_meta is None:
                # Already released: every reader of this transfer reported in.
                continue
            if message.failed_transfers and transfer_id in message.failed_transfers:
                logger.warning(
                    "Reader %s failed to read transfer %s; releasing its blocks.",
                    readers_key,
                    transfer_id,
                )
            self._mark_reader_done(send_meta, readers_key)

    ############################################################
    # Decoder side: read the peer's blocks
    ############################################################

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        """Read this peer's share of the batch, and settle it when it is over.

        Settling means crediting every request of the batch once, which is what
        lets the scheduler reuse the decoder's blocks. The engine call runs in a
        thread that cannot be cancelled, so cancelling this coroutine must not
        settle: the read would keep writing into blocks another request may
        already own. The work therefore runs in its own shielded task and is
        settled from its completion, whatever happens to this coroutine.
        """
        work = asyncio.create_task(self._read_from_peer(worker_addr, pull_metas))
        deferred = False
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            # The peer is still reading, so its report has to come from the work
            # itself: the blocks of this batch stay leased until then.
            self._fail_requests(pull_metas, set(pull_metas), "read cancelled")
            work.add_done_callback(partial(self._settle_deferred, pull_metas))
            deferred = True
            raise
        finally:
            if not deferred:
                self._settle_batch(pull_metas)

    def _settle_deferred(
        self, pull_metas: dict[ReqId, PullReqMeta], task: asyncio.Task
    ) -> None:
        """Settle a batch whose read outlived the coroutine that started it."""
        _consume_task_result(task)
        self._settle_batch(pull_metas)

    def _settle_batch(self, pull_metas: dict[ReqId, PullReqMeta]) -> None:
        for pull_meta in pull_metas.values():
            self._finish_reader_task(pull_meta)

    async def _read_from_peer(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
        request = MooncakePullRequest(
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            reqs={
                req_id: (
                    pull_meta.transfer_id,
                    any(len(group) > 0 for group in pull_meta.local_block_ids),
                )
                for req_id, pull_meta in pull_metas.items()
            },
            want_regions=worker_addr not in self._peer_layouts,
        )
        encoded_data = self._encoder.encode(request)
        logger.debug(
            "Reading kv_caches for %s from producer worker %s", req_ids, worker_addr
        )

        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let the prefiller wait for its own
                # timeout before we give up on the request.
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    response = self._xfer_resp_decoder.decode(await sock.recv())
                    if not await self._handle_pull_response(
                        sock, worker_addr, pull_metas, response
                    ):
                        return
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except msgspec.DecodeError as e:
            self._fail_requests(
                pull_metas,
                req_ids,
                f"undecodable response from {worker_addr}: {e}. The peer may run "
                "another Mooncake connector mode than this decoder.",
            )
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            self._fail_requests(pull_metas, req_ids, f"pull failed: {e}")

    def _fail_requests(
        self,
        pull_metas: dict[ReqId, PullReqMeta],
        req_ids: Collection[ReqId],
        reason: str,
    ) -> None:
        """Mark requests failed without reporting them to the scheduler yet."""
        failed: list[ReqId] = []
        for req_id in req_ids:
            pull_meta = pull_metas.get(req_id)
            if pull_meta is None or pull_meta.failed:
                continue
            pull_meta.failed = True
            failed.append(req_id)
            self.xfer_stats.record_failed_recv()
        if failed:
            logger.error("pulling kv_caches for %s failed: %s", failed, reason)

    def _finish_reader_task(self, pull_meta: PullReqMeta) -> None:
        """Credit one peer's read and report the request once all peers are done."""
        pull_meta.pull_tasks_count -= 1
        if pull_meta.pull_tasks_count > 0:
            return
        if not any(pull_meta.local_block_ids):
            # An empty pull only releases the prefiller's blocks; this decoder
            # never entered WAITING_FOR_REMOTE_KVS.
            return
        if pull_meta.failed:
            if self._is_hma_required:
                self._failed_recv_reqs.put(pull_meta.d_req_id)
            else:
                self._invalid_block_ids.put(
                    {block for group in pull_meta.local_block_ids for block in group}
                )
        self.finished_recving_reqs.add(pull_meta.d_req_id)

    async def _abandon_batch(
        self,
        sock: zmq.asyncio.Socket,
        pull_metas: dict[ReqId, PullReqMeta],
        reason: str,
    ) -> None:
        """Fail this batch locally, then release the blocks the peer holds.

        The prefiller hands out block ids before the decoder can validate the
        response, so it holds those blocks until this reader reports in. An
        abandoned batch has to report as well, or the blocks would wait for the
        prefiller's own timeout.
        """
        self._fail_requests(pull_metas, set(pull_metas), reason)
        transfer_ids = [meta.transfer_id for meta in pull_metas.values()]
        done = MooncakePullDone(
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            transfer_ids=transfer_ids,
            failed_transfers=transfer_ids,
        )
        await sock.send(self._encoder.encode(done))

    async def _handle_pull_response(
        self,
        sock: zmq.asyncio.Socket,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
        response: MooncakeXferResponse,
    ) -> bool:
        """Act on one prefiller response; False ends this peer's read."""
        if response.status == MooncakeXferResponseStatus.ERROR:
            await self._abandon_batch(
                sock, pull_metas, response.err_msg or "transfer error"
            )
            return False

        meta = response.metadata
        if meta is None:
            await self._abandon_batch(
                sock,
                pull_metas,
                f"producer worker {worker_addr} answered without its KV layout",
            )
            return False

        cached = self._peer_layouts.get(worker_addr)
        if cached is None:
            if not meta.kv_caches_base_addr:
                await self._abandon_batch(
                    sock,
                    pull_metas,
                    f"producer worker {worker_addr} did not send its KV layout, "
                    "so its blocks cannot be addressed",
                )
                return False
            layout_err, _, _ = self._prepare_layout(meta)
            if layout_err is not None:
                await self._abandon_batch(sock, pull_metas, layout_err)
                return False
            self._peer_layouts[worker_addr] = meta
        elif cached.layout_id != meta.layout_id:
            # Drop the stale layout so the recomputed request asks for the new
            # one instead of failing the same way again.
            self._peer_layouts.pop(worker_addr)
            await self._abandon_batch(
                sock,
                pull_metas,
                f"producer worker {worker_addr} changed its KV layout "
                f"({cached.layout_id} -> {meta.layout_id}); it restarted and the "
                "KV of this request must be recomputed",
            )
            return False
        else:
            # Later waves carry only the blocks, so the regions come from the
            # layout this peer already reported.
            meta = msgspec.structs.replace(cached, req_blocks=meta.req_blocks)

        if meta.req_blocks:
            await self._read_batch(sock, worker_addr, meta, pull_metas)

        if response.err_reqs:
            self._fail_requests(
                pull_metas, response.err_reqs, response.err_msg or "unknown error"
            )
        return True

    def _local_transfer_regions(self) -> list[TransferRegion]:
        registered_layers = len(self.registered_layer_names)
        shared_groups = (
            None
            if len(self.region_shared_groups) != registered_layers
            else self.region_shared_groups
        )
        row_offsets = (
            None
            if len(self.region_row_offsets) != registered_layers
            else self.region_row_offsets
        )
        return self._get_transfer_regions(
            base_addrs=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            kv_block_lens=self.kv_block_len_per_layer,
            layer_names=self.registered_layer_names,
            layer_indices=self.registered_layer_indices,
            group_indices=self.registered_group_indices,
            shared_group_ids=shared_groups,
            row_offsets=row_offsets,
        )

    def _total_num_kv_heads(self) -> int | None:
        if self.use_mla or self.kv_cache_config.has_mamba_layers:
            return None
        return self.transfer_topo.total_num_kv_heads

    def _prepare_layout(
        self, meta: MooncakeXferMetadata
    ) -> tuple[str | None, list[TransferRegion], list[TransferRegion]]:
        """Prepare this peer's regions and say why they cannot describe a read."""
        consumer_regions, producer_regions, prepare_err = (
            self._prepare_transfer_regions(meta)
        )
        if prepare_err is not None:
            return prepare_err, [], []
        validation_err = _validate_asymmetric_region_lengths(
            local_regions=producer_regions,
            remote_regions=consumer_regions,
            local_tp_size=meta.remote_tp_size,
            remote_tp_size=self.tp_size,
            producer_cache_replicated=meta.producer_cache_replicated,
            total_num_kv_heads=self._total_num_kv_heads(),
        )
        return validation_err, consumer_regions, producer_regions

    def _plan_reads(
        self,
        meta: MooncakeXferMetadata,
        pull_metas: dict[ReqId, PullReqMeta],
        resolved: dict[ReqId, tuple[TransferId, list[list[int]]]],
    ) -> tuple[list[int], list[int], list[int], list[ReqId], str | None]:
        """Plan the reads of one response batch.

        Returns:
            Local pointers to read into, remote pointers to read from, lengths,
            requests that failed planning, and the failure reason.

        """
        validation_err, consumer_regions, producer_regions = self._prepare_layout(meta)
        if validation_err is not None:
            return [], [], [], list(resolved), validation_err

        total_num_kv_heads = self._total_num_kv_heads()

        requests = [
            BlockTransferRequest(
                key=d_req_id,
                src_block_ids=producer_block_ids,
                dst_block_ids=pull_metas[d_req_id].local_block_ids,
            )
            for d_req_id, (_, producer_block_ids) in resolved.items()
            if d_req_id in pull_metas
        ]

        def plan(
            producer_region: TransferRegion, consumer_region: TransferRegion
        ) -> tuple[bool, int, int, int]:
            # Plan the copy from the prefiller's side: it owns the bytes.
            return _compute_sender_transfer_plan(
                local_tp_rank=meta.remote_tp_rank,
                local_tp_size=meta.remote_tp_size,
                remote_tp_rank=self.tp_rank,
                remote_tp_size=self.tp_size,
                local_kv_block_len=producer_region.kv_block_len,
                remote_kv_block_len=consumer_region.kv_block_len,
                producer_cache_replicated=meta.producer_cache_replicated,
                total_num_kv_heads=total_num_kv_heads,
            )

        # The prefiller holds the bytes, so its blocks are the source side.
        remote_ptrs, local_ptrs, lengths, err_reqs, err_msg = plan_block_descriptors(
            requests,
            src_regions=producer_regions,
            dst_regions=consumer_regions,
            group_specs=self.kv_cache_config.transfer_groups,
            blocks_per_logical=meta.physical_blocks_per_logical,
            plan=plan,
        )
        return local_ptrs, remote_ptrs, lengths, err_reqs, err_msg

    async def _read_batch(
        self,
        sock: zmq.asyncio.Socket,
        worker_addr: str,
        meta: MooncakeXferMetadata,
        pull_metas: dict[ReqId, PullReqMeta],
    ) -> None:
        """Read one response batch and report it back to the prefiller."""
        resolved = meta.req_blocks
        (
            local_ptrs,
            remote_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = self._plan_reads(meta, pull_metas, resolved)

        failed_transfers: list[TransferId] = []
        if err_reqs:
            failed_set = set(err_reqs)
            self._fail_requests(pull_metas, err_reqs, err_msg or "read planning failed")
            failed_transfers.extend(
                transfer_id
                for d_req_id, (transfer_id, _) in resolved.items()
                if d_req_id in failed_set
            )

        if local_ptrs:
            remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
            read_over = asyncio.Event()
            loop = asyncio.get_running_loop()

            def read_blocks() -> int:
                try:
                    return self._read_blocks(
                        remote_session, local_ptrs, remote_ptrs, lengths
                    )
                finally:
                    # The lease covers the engine call, and the call outlives the
                    # await below, so the thread ends the lease, not the await.
                    loop.call_soon_threadsafe(read_over.set)

            lease_task = asyncio.create_task(
                self._hold_read_lease(
                    worker_addr,
                    [transfer_id for transfer_id, _ in resolved.values()],
                    read_over,
                )
            )
            # A cancelled await leaves the engine call running, so this task
            # can outlive the coroutine that would otherwise await it.
            lease_task.add_done_callback(_consume_task_result)
            ret_value = await loop.run_in_executor(self._reader_executor, read_blocks)
            # Waiting for the lease orders the last renewal before the report.
            await asyncio.gather(lease_task, return_exceptions=True)
            if ret_value != 0:
                # Every request in this batch shares the failed engine call.
                read_reqs = [
                    d_req_id
                    for d_req_id, (_, producer_blocks) in resolved.items()
                    if any(len(group) > 0 for group in producer_blocks)
                    and d_req_id in pull_metas
                ]
                read_req_set = set(read_reqs)
                self._fail_requests(
                    pull_metas,
                    read_reqs,
                    f"Mooncake transfer engine returned {ret_value}",
                )
                failed_transfers.extend(
                    transfer_id
                    for d_req_id, (transfer_id, _) in resolved.items()
                    if d_req_id in read_req_set
                )

        # Report the batch so the prefiller can release these blocks.
        done = MooncakePullDone(
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            transfer_ids=[transfer_id for transfer_id, _ in resolved.values()],
            failed_transfers=failed_transfers or None,
        )
        await sock.send(self._encoder.encode(done))

    async def _hold_read_lease(
        self,
        worker_addr: str,
        transfer_ids: list[TransferId],
        read_over: asyncio.Event,
    ) -> None:
        """Tell the prefiller this read is still running, until it finishes.

        The prefiller reclaims the blocks of a reader it has not heard from
        within the abort timeout, which is how a dead decoder's blocks come
        back. An engine call that outlives that timeout therefore reports in
        while it waits, so a slow read is never mistaken for a gone reader.
        """
        interval = max(envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT / 3, 0.5)
        payload = self._encoder.encode(
            MooncakePullReading(
                engine_id=self.engine_id,
                tp_rank=self.tp_rank,
                pp_rank=self.pp_rank,
                transfer_ids=transfer_ids,
            )
        )
        # One socket for the whole read: closing it after every renewal would
        # drop the message that has just been queued.
        with make_zmq_socket(
            self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
        ) as sock:
            while not read_over.is_set():
                await sock.send(payload)
                try:
                    await asyncio.wait_for(read_over.wait(), interval)
                except TimeoutError:
                    continue

    def _read_blocks(
        self,
        remote_session: str,
        local_ptrs: list[int],
        remote_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_read(
            remote_session, local_ptrs, remote_ptrs, lengths
        )
        duration = time.perf_counter() - start_time
        if ret_value == 0:
            self.xfer_stats.record_transfer(
                duration_s=duration,
                total_bytes=sum(lengths),
                num_descs=len(local_ptrs),
            )
            logger.debug(
                "Reading from %s done, took %s (%d descriptors, %d bytes)",
                remote_session,
                duration,
                len(local_ptrs),
                sum(lengths),
            )
        else:
            self.xfer_stats.record_failed_transfer()
            logger.warning(
                "Reading from %s failed (ret=%s) after %s (%d descriptors, %d bytes)",
                remote_session,
                ret_value,
                duration,
                len(local_ptrs),
                sum(lengths),
            )
        return ret_value
