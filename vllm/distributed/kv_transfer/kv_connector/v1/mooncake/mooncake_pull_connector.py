# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake pull connector: the decoder reads the prefill KV cache itself.

`MooncakeConnector` has the prefiller write into the decoder's registered
memory, so every request batch carries the decoder's KV region table to the
prefiller. This connector turns the transfer around: the decoder asks each
paired prefiller worker for its KV address book (once per peer, then cached)
and for the block ids of the requests it is about to read, then reads the bytes
with `batch_transfer_sync_read` at its own scheduling pace. The prefiller holds
the blocks until every reader reports the read as finished.

Both connectors share the scheduler, the request metadata, and the registered
regions; only the transfer direction and the side-channel messages differ.
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
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compat import (
    compute_mooncake_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorWorker,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    ReqId,
    SendBlockMeta,
    TransferId,
    TransferRegion,
    _align_transfer_regions,
    _compute_sender_transfer_plan,
    _validate_asymmetric_region_lengths,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.pull_protocol import (
    PULL_PROTOCOL,
    MooncakePullAck,
    MooncakePullDone,
    MooncakePullLayout,
    MooncakePullReading,
    MooncakePullRequest,
    MooncakePullResponse,
    PullMessage,
    PullStatus,
    ReaderKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.read_leases import (
    ReadLeases,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer_plan import (
    ReadRequest,
    plan_read_block_descriptors,
)
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
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

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return worker-local transfer stats since the last call.

        Note the P/D asymmetry of the pull direction: the decoder performs the
        read, so transfer latency, bytes and descriptor counts are recorded on
        the decoder worker, while the prefiller records the failures it observes
        and the requests whose blocks expired before any reader came.
        """
        return super().get_kv_connector_stats()


class MooncakePullConnectorWorker(MooncakeConnectorWorker):
    """Worker-side implementation of the pull connector.

    On the prefiller the worker holds KV blocks and hands out its address book
    and block ids. On the decoder it reads those blocks into its own cache and
    reports completion, which releases the prefiller's blocks.
    """

    connector_name = "MooncakePullConnector"
    # A prefiller that never reports a connector name predates the pull
    # connector, so a reader cannot tell whether it speaks this protocol.
    allow_missing_peer_connector = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        self._pull_message_decoder = msgspec.msgpack.Decoder(PullMessage)
        self._pull_response_decoder = msgspec.msgpack.Decoder(MooncakePullResponse)
        self._pull_ack_decoder = msgspec.msgpack.Decoder(MooncakePullAck)
        # Both sides hash the same transfer-relevant configuration and compare
        # it during the handshake, so a peer that cannot interpret these bytes
        # is refused before any transfer starts.
        self._compat_hash = compute_mooncake_compatibility_hash(
            vllm_config=vllm_config,
            attn_backend_names=[backend.get_name() for backend in self.attn_backends],
            transfer_mode="pull",
        )

        # Prefiller side: this worker's own address book, built on first use,
        # and the leases of the readers that were handed block ids.
        self._layout: MooncakePullLayout | None = None
        self._leases = ReadLeases()

        # Decoder side: peer worker address -> its cached address book.
        self._peer_layouts: dict[str, MooncakePullLayout] = {}

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
    # Prefiller side: hand out blocks and address book
    ############################################################

    def _get_layout(self) -> MooncakePullLayout:
        """This worker's address book, hashed so readers notice a new one."""
        if self._layout is None:
            layout = MooncakePullLayout(
                engine_id=self.engine_id,
                hostname=self.hostname,
                rpc_port=self.rpc_port,
                tp_rank=self.tp_rank,
                pp_rank=self.pp_rank,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                block_size=self.block_size,
                physical_blocks_per_logical=(
                    self._physical_blocks_per_logical_kv_block
                ),
                use_mla=self.use_mla,
                has_mamba=self.kv_cache_config.has_mamba_layers,
                producer_cache_replicated=self._producer_cache_is_replicated(),
                kv_caches_base_addr=self.kv_caches_base_addr,
                block_lens=self.block_len_per_layer,
                kv_block_lens=self.kv_block_len_per_layer,
                layer_names=self.registered_layer_names,
                layer_indices=self.registered_layer_indices,
                group_indices=self.registered_group_indices,
                shared_group_ids=[list(groups) for groups in self.region_shared_groups],
                row_offsets=list(self.region_row_offsets),
            )
            layout.layout_id = safe_hash(
                self._encoder.encode(layout), usedforsecurity=False
            ).hexdigest()[:16]
            self._layout = layout
        return self._layout

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
        self._check_protocol(message.protocol)
        if isinstance(message, MooncakePullRequest):
            await self._handle_pull_request(identity, sock, message)
        elif isinstance(message, MooncakePullReading):
            self._handle_read_lease(message)
        else:
            await self._handle_read_done(identity, sock, message)

    @staticmethod
    def _check_protocol(protocol: str) -> None:
        if protocol != PULL_PROTOCOL:
            raise RuntimeError(
                f"Unsupported pull protocol {protocol!r}, expected {PULL_PROTOCOL!r}."
            )

    def _compat_mismatch_reason(self, peer_hash: str | None) -> str:
        return (
            "Mooncake pull transfer refused: the peer configuration is "
            f"incompatible with this one (peer compat hash {peer_hash}, local "
            f"{self._compat_hash}). Both sides must run the same model, dtype, KV "
            "cache dtype, attention backend, speculative configuration and "
            "connector version."
        )

    async def _send_side_channel_error(
        self, identity: bytes, sock: zmq.asyncio.Socket, err_msg: str
    ) -> None:
        # Reply in the push connector's error envelope: a peer that runs the
        # push connector fails its requests with this message instead of
        # waiting for its timeout, and the message names the mismatch.
        response = MooncakeXferResponse(
            status=MooncakeXferResponseStatus.ERROR, err_msg=err_msg
        )
        await sock.send_multipart((identity, self._encoder.encode(response)))

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
        if request.compat_hash != self._compat_hash:
            # Answer in this protocol so the reader reports the real reason
            # instead of an undecodable response.
            response = MooncakePullResponse(
                status=PullStatus.ERROR,
                compat_hash=self._compat_hash,
                err_msg=self._compat_mismatch_reason(request.compat_hash),
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return

        readers_key: ReaderKey = (request.engine_id, request.tp_rank, request.pp_rank)

        # Handing out block ids lets the reader slice this worker's cache by KV
        # head, so the reader's TP degree has to be a layout this cache can
        # serve. Only this side holds the registered layer specs the check
        # needs, so it runs here.
        head_err = self._validate_head_resharding_layout(
            request.tp_size, self._local_transfer_regions()
        )
        if head_err is not None:
            response = MooncakePullResponse(
                status=PullStatus.ERROR,
                compat_hash=self._compat_hash,
                err_msg=head_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return

        layout = self._get_layout() if request.want_layout else None

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
        try:
            while True:
                status = PullStatus.FINISH if not waiters else PullStatus.CONTINUE
                response = MooncakePullResponse(
                    status=status,
                    layout_id=self._get_layout().layout_id,
                    compat_hash=self._compat_hash,
                    layout=layout,
                    resolved=resolved or None,
                    err_reqs=err_reqs or None,
                    err_msg=err_msg,
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))

                # The address book travels with the first response only.
                layout = None
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

    async def _handle_read_done(
        self, identity: bytes, sock: zmq.asyncio.Socket, message: MooncakePullDone
    ) -> None:
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

        ack = MooncakePullAck()
        await sock.send_multipart((identity, self._encoder.encode(ack)))

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
            protocol=PULL_PROTOCOL,
            compat_hash=self._compat_hash,
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
            want_layout=worker_addr not in self._peer_layouts,
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
                    response = self._pull_response_decoder.decode(await sock.recv())
                    if not await self._handle_pull_response(
                        sock, worker_addr, pull_metas, response
                    ):
                        return
                    if response.status == PullStatus.FINISH:
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
            protocol=PULL_PROTOCOL,
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
        response: MooncakePullResponse,
    ) -> bool:
        """Act on one prefiller response; False ends this peer's read."""
        if response.status == PullStatus.ERROR:
            await self._abandon_batch(
                sock, pull_metas, response.err_msg or "transfer error"
            )
            return False

        if response.compat_hash != self._compat_hash:
            await self._abandon_batch(
                sock,
                pull_metas,
                self._compat_mismatch_reason(response.compat_hash),
            )
            return False

        if response.layout is not None:
            layout_err = self._accept_layout(worker_addr, response.layout)
            if layout_err is not None:
                await self._abandon_batch(sock, pull_metas, layout_err)
                return False

        layout = self._peer_layouts[worker_addr]
        if response.layout_id and response.layout_id != layout.layout_id:
            self._peer_layouts.pop(worker_addr)
            await self._abandon_batch(
                sock,
                pull_metas,
                f"producer worker {worker_addr} changed its KV layout "
                f"({layout.layout_id} -> {response.layout_id}); it restarted and "
                "the KV of this request must be recomputed",
            )
            return False

        if response.resolved:
            await self._read_batch(sock, worker_addr, pull_metas, response.resolved)

        if response.err_reqs:
            self._fail_requests(
                pull_metas, response.err_reqs, response.err_msg or "unknown error"
            )
        return True

    def _accept_layout(
        self, worker_addr: str, layout: MooncakePullLayout
    ) -> str | None:
        """Cache a peer address book, or explain why it cannot be used."""
        cached = self._peer_layouts.get(worker_addr)
        if cached is not None:
            if cached.layout_id != layout.layout_id:
                # Drop the stale address book so the next request for this peer
                # asks for the new one instead of failing the same way again.
                self._peer_layouts.pop(worker_addr)
                return (
                    f"producer worker {worker_addr} reports a new KV layout "
                    f"({cached.layout_id} -> {layout.layout_id}); it restarted, "
                    "so this request must be recomputed"
                )
            return None

        mismatch = self._layout_mismatch_reason(layout)
        if mismatch is not None:
            return mismatch
        self._peer_layouts[worker_addr] = layout
        return None

    def _layout_mismatch_reason(self, layout: MooncakePullLayout) -> str | None:
        """Reject address books that cannot describe this worker's own cache."""
        if layout.block_size != self.block_size:
            return (
                f"producer block size {layout.block_size} differs from the local "
                f"{self.block_size}; a pull needs the same physical block size on "
                "both sides"
            )
        if (
            layout.physical_blocks_per_logical
            != self._physical_blocks_per_logical_kv_block
        ):
            return (
                "producer uses "
                f"{layout.physical_blocks_per_logical} physical blocks per logical "
                "block while this decoder uses "
                f"{self._physical_blocks_per_logical_kv_block}"
            )
        if layout.use_mla != self.use_mla:
            return (
                f"producer use_mla={layout.use_mla} differs from the local "
                f"{self.use_mla}"
            )
        if layout.has_mamba != self.kv_cache_config.has_mamba_layers:
            return (
                f"producer has_mamba={layout.has_mamba} differs from the local "
                f"{self.kv_cache_config.has_mamba_layers}"
            )
        return None

    def _peer_transfer_regions(
        self, layout: MooncakePullLayout
    ) -> list[TransferRegion]:
        shared = [tuple(groups) for groups in layout.shared_group_ids]
        shared_groups = None if len(shared) != len(layout.layer_names) else shared
        row_offsets = (
            None
            if len(layout.row_offsets) != len(layout.layer_names)
            else layout.row_offsets
        )
        return self._get_transfer_regions(
            base_addrs=layout.kv_caches_base_addr,
            block_lens=layout.block_lens,
            kv_block_lens=layout.kv_block_lens,
            layer_names=layout.layer_names,
            layer_indices=layout.layer_indices,
            group_indices=layout.group_indices,
            shared_group_ids=shared_groups,
            row_offsets=row_offsets,
        )

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

    def _plan_reads(
        self,
        layout: MooncakePullLayout,
        pull_metas: dict[ReqId, PullReqMeta],
        resolved: dict[ReqId, tuple[TransferId, list[list[int]]]],
    ) -> tuple[list[int], list[int], list[int], list[ReqId], str | None]:
        """Plan the reads of one response batch.

        Returns:
            Local pointers to read into, remote pointers to read from, lengths,
            requests that failed planning, and the failure reason.

        """
        producer_regions = self._peer_transfer_regions(layout)
        consumer_regions = self._local_transfer_regions()
        producer_regions, consumer_regions, align_err = _align_transfer_regions(
            producer_regions,
            consumer_regions,
            allow_partial_layers=(
                layout.pp_size > 1 and layout.pp_size != self.pp_size
            ),
        )
        if align_err is not None:
            return [], [], [], list(resolved), align_err

        total_num_kv_heads = (
            None
            if self.use_mla or self.kv_cache_config.has_mamba_layers
            else self.transfer_topo.total_num_kv_heads
        )
        validation_err = _validate_asymmetric_region_lengths(
            local_regions=producer_regions,
            remote_regions=consumer_regions,
            local_tp_size=layout.tp_size,
            remote_tp_size=self.tp_size,
            producer_cache_replicated=layout.producer_cache_replicated,
            total_num_kv_heads=total_num_kv_heads,
        )
        if validation_err is not None:
            return [], [], [], list(resolved), validation_err

        requests = [
            ReadRequest(
                key=d_req_id,
                producer_block_ids=producer_block_ids,
                consumer_block_ids=pull_metas[d_req_id].local_block_ids,
            )
            for d_req_id, (_, producer_block_ids) in resolved.items()
            if d_req_id in pull_metas
        ]

        def plan(
            producer_region: TransferRegion, consumer_region: TransferRegion
        ) -> tuple[bool, int, int, int]:
            # Plan the copy from the prefiller's side: it owns the bytes.
            return _compute_sender_transfer_plan(
                local_tp_rank=layout.tp_rank,
                local_tp_size=layout.tp_size,
                remote_tp_rank=self.tp_rank,
                remote_tp_size=self.tp_size,
                local_kv_block_len=producer_region.kv_block_len,
                remote_kv_block_len=consumer_region.kv_block_len,
                producer_cache_replicated=layout.producer_cache_replicated,
                total_num_kv_heads=total_num_kv_heads,
            )

        # The prefiller holds the bytes, so its blocks are the remote side.
        local_ptrs, remote_ptrs, lengths, err_reqs, err_msg = (
            plan_read_block_descriptors(
                requests,
                producer_regions=producer_regions,
                consumer_regions=consumer_regions,
                group_specs=self.kv_cache_config.transfer_groups,
                producer_blocks_per_logical=layout.physical_blocks_per_logical,
                consumer_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
                plan=plan,
            )
        )
        return local_ptrs, remote_ptrs, lengths, err_reqs, err_msg

    async def _read_batch(
        self,
        sock: zmq.asyncio.Socket,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
        resolved: dict[ReqId, tuple[TransferId, list[list[int]]]],
    ) -> None:
        """Read one response batch and report it back to the prefiller."""
        layout = self._peer_layouts[worker_addr]
        (
            local_ptrs,
            remote_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = self._plan_reads(layout, pull_metas, resolved)

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
            remote_session = f"{layout.hostname}:{layout.rpc_port}"
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
            protocol=PULL_PROTOCOL,
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            transfer_ids=[transfer_id for transfer_id, _ in resolved.values()],
            failed_transfers=failed_transfers or None,
        )
        await sock.send(self._encoder.encode(done))
        ack = self._pull_ack_decoder.decode(await sock.recv())
        if not ack.ok:
            logger.warning(
                "Producer worker %s did not acknowledge the read of %s",
                worker_addr,
                list(resolved),
            )

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
                protocol=PULL_PROTOCOL,
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
