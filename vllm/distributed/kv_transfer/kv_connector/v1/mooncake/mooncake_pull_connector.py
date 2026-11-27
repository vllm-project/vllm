# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake pull connector: the decoder reads the prefill KV cache itself.

The prefiller holds the blocks it hands out until every reader reports the
read finished. Both connectors share the scheduler, metadata and response
envelope.
"""

import asyncio
import contextlib
import time
from collections.abc import Collection, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import msgspec
import numpy as np
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _PULL_HEARTBEAT_IVL_MS,
    _PULL_HEARTBEAT_TIMEOUT_MS,
    MooncakeBaseConnector,
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
    TransferRequest,
    _compute_sender_transfer_plan,
    _lost_peer_monitor,
    _recv_unless_lost,
    _validate_asymmetric_region_lengths,
    plan_transfer_descriptors,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.read_leases import (
    ReaderKey,
    ReadLeases,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.backend import (
    TransferOp,
    TransferState,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class MooncakePullRequest(
    msgspec.Struct,
    tag="read",
    tag_field="kind",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Decoder to prefiller: these requests are about to be read."""

    engine_id: EngineId
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    # d_req_id -> (transfer_id, whether this reader needs any bytes)
    reqs: dict[ReqId, tuple[TransferId, bool]]
    # The reader asks for the registered regions on its first pull from this
    # peer and reuses them afterwards.
    want_regions: bool = False


class MooncakePullDone(
    msgspec.Struct,
    tag="done",
    tag_field="kind",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Decoder to prefiller: these transfers are finished on the reader."""

    engine_id: EngineId
    tp_rank: int
    pp_rank: int
    transfer_ids: list[TransferId]


class MooncakePullReading(
    msgspec.Struct,
    tag="reading",
    tag_field="kind",
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Decoder to prefiller: the read of these transfers is still running."""

    engine_id: EngineId
    tp_rank: int
    pp_rank: int
    transfer_ids: list[TransferId]


PullMessage = MooncakePullRequest | MooncakePullDone | MooncakePullReading


class _PeerLostError(RuntimeError):
    """The prefiller's connection disappeared before it finished answering."""


class _PeerLayout(NamedTuple):
    """One prefiller's validated layout, reusable across waves and batches."""

    layout_id: int
    consumer_regions: list[TransferRegion]
    producer_regions: list[TransferRegion]


@dataclass(slots=True)
class _ReadWave:
    """Descriptors of one prefiller response, read in arrival order."""

    reqs: dict[ReqId, TransferId]
    local_ptrs: np.ndarray
    remote_ptrs: np.ndarray
    lengths: np.ndarray
    remote_session: str
    err_reqs: list[ReqId]
    err_msg: str | None


def _consume_task_result(task: asyncio.Task) -> None:
    """Read the result of a task nobody awaits, so it is not reported twice."""
    if task.done() and not task.cancelled():
        task.exception()


class MooncakePullConnector(MooncakeBaseConnector):
    """Pull-based (READ) Mooncake KV transfer connector.

    Same scheduler, metadata and registered regions as the push connector;
    only the worker differs, because the decoder reads the prefiller's KV cache
    instead of the prefiller writing into the decoder's memory.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            self.connector_worker = MooncakePullConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )


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
        # Prefiller side: registered regions stay fixed until the layout id
        # changes, and every pull asks for them.
        self._local_regions_layout_id = -1
        self._local_regions_cache: tuple[list[TransferRegion], str | None] | None = None

        # Decoder side: peer worker address -> the layout it reported once.
        self._peer_layouts: dict[str, _PeerLayout] = {}

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
        super().shutdown()

    ############################################################
    # Prefiller side: hand out blocks and registered regions
    ############################################################

    def _resolve_expected_readers(
        self, send_meta: SendBlockMeta, reader_tp_size: int, reader_pp_size: int
    ) -> None:
        """How many reader workers a set of blocks is served to."""
        expected = self._expected_peer_workers(
            self.transfer_topo.handshake_target_ranks(reader_tp_size), reader_pp_size
        )
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
        """Report released transfers and forget leases that ended with them."""
        finished = await super().fetch_finished_sending_reqs()
        for transfer_id in self._leases:
            if transfer_id not in self.reqs_need_send:
                self._leases.forget(transfer_id)
        return finished

    def _read_lease_timeout(self) -> float:
        return self._kv_lease_duration or envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT

    def _send_idle(self, transfer_id: TransferId) -> bool:
        """Whether the blocks of a transfer are free to be released."""
        return self._leases.may_release(transfer_id, self._read_lease_timeout())

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata) -> None:
        """Register finished requests, then re-check completions reported early."""
        await super().record_send_reqs(metadata)
        for send_meta in list(self.reqs_need_send.values()):
            self._complete_transfer_if_read_by_all(send_meta)

    async def _handle_control_message(
        self, identity: bytes, payload: bytes, sock: zmq.asyncio.Socket
    ) -> bool:
        """Handle lease traffic ahead of the queued pull requests."""
        if await super()._handle_control_message(identity, payload, sock):
            return True
        try:
            message = self._pull_message_decoder.decode(payload)
        except msgspec.DecodeError:
            # Leave it to the worker, which names the protocol mismatch.
            return False
        if isinstance(message, MooncakePullReading):
            self._handle_read_lease(message)
            return True
        if isinstance(message, MooncakePullDone):
            self._handle_read_done(message)
            return True
        return False

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
        if not isinstance(message, MooncakePullRequest):
            raise RuntimeError(
                "Only pull requests may reach a sender worker; the listener "
                f"handles lease traffic. Got {type(message).__name__}."
            )
        await self._handle_pull_request(identity, sock, message)

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

    def _cached_local_transfer_regions(
        self,
    ) -> tuple[list[TransferRegion], str | None]:
        """Expand the registered regions once per layout id."""
        if self._local_regions_layout_id != self._kv_layout_id:
            self._local_regions_cache = self._local_transfer_regions()
            self._local_regions_layout_id = self._kv_layout_id
        assert self._local_regions_cache is not None
        return self._local_regions_cache

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
        local_regions, local_err = self._cached_local_transfer_regions()
        if local_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR, err_msg=local_err
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        head_err = self._validate_head_resharding_layout(request.tp_size, local_regions)
        if head_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR, err_msg=head_err
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return

        resolved: dict[ReqId, tuple[TransferId, list[list[int]]]] = {}
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        waiters: dict[asyncio.Task, tuple[ReqId, SendBlockMeta]] = {}

        for d_req_id, (transfer_id, needs_data) in request.reqs.items():
            send_meta = self.reqs_need_send.get(transfer_id)
            if send_meta is None:
                if transfer_id in self._expired_transfer_ids:
                    # The prefiller freed these blocks already; waiting for
                    # them would only end at the ready timeout.
                    logger.warning(
                        "Request %s was pulled after its KV expired on P side.",
                        d_req_id,
                    )
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = (
                            "The prefiller released this request before handing "
                            "out its blocks."
                        )
                    continue
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
            waiters[asyncio.create_task(self._wait_for_blocks(d_req_id, send_meta))] = (
                d_req_id,
                send_meta,
            )

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
                        d_req_id, send_meta = waiters[task]
                        if send_meta.p_req_id == "" and not self._leases.holders(
                            send_meta.transfer_id
                        ):
                            self.reqs_need_send.pop(send_meta.transfer_id, None)
                            self._leases.forget(send_meta.transfer_id)
                        task.cancel()
                    waiters.clear()
                    continue

                for task in done:
                    d_req_id, send_meta = waiters.pop(task)
                    if (
                        send_meta.aborted.is_set()
                        or self.reqs_need_send.get(send_meta.transfer_id)
                        is not send_meta
                    ):
                        err_reqs.append(d_req_id)
                        if err_msg is None:
                            err_msg = (
                                "The prefiller released this request before "
                                "handing out its blocks."
                            )
                        continue
                    if not self._leases.hand_out(send_meta.transfer_id, readers_key):
                        err_reqs.append(d_req_id)
                        if err_msg is None:
                            err_msg = "This reader already reported the transfer."
                        continue
                    resolved[d_req_id] = (
                        send_meta.transfer_id,
                        send_meta.local_block_ids,
                    )
        finally:
            for task in waiters:
                task.cancel()

    def _handle_read_lease(self, message: MooncakePullReading) -> None:
        """Extend the lease of transfers this reader is still reading."""
        readers_key: ReaderKey = (message.engine_id, message.tp_rank, message.pp_rank)
        for transfer_id in message.transfer_ids:
            self._leases.renew(transfer_id, readers_key)

    def _handle_read_done(self, message: MooncakePullDone) -> None:
        """Release the blocks a reader reported. Nothing is answered."""
        readers_key: ReaderKey = (message.engine_id, message.tp_rank, message.pp_rank)
        for transfer_id in message.transfer_ids:
            send_meta = self.reqs_need_send.get(transfer_id)
            if send_meta is not None:
                self._mark_reader_done(send_meta, readers_key)

    ############################################################
    # Decoder side: read the peer's blocks
    ############################################################

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        """Read this peer's share of a batch, and settle it when it is over.

        The engine call cannot be cancelled, so a cancelled caller defers
        settling to the read itself and never frees blocks still being written.
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
        needs_data = {
            req_id: any(self._filter_host_block_ids(pull_meta.local_block_ids))
            for req_id, pull_meta in pull_metas.items()
        }
        settled_reqs = {req_id for req_id, needs in needs_data.items() if not needs}
        request = MooncakePullRequest(
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            reqs={
                req_id: (pull_meta.transfer_id, needs_data[req_id])
                for req_id, pull_meta in pull_metas.items()
            },
            want_regions=worker_addr not in self._peer_layouts,
        )
        logger.debug(
            "Reading kv_caches for %s from producer worker %s", req_ids, worker_addr
        )

        queue: asyncio.Queue[_ReadWave | None] = asyncio.Queue()
        active: set[TransferId] = set()
        done_sent: set[TransferId] = set()
        read_over = asyncio.Event()
        lease_task: asyncio.Task | None = None
        reader: asyncio.Task | None = None

        try:
            with (
                make_zmq_socket(
                    self.async_zmq_ctx,
                    worker_addr,
                    zmq.DEALER,
                    bind=False,
                    linger=0,
                    heartbeat_ivl_ms=_PULL_HEARTBEAT_IVL_MS,
                    heartbeat_timeout_ms=_PULL_HEARTBEAT_TIMEOUT_MS,
                ) as sock,
                _lost_peer_monitor(sock) as monitor,
            ):
                # If something goes wrong, let the prefiller wait for its own
                # timeout before we give up on the request.
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(self._encoder.encode(request))
                reader = asyncio.create_task(
                    self._read_pending(sock, queue, pull_metas, active, done_sent)
                )
                try:
                    while True:
                        ret_msg = await _recv_unless_lost(sock, monitor)
                        if ret_msg is None:
                            raise _PeerLostError(
                                f"lost connection to prefill worker {worker_addr}"
                            )
                        response = self._xfer_resp_decoder.decode(ret_msg)
                        if response.status == MooncakeXferResponseStatus.ERROR:
                            await self._abandon_batch(
                                sock,
                                pull_metas,
                                needs_data,
                                active,
                                done_sent,
                                response.err_msg or "transfer error",
                            )
                            return
                        wave, layout_err = self._prepare_wave(
                            worker_addr, response, pull_metas, needs_data
                        )
                        if layout_err is not None:
                            await self._abandon_batch(
                                sock,
                                pull_metas,
                                needs_data,
                                active,
                                done_sent,
                                layout_err,
                            )
                            return
                        if response.err_reqs:
                            settled_reqs.update(response.err_reqs)
                            self._fail_requests(
                                pull_metas,
                                response.err_reqs,
                                response.err_msg or "unknown error",
                            )
                        if response.metadata is not None:
                            settled_reqs.update(response.metadata.req_blocks)
                        if wave is not None:
                            if lease_task is None:
                                lease_task = asyncio.create_task(
                                    self._hold_read_lease(
                                        worker_addr, active, read_over
                                    )
                                )
                                lease_task.add_done_callback(_consume_task_result)
                            active.update(wave.reqs.values())
                            await queue.put(wave)
                        if response.status == MooncakeXferResponseStatus.FINISH:
                            missing = req_ids - settled_reqs
                            if missing:
                                self._fail_requests(
                                    pull_metas,
                                    missing,
                                    "prefiller finished without resolving request",
                                )
                            break
                finally:
                    await queue.put(None)
                    if reader is not None:
                        await reader
        except _PeerLostError as e:
            self._peer_layouts.pop(worker_addr, None)
            self._forget_peer_agents(worker_addr)
            self._fail_requests(pull_metas, req_ids, str(e))
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
        finally:
            read_over.set()
            if lease_task is not None:
                await asyncio.gather(lease_task, return_exceptions=True)

    def _forget_peer_agents(self, worker_addr: str) -> None:
        """Drop a lost peer's engine so the next request queries it again."""
        for engine_id, agents in list(self._remote_agents.items()):
            if any(worker_addr in pp_addrs.values() for pp_addrs in agents.values()):
                del self._remote_agents[engine_id]

    def _prepare_wave(
        self,
        worker_addr: str,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
        needs_data: dict[ReqId, bool],
    ) -> tuple[_ReadWave | None, str | None]:
        """Validate one response and plan the reads it asks for."""
        meta = response.metadata
        if meta is None:
            return (
                None,
                f"producer worker {worker_addr} answered without its KV layout",
            )

        layout = self._peer_layouts.get(worker_addr)
        if layout is None:
            if not meta.kv_caches_base_addr:
                return (
                    None,
                    f"producer worker {worker_addr} did not send its KV layout, "
                    "so its blocks cannot be addressed",
                )
            layout_err, consumer_regions, producer_regions = self._prepare_layout(meta)
            if layout_err is not None:
                return None, layout_err
            layout = _PeerLayout(meta.layout_id, consumer_regions, producer_regions)
            self._peer_layouts[worker_addr] = layout
        elif layout.layout_id != meta.layout_id:
            # Drop the stale layout so the recomputed request asks for the new
            # one instead of failing the same way again.
            self._peer_layouts.pop(worker_addr)
            return (
                None,
                f"producer worker {worker_addr} changed its KV layout "
                f"({layout.layout_id} -> {meta.layout_id}); it restarted and "
                "the KV of this request must be recomputed",
            )

        # A full local hit was already reported by the prefiller.
        resolved = {
            req_id: entry
            for req_id, entry in meta.req_blocks.items()
            if needs_data.get(req_id, False)
        }
        if not resolved:
            return None, None
        local_ptrs, remote_ptrs, lengths, err_reqs, err_msg = self._plan_reads(
            meta,
            pull_metas,
            resolved,
            layout.consumer_regions,
            layout.producer_regions,
        )
        return (
            _ReadWave(
                reqs={
                    req_id: transfer_id for req_id, (transfer_id, _) in resolved.items()
                },
                local_ptrs=local_ptrs,
                remote_ptrs=remote_ptrs,
                lengths=lengths,
                remote_session=f"{meta.remote_hostname}:{meta.remote_port}",
                err_reqs=err_reqs,
                err_msg=err_msg,
            ),
            None,
        )

    async def _read_pending(
        self,
        sock: zmq.asyncio.Socket,
        queue: asyncio.Queue[_ReadWave | None],
        pull_metas: dict[ReqId, PullReqMeta],
        active: set[TransferId],
        done_sent: set[TransferId],
    ) -> None:
        """Read each wave in arrival order, then report the transfers."""
        loop = asyncio.get_running_loop()
        while True:
            wave = await queue.get()
            if wave is None:
                return
            try:
                if wave.err_reqs:
                    self._fail_requests(
                        pull_metas,
                        wave.err_reqs,
                        wave.err_msg or "read planning failed",
                    )
                if len(wave.local_ptrs):
                    ret_value = await loop.run_in_executor(
                        self._reader_executor,
                        partial(
                            self._read_blocks,
                            wave.remote_session,
                            wave.local_ptrs,
                            wave.remote_ptrs,
                            wave.lengths,
                        ),
                    )
                    if ret_value != 0:
                        self._fail_requests(
                            pull_metas,
                            list(wave.reqs),
                            f"Mooncake transfer engine returned {ret_value}",
                        )
                await self._report_done(sock, wave.reqs.values(), done_sent)
                active.difference_update(wave.reqs.values())
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._fail_requests(pull_metas, list(wave.reqs), f"read failed: {e}")
                await self._report_done(sock, wave.reqs.values(), done_sent)
                active.difference_update(wave.reqs.values())
                raise

    async def _report_done(
        self,
        sock: zmq.asyncio.Socket,
        transfer_ids: Iterable[TransferId],
        done_sent: set[TransferId],
    ) -> None:
        """Tell the prefiller which transfers this reader is finished with."""
        fresh = [
            transfer_id for transfer_id in transfer_ids if transfer_id not in done_sent
        ]
        if not fresh:
            return
        done = MooncakePullDone(
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            transfer_ids=fresh,
        )
        await sock.send(self._encoder.encode(done))
        done_sent.update(fresh)

    async def _abandon_batch(
        self,
        sock: zmq.asyncio.Socket,
        pull_metas: dict[ReqId, PullReqMeta],
        needs_data: dict[ReqId, bool],
        active: set[TransferId],
        done_sent: set[TransferId],
        reason: str,
    ) -> None:
        """Fail this batch locally, then release the blocks the peer holds.

        A transfer still being read reports through `_read_pending` instead.
        """
        self._fail_requests(pull_metas, set(pull_metas), reason)
        transfer_ids = [
            pull_meta.transfer_id
            for req_id, pull_meta in pull_metas.items()
            if needs_data.get(req_id, False) and pull_meta.transfer_id not in active
        ]
        await self._report_done(sock, transfer_ids, done_sent)

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

    def _total_num_kv_heads(self) -> int | None:
        if self.use_mla or self.kv_cache_config.has_mamba_layers:
            return None
        return self.transfer_topo.total_num_kv_heads

    def _prepare_layout(
        self, meta: MooncakeXferMetadata
    ) -> tuple[str | None, list[TransferRegion], list[TransferRegion]]:
        """Prepare this peer's regions and say why they cannot describe a read."""
        consumer_regions, producer_regions, prepare_err = (
            self._prepare_transfer_regions(meta, is_pull=True)
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
        consumer_regions: list[TransferRegion],
        producer_regions: list[TransferRegion],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ReqId], str | None]:
        """Plan one response batch into local and remote descriptors."""
        total_num_kv_heads = self._total_num_kv_heads()

        requests = [
            TransferRequest(
                key=d_req_id,
                src_block_ids=producer_block_ids,
                dst_block_ids=self._filter_host_block_ids(
                    pull_metas[d_req_id].local_block_ids
                ),
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
        remote_ptrs, local_ptrs, lengths, err_reqs, err_msg = plan_transfer_descriptors(
            requests,
            src_regions=producer_regions,
            dst_regions=consumer_regions,
            group_specs=self.kv_cache_config.transfer_groups,
            blocks_per_logical=meta.physical_blocks_per_logical,
            plan=plan,
        )
        return local_ptrs, remote_ptrs, lengths, err_reqs, err_msg

    async def _hold_read_lease(
        self,
        worker_addr: str,
        active_transfers: set[TransferId],
        read_over: asyncio.Event,
    ) -> None:
        """Tell the prefiller the transfers still being read are alive.

        The socket opens on the first report, so a read that finishes early
        sends nothing.
        """
        interval = max(self._read_lease_timeout() / 3, 0.5)
        sock: zmq.asyncio.Socket | None = None
        try:
            while not read_over.is_set():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(read_over.wait(), interval)
                if read_over.is_set():
                    return
                if not active_transfers:
                    continue
                if sock is None:
                    sock = make_zmq_socket(
                        self.async_zmq_ctx,
                        worker_addr,
                        zmq.DEALER,
                        bind=False,
                        linger=0,
                    )
                payload = self._encoder.encode(
                    MooncakePullReading(
                        engine_id=self.engine_id,
                        tp_rank=self.tp_rank,
                        pp_rank=self.pp_rank,
                        transfer_ids=sorted(active_transfers),
                    )
                )
                await sock.send(payload)
        finally:
            if sock is not None:
                sock.close(linger=0)

    def _read_blocks(
        self,
        remote_session: str,
        local_ptrs: np.ndarray,
        remote_ptrs: np.ndarray,
        lengths: np.ndarray,
    ) -> int:
        start_time = time.perf_counter()
        result = self.transfer_backend.transfer_sync_arrays(
            remote_session,
            TransferOp.READ,
            local_ptrs,
            remote_ptrs,
            lengths,
        )
        ret_value = 0 if result.state is TransferState.COMPLETED else -1
        duration = time.perf_counter() - start_time
        total_bytes = int(lengths.sum())
        if ret_value == 0:
            self.xfer_stats.record_transfer(
                duration_s=duration,
                total_bytes=total_bytes,
                num_descs=len(local_ptrs),
            )
            logger.debug(
                "Reading from %s done, took %s (%d descriptors, %d bytes)",
                remote_session,
                duration,
                len(local_ptrs),
                total_bytes,
            )
        else:
            self.xfer_stats.record_failed_transfer()
            logger.warning(
                "Reading from %s failed (ret=%s) after %s (%d descriptors, %d bytes)",
                remote_session,
                ret_value,
                duration,
                len(local_ptrs),
                total_bytes,
            )
        return ret_value
