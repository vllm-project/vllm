# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base scheduler-side logic for the NIXL connector."""

import threading
import time
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
import zmq

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    clip_ssm_state_blocks,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_DIGESTS_MSG,
    GET_META_MSG,
    HeartbeatInfo,
    NixlConnectorMetadata,
    NixlDigestMetadata,
    NixlHandshakePayload,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Server-side wait budget for GET_DIGESTS_MSG: how long the listener waits
# for digests still in flight from the workers before answering None
# (meaning: the producer has no digests for this request). Entries expire
# with the block lease, so the wait never outlives them either. The D-side
# recv timeout (_DIGEST_FETCH_TIMEOUT_MS in base_worker.py) must exceed this
# plus network slack, or a client timeout would masquerade as "no digests".
_DIGEST_SERVE_WAIT_S = 5.0


class _ReqDigestState(NamedTuple):
    """Digest tracking for one P-side producer request."""

    prompt_len: int
    # Cumulative block ids, all KV cache groups.
    block_ids: BlockIds


class NixlBaseConnectorScheduler:
    """Base implementation of Scheduler side methods shared by pull and push."""

    # Emitted in kv_transfer_params so an external router can distinguish a
    # pull (READ) producer from a push (WRITE) one. Overridden by the push
    # scheduler.
    _TRANSFER_MODE: str = "pull"

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        parallel_config = vllm_config.parallel_config
        # TP1 PCP+DCP exposes its DCP shards as transfer ranks.
        self.transfer_tp_size = max(
            parallel_config.tensor_parallel_size,
            parallel_config.decode_context_parallel_size,
        )
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.kv_cache_config = kv_cache_config
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._heartbeat_interval = self._kv_lease_duration // 6
        if current_platform.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = (
                vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
            )
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            # Also handle unlikely SW-only model case instead of checking num_groups>1.
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.transfer_groups
            )
        )
        self._has_mamba = kv_cache_config.has_mamba_layers

        logger.info("Initializing NIXL Scheduler %s", engine_id)
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            logger.info("Hybrid Memory Allocator is enabled with NIXL")

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[
            ReqId, tuple[Request, BlockIds, tuple[int, ...], bool]
        ] = {}
        self._reqs_need_save: dict[ReqId, Request] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

        # Heartbeat tracking: requests needing periodic lease-renewal heartbeats to
        # remote P-side, stored as ready-to-send HeartbeatInfo grouped by remote engine
        self._heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Reverse lookup: local req_id -> (engine_id, remote_req_id) for O(1) removal
        self._heartbeat_req_engine: dict[ReqId, tuple[EngineId, ReqId]] = {}
        self._last_heartbeat_time: float = 0.0

        # Gather Sliding Window sizes for each kv cache group (if any) in number of
        # blocks per KV cache group. This is used to clip the local attention window.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.transfer_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to conservatively
        # account for boundary overlap eg window isn't fully aligned with blocks.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

        # Trailing scratch slots that mamba managers co-allocate per request
        # for speculative decoding; None for non-SSM groups.
        self._ssm_spec_blocks = [
            g.kv_cache_spec.num_speculative_blocks
            if isinstance(g.kv_cache_spec, MambaSpec)
            else None
            for g in kv_cache_config.transfer_groups
        ]

        # Threshold to decide whether to compute kv cache locally
        # or pull from a remote node: minimum number of remote
        # tokens to amortize the xfer latencies
        self.kv_recompute_threshold: int = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_recompute_threshold", 64
            )
        )

        # Bi-directional KV transfer feature supports KV block
        # transfers from D node to P node
        self.is_bidirectional_kv_xfer_enabled = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )
        self.decoder_kv_blocks_ttl = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder_kv_blocks_ttl", 480
            )
        )

        if self.is_bidirectional_kv_xfer_enabled and self.kv_recompute_threshold > 0:
            logger.info(
                "Bidirectional KV transfer is enabled and the kv "
                "recompute threshold is set to %d tokens."
                "KV blocks on D are released after a TTL of %d seconds.",
                self.kv_recompute_threshold,
                self.decoder_kv_blocks_ttl,
            )

        # KV digest (checksum) prototype, pull mode only: the P side digests
        # the blocks it exposes and ships the digests in kv_transfer_params.
        self._enable_kv_digest: bool = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "enable_kv_digest", False
            )
        )
        # P-side producer (do_remote_decode) requests; scheduler_output does
        # not carry kv_transfer_params, so this set gates digest tracking.
        self._req_is_producer: set[ReqId] = set()
        # Per-request digest tracking, driven purely by scheduler_output.
        self._req_digest_state: dict[ReqId, _ReqDigestState] = {}
        # Worker-computed digests awaiting pickup by request_finished, keyed
        # by request id then producer TP rank.
        self._pending_digests: dict[ReqId, dict[int, list[list[str]]]] = {}
        # Push mode: digests served over the side channel (D fetches after
        # the WRITE completes, so they outlive the request on P). Values are
        # (per-rank digests, perf_counter deadline); served entries are
        # deleted per rank and unserved ones reaped at the deadline. The
        # condition lets the listener wait for digests still in flight from
        # the workers; update_worker_meta notifies after storing.
        self._digests_by_req: dict[ReqId, tuple[dict[int, list[list[str]]], float]] = {}
        self._digests_cond = threading.Condition()

    def shutdown(self):
        self._stop_event.set()
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join()
            self._nixl_handshake_listener_t = None

    def on_new_request(self, request: "Request") -> None:
        """Track a request that may need heartbeats."""
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode"):
            self._truncate_request_for_prefill(request)

        # NOTE (NickLucche) This excludes request meant for P, ie heartbeats are
        # effectively disabled for Bidirectional KV transfer.
        if params is None or not params.get("do_remote_prefill"):
            return
        # Only track if all required remote fields are present.
        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        dcp_size = params.get("dcp_size", 1)
        pp_size = params.get("pp_size", 1)
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return
        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                req_ids=set(),
                host=host,
                port=port,
                tp_size=tp_size,
                dcp_size=dcp_size,
                pp_size=pp_size,
            )
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )

    def _stop_heartbeat(self, req_id: ReqId) -> None:
        """Remove *req_id* from heartbeat tracking (if tracked)."""
        if key := self._heartbeat_req_engine.pop(req_id, None):
            engine_id, remote_id = key
            if info := self._heartbeat_by_engine.get(engine_id):
                info.req_ids.discard(remote_id)
                if not info.req_ids:
                    # Clean up empty engines so we don't leak a key when remote dies.
                    del self._heartbeat_by_engine[engine_id]

    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Clip a request's block lists down to the transferable blocks.

        Sliding-window groups keep only the in-window tail: the KV cache
        manager allocates blocks for the entire sequence length and cleans up
        out-of-window blocks only prior to the `request_finished_all_groups`
        hook.

        SSM groups keep only their state-bearing slot: the trailing
        speculative scratch slots and everything before the running state
        (null placeholders and the previous step's superseded state) go.

        Use this at every block-id exchange point. Pass ``clip_ssm=False``
        for per-step partial lists (host-buffer save), where the SSM strip
        does not apply.
        """
        if len(block_ids) == 0:
            # No blocks to clip, e.g. a full prefix cache hit.
            return block_ids
        block_ids = self.kv_cache_config.select_transfer_block_ids(block_ids)
        if not self._is_hma_required:
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        clipped = []
        for i, blocks in enumerate(block_ids):
            if n_sw := self.blocks_per_sw[i]:
                blocks = blocks[-n_sw:]
            elif (
                clip_ssm
                and blocks
                and (n_spec_blocks := self._ssm_spec_blocks[i]) is not None
            ):
                blocks = clip_ssm_state_blocks(blocks, n_spec_blocks)
            clipped.append(blocks)
        return tuple(clipped)

    def set_xfer_handshake_metadata(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.

        """
        encoded_data: dict[tuple[int, int], bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        for (pp_rank, tp_rank), rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            encoded_data[(pp_rank, tp_rank)] = encoder.encode(rank_metadata)
            logger.debug(
                "PP rank %d, TP rank %d: encoded NixlHandshakePayload size: %s bytes",
                pp_rank,
                tp_rank,
                str(len(encoded_data[(pp_rank, tp_rank)])),
            )

        # Only start the listener when we have metadata to serve.
        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    encoded_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_host,
                    self.side_channel_port,
                    self._digests_by_req,
                    self._digests_cond,
                ),
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[tuple[int, int], Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
        digests_by_req: dict[ReqId, tuple[dict[int, list[list[str]]], float]],
        digests_cond: threading.Condition,
    ):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        # Listen for new requests for metadata.
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                # Decode (GET_META_MSG, pp_rank, tp_rank) or
                # (GET_DIGESTS_MSG, req_id, tp_rank).
                parts = msgspec.msgpack.decode(msg)
                if parts[0] == GET_DIGESTS_MSG:
                    _, req_id, tp_rank = parts
                    with digests_cond:
                        # Wait for digests still in flight from the workers
                        # (worker meta lags the WRITE). A timeout means the
                        # producer has no digests for this request (feature
                        # off, non-producer request, or expired entry).
                        end = time.perf_counter() + _DIGEST_SERVE_WAIT_S
                        while (entry := digests_by_req.get(req_id)) is None:
                            if not digests_cond.wait(end - time.perf_counter()):
                                break
                        rank_digests = None
                        if entry is not None:
                            # Serve-and-delete per rank: each D rank fetches
                            # only its own entry. A rank missing from an
                            # existing entry will not appear later (all
                            # producer ranks land in one worker-meta batch).
                            rank_digests = entry[0].pop(tp_rank, None)
                            if not entry[0]:
                                del digests_by_req[req_id]
                    sock.send_multipart(
                        (identity, b"", msgspec.msgpack.encode(rank_digests))
                    )
                    continue
                msg_type, target_pp_rank, target_tp_rank = parts
                logger.debug(
                    "Received message for pp rank %s, tp rank %s",
                    target_pp_rank,
                    target_tp_rank,
                )
                if msg_type != GET_META_MSG:
                    logger.warning(
                        "Connection listener got unexpected message %s", msg_type
                    )
                # Echo our perf_counter so P can estimate the clock offset.
                # perf_counter is only comparable within a process, so this
                # listener must run in the same process that stamps the block
                # expiry deadline (`_reqs_need_send`).
                ts = msgspec.msgpack.encode(time.perf_counter())
                sock.send_multipart(
                    (identity, b"", encoded_data[(target_pp_rank, target_tp_rank)], ts)
                )

    def _prefill_backoff(self) -> int:
        """Trailing prompt tokens the prefiller must not compute; the decoder
        recomputes them locally.

        Mamba needs h(N-1) so the decoder can derive h(N) itself. Multi-module
        MTP needs to keep its whole lookahead window off of the prefiller, which
        would otherwise embed the unverified drafts in the MTP layer's KV cache. The
        decoder would never rebuild them, because the update is sized by the rejection
        count, which is zero for the first decode.
        """
        return max(
            1 if self._has_mamba else 0,
            self.vllm_config.num_prefill_lookahead_tokens - 1,
        )

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. The number of prompt tokens to load from the prefiller.
        Stops short of the trailing ``_prefill_backoff()`` tokens that the decoder
        will recompute locally."""
        backoff = self._prefill_backoff()
        if backoff and num_prompt_tokens > backoff:
            return num_prompt_tokens - backoff
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the trailing ``_prefill_backoff()`` prompt tokens
        so the prefiller stops short of what the decoder recomputes locally.
        For Mamba that is the single token needed to yield h(N-1); for
        multi-module MTP it is the drafter's whole lookahead window.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        backoff = self._prefill_backoff()
        params = request.kv_transfer_params
        if (
            backoff
            and params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > backoff
        ):
            if request.prompt_token_ids is not None:
                del request.prompt_token_ids[-backoff:]
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-backoff]
            else:
                return

            del request._all_token_ids[-backoff:]
            request.num_prompt_tokens -= backoff
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def _build_save_meta(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # only called when use_host_buffer is True to build the save metadata

        # NOTE: For the prefill side, there might be a chance that an early added
        # request is a chunked prefill, so we need to check if new blocks are added
        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            req_to_save = self._reqs_need_save.get(req_id)
            if req_to_save is None or new_block_id_groups is None:
                continue
            req = req_to_save

            assert req.kv_transfer_params is not None
            clipped_block_id_groups = self.get_exchange_clipped_blocks(
                new_block_id_groups, clip_ssm=False
            )
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=clipped_block_id_groups,
                kv_transfer_params=req.kv_transfer_params,
            )
            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            is_partial = (
                req.num_computed_tokens + num_scheduled_tokens
            ) < req.num_prompt_tokens
            if not is_partial:
                # For non-partial prefills, once new req_meta is scheduled, it
                # can be removed from _reqs_need_save.
                # For partial prefill case, we will retain the request in
                # _reqs_need_save until all blocks are scheduled with req_meta.
                # Therefore, only pop if `not is_partial`.
                self._reqs_need_save.pop(req_id)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (
            req,
            block_ids,
            cached,
            awaiting_kvs,
        ) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                local_num_computed_blocks=cached,
                awaiting_kvs=awaiting_kvs,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        # Clock reference for reqs_to_send: deadlines above are in this
        # process's perf_counter domain; workers (possibly on other nodes,
        # where perf_counter has a different epoch) rebase against this.
        meta.scheduler_clock = time.perf_counter()
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        if self._enable_kv_digest and self._TRANSFER_MODE == "pull":
            self._update_blocks_to_checksum(meta, scheduler_output)
        if self._digests_by_req:
            self._reap_expired_digests()

        # Package heartbeats, throttled by heartbeat_interval.
        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Stop heartbeating for requests whose KV transfer completed."""
        for req_id in connector_output.finished_recving or ():
            self._stop_heartbeat(req_id)

    def update_worker_meta(self, worker_meta: KVConnectorWorkerMetadata) -> None:
        """Stash worker-computed block digests: for request_finished in pull
        mode, or for side-channel serving in push mode."""
        if not isinstance(worker_meta, NixlDigestMetadata):
            return
        if self._TRANSFER_MODE == "push":
            # Digests outlive the request on P: D fetches them after the
            # WRITE completes. TTL them like the block lease.
            deadline = time.perf_counter() + self._kv_lease_duration
            with self._digests_cond:
                for req_id, rank_digests in worker_meta.digests.items():
                    self._digests_by_req[req_id] = (rank_digests, deadline)
                # Wake listeners waiting on a GET_DIGESTS_MSG for these.
                self._digests_cond.notify_all()
            return
        self._pending_digests.update(worker_meta.digests)

    def _reap_expired_digests(self) -> None:
        """Drop unserved push-mode digest entries past their deadline."""
        now = time.perf_counter()
        with self._digests_cond:
            expired = [
                req_id
                for req_id, (_, deadline) in self._digests_by_req.items()
                if now >= deadline
            ]
            for req_id in expired:
                del self._digests_by_req[req_id]
        for req_id in expired:
            logger.warning("KV digests for request %s expired unserved", req_id)

    def _update_blocks_to_checksum(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Attach to *meta* the blocks of P-side producer requests whose
        prefill completes this step, so workers digest them before the
        request finishes. Tracking is driven purely by scheduler_output."""
        computed_tokens = self._track_digest_blocks(scheduler_output)
        num_scheduled = scheduler_output.num_scheduled_tokens
        for req_id, state in list(self._req_digest_state.items()):
            n_scheduled = num_scheduled.get(req_id)
            if n_scheduled is None:
                continue
            if self._prefill_finishes_this_step(
                computed_tokens[req_id], n_scheduled, state.prompt_len
            ):
                meta.blocks_to_checksum[req_id] = self._digest_block_window(
                    state.prompt_len, state.block_ids
                )
                del self._req_digest_state[req_id]

    def _track_digest_blocks(
        self, scheduler_output: SchedulerOutput
    ) -> dict[ReqId, int]:
        """Maintain ``_req_digest_state`` from scheduler_output.

        Returns the pre-step num_computed_tokens of tracked requests
        scheduled this step.
        """
        computed_tokens: dict[ReqId, int] = {}
        # New requests carry their full cumulative block list and the
        # prefix-cache-hit count as num_computed_tokens.
        for req_data in scheduler_output.scheduled_new_reqs:
            if req_data.req_id not in self._req_is_producer:
                continue
            if req_data.prompt_token_ids is None:
                logger.debug(
                    "KV digest: not tracking request %s with prompt_embeds",
                    req_data.req_id,
                )
                continue
            self._req_digest_state[req_data.req_id] = _ReqDigestState(
                prompt_len=len(req_data.prompt_token_ids),
                block_ids=req_data.block_ids,
            )
            computed_tokens[req_data.req_id] = req_data.num_computed_tokens
        # Cached requests: running requests contribute only this step's new
        # blocks; resumed-from-preemption requests carry a full replacement
        # list. num_computed_tokens is the pre-step value in both cases.
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            state = self._req_digest_state.get(req_id)
            if state is None:
                continue
            new_block_ids = cached.new_block_ids[i]
            block_ids: BlockIds
            if req_id in cached.resumed_req_ids:
                block_ids = new_block_ids or ()
            elif new_block_ids:
                block_ids = tuple(
                    list(old) + list(new)
                    for old, new in zip(state.block_ids, new_block_ids)
                )
            else:
                block_ids = state.block_ids
            self._req_digest_state[req_id] = state._replace(block_ids=block_ids)
            computed_tokens[req_id] = cached.num_computed_tokens[i]
        return computed_tokens

    @staticmethod
    def _prefill_finishes_this_step(
        num_computed: int, n_scheduled: int, prompt_len: int
    ) -> bool:
        """Whether this step computes the request's last prompt token.

        ``num_computed`` is the pre-step count: build_connector_meta runs
        before the scheduler advances num_computed_tokens.
        """
        return num_computed + n_scheduled >= prompt_len

    def _digest_block_window(
        self, num_prompt_tokens: int, block_ids: BlockIds
    ) -> BlockIds:
        """Returns the blocks the peer will read for this request.

        Keeps the digest list aligned 1:1 with the ``remote_block_ids``
        that ``request_finished`` advertises: exchange-clip exotic groups
        (SWA in-window tail, SSM state slot) the same way, then trim
        attention groups to the blocks covering the prompt. Allocation
        can run past the prompt (e.g. the generated token's slot); the
        peer never reads those trailing blocks.
        """
        clipped = self.get_exchange_clipped_blocks(block_ids)
        return tuple(
            ids[: cdiv(num_prompt_tokens, group.kv_cache_spec.block_size)]
            if isinstance(group.kv_cache_spec, AttentionSpec)
            and not isinstance(
                group.kv_cache_spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            )
            else ids
            for group, ids in zip(self.kv_cache_config.transfer_groups, clipped)
        )

    def has_pending_push_work(self) -> bool:
        return False

    ############################################################
    # Abstract methods that subclasses must implement
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        raise NotImplementedError

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError
