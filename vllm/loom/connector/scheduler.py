# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Iterable
import hashlib
import json
import time
from itertools import islice
from typing import Any

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .metadata import (
    LoomConnectorMetadata,
    LoomSharedPrefixHandshake,
    ReqId,
    RequestPhase,
)
from .policy import LoomPolicy
from ..logger import get_loom_logger

logger = get_loom_logger(__name__)


class LoomConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = self.offloaded_block_size // self.gpu_block_size
        self.manager: OffloadingManager = spec.get_manager()

        self.policy = LoomPolicy(
            offloaded_block_size=self.offloaded_block_size,
            block_size_factor=self.block_size_factor,
            manager=self.manager,
        )

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

        self._request_phases: dict[ReqId, RequestPhase] = {}

        self._timing: dict[ReqId, dict[str, float | int | bool | None]] = {}

        # MVP-0: request-level recompute (token_ids-only seed).
        # If a request is marked for recompute, we will return 0 external
        # tokens so vLLM falls back to local compute.
        loom_cfg = getattr(spec, "loom_config", None)
        if loom_cfg is None:
            raise ValueError(
                "LoomConnectorScheduler requires LoomOffloadingSpec (missing spec.loom_config)"
            )

        recompute_ratio_raw: object = getattr(loom_cfg, "loom_recompute_ratio", 0.0)
        disable_store_raw: object = getattr(loom_cfg, "loom_disable_store_for_recompute", False)
        load_only_raw: object = getattr(loom_cfg, "loom_load_only", False)
        log_every_raw: object = getattr(loom_cfg, "loom_recompute_log_every_steps", 50)

        self._loom_recompute_auto: bool = False
        if isinstance(recompute_ratio_raw, str):
            if recompute_ratio_raw != "auto":
                raise ValueError(
                    "loom_recompute_ratio must be a float in [0.0, 1.0] or 'auto'"
                )
            self._loom_recompute_auto = True
            self._loom_recompute_ratio = 0.0
        else:
            self._loom_recompute_ratio = float(recompute_ratio_raw)
            if not (0.0 <= self._loom_recompute_ratio <= 1.0):
                raise ValueError(
                    "loom_recompute_ratio must be a float in [0.0, 1.0] or 'auto'"
                )

        self._loom_disable_store_for_recompute: bool = bool(disable_store_raw)
        self._loom_load_only: bool = bool(load_only_raw)
        self._loom_force_recompute: dict[ReqId, bool] = {}

        self._loom_step_counter: int = 0
        try:
            self._loom_recompute_log_every_steps: int = int(log_every_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "loom_recompute_log_every_steps must be an int (>=0)"
            ) from e
        if self._loom_recompute_log_every_steps < 0:
            raise ValueError("loom_recompute_log_every_steps must be >= 0")

    def set_xfer_handshake_metadata(self, metadata: dict[int, object]) -> None:
        """Receive worker-side handshake metadata.

        For Loom, workers ingest shared-prefix KV into pinned CXL/DRAM buffers
        and allocate extents via LoomManager. The scheduler must learn the
        (prefix_id, layer_group_id) -> (base_block_id, num_blocks) mapping so
        prefix lookup/load can be planned.
        """

        # Merge all ranks' handshake payloads.
        extents: list[dict[str, int]] = []
        for tp_rank, payload in (metadata or {}).items():
            if payload is None:
                continue
            if not isinstance(payload, LoomSharedPrefixHandshake):
                # Ignore other connector payloads.
                continue
            extents.extend(payload.extents)

        if not extents:
            return

        manager = self.manager
        directory = getattr(manager, "shared_prefix_directory", None)
        if not isinstance(directory, dict):
            logger.warning(
                "Loom scheduler cannot apply shared prefix handshake: manager has no shared_prefix_directory"
            )
            return

        # Populate directory and reserve block ranges.
        reserved_total = 0
        for ent in extents:
            try:
                prefix_id = int(ent["prefix_id"])
                layer_group_id = int(ent["layer_group_id"])
                base_block_id = int(ent["base_block_id"])
                num_blocks = int(ent["num_blocks"])
                layout_version = int(ent.get("layout_version", 0))
            except Exception:
                continue

            key = (prefix_id, layer_group_id)
            if key not in directory:
                # Create SharedPrefixExtent using the manager's class.
                extent_cls = type(next(iter(directory.values()))) if directory else None
                if extent_cls is not None:
                    directory[key] = extent_cls(
                        base_block_id=base_block_id,
                        num_blocks=num_blocks,
                        layout_version=layout_version,
                    )
                else:
                    # Fallback: import local dataclass type.
                    from ..kv_offload.manager import SharedPrefixExtent

                    directory[key] = SharedPrefixExtent(
                        base_block_id=base_block_id,
                        num_blocks=num_blocks,
                        layout_version=layout_version,
                    )

            backend = getattr(manager, "cxl_backend", None)
            reserve_fn = getattr(backend, "reserve_extent", None)
            if callable(reserve_fn):
                reserve_fn(base_block_id, num_blocks)
                reserved_total += num_blocks

        logger.debug(
            "Loom shared prefix handshake applied: extents=%d reserved_blocks=%d",
            len(extents),
            reserved_total,
        )

    def _should_force_recompute(self, req_id: ReqId) -> bool:
        forced = self._loom_force_recompute.get(req_id)
        if forced is not None:
            return forced

        if self._loom_recompute_auto:
            forced = False
            self._loom_force_recompute[req_id] = forced
            return forced

        ratio = self._loom_recompute_ratio
        if ratio <= 0.0:
            forced = False
        elif ratio >= 1.0:
            forced = True
        else:
            # Deterministic split by req_id.
            digest = hashlib.sha256(req_id.encode("utf-8")).digest()
            val = int.from_bytes(digest[:8], "little", signed=False)
            forced = (val % 10_000) < int(ratio * 10_000)

        self._loom_force_recompute[req_id] = forced
        return forced

    def _refresh_request_phases(self) -> None:
        for req_id, req in self._requests.items():
            new_phase = (
                RequestPhase.PREFILL
                if req.num_computed_tokens < req.num_prompt_tokens
                else RequestPhase.DECODE
            )
            old_phase = self._request_phases.get(req_id)
            if old_phase is not None and old_phase != new_phase:
                if new_phase == RequestPhase.DECODE:
                    now = time.perf_counter()
                    stats = self._timing.get(req_id)
                    if stats is not None and stats.get("decode_start_ts") is None:
                        stats["decode_start_ts"] = now
                logger.debug(
                    "Request %s phase %s -> %s (num_computed_tokens=%d, num_prompt_tokens=%d)",
                    req_id,
                    old_phase,
                    new_phase,
                    req.num_computed_tokens,
                    req.num_prompt_tokens,
                )
            self._request_phases[req_id] = new_phase

    def _get_block_hashes(
        self,
        req: Request,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterable[BlockHash]:
        return islice(
            req.block_hashes,
            self.block_size_factor * start_idx + self.block_size_factor - 1,
            self.block_size_factor * end_idx if end_idx else None,
            self.block_size_factor,
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        if self._should_force_recompute(request.request_id):
            return 0, False

        kv_params = getattr(request, "kv_transfer_params", None)
        shared_prefix_id: int | None = None
        shared_prefix_len: int | None = None
        if isinstance(kv_params, dict):
            pid = kv_params.get("shared_prefix_id")
            plen = kv_params.get("shared_prefix_len")
            if pid is not None:
                try:
                    shared_prefix_id = int(pid)
                except (TypeError, ValueError):
                    shared_prefix_id = None
            if plen is not None:
                try:
                    shared_prefix_len = int(plen)
                except (TypeError, ValueError):
                    shared_prefix_len = None

        if shared_prefix_id is not None and shared_prefix_len is not None:
            if shared_prefix_len <= 0:
                return 0, False
            num_blocks = shared_prefix_len // self.offloaded_block_size
            if num_blocks <= 0:
                return 0, False
            full_block_tokens = self.offloaded_block_size * num_blocks
            if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
                return 0, False
            start_block_idx = num_computed_tokens // self.offloaded_block_size
            logger.debug(
                "Loom prefix lookup: req_id=%s prefix_id=%d prefix_len=%d start_block_idx=%d num_blocks=%d computed_tokens=%d",
                request.request_id,
                shared_prefix_id,
                shared_prefix_len,
                start_block_idx,
                num_blocks,
                num_computed_tokens,
            )
            hits = self.manager.lookup_prefix(
                prefix_id=shared_prefix_id,
                start_block_idx=start_block_idx,
                num_blocks=num_blocks,
                extra=kv_params,
            )
            logger.debug(
                "Loom prefix lookup result: req_id=%s prefix_id=%d hits=%d",
                request.request_id,
                shared_prefix_id,
                hits,
            )
        else:
            num_blocks = request.num_tokens // self.offloaded_block_size

            assert len(request.block_hashes) // self.block_size_factor == num_blocks
            block_hashes = self._get_block_hashes(request)

            self.manager.touch(block_hashes)

            full_block_tokens = self.offloaded_block_size * num_blocks
            if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
                # we can load less than a block, skip
                return 0, False

            start_block_idx = num_computed_tokens // self.offloaded_block_size
            hits = self.manager.lookup(
                self._get_block_hashes(request, start_idx=start_block_idx)
            )
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        self._request_phases[request.request_id] = RequestPhase.PREFILL
        # the block ids are updated in _get_reqs_to_store
        self._request_block_ids[request.request_id] = []

        self._timing.setdefault(
            request.request_id,
            {
                "arrival_time": request.arrival_time,
                "forced_recompute": self._should_force_recompute(request.request_id),
                "num_prompt_tokens": int(request.num_prompt_tokens),
                "load_enqueue_ts": None,
                "load_done_ts": None,
                "decode_start_ts": None,
                "finish_ts": None,
                "num_loaded_blocks": None,
            },
        )

        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert num_external_tokens == num_pending_gpu_blocks * self.gpu_block_size

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size

        kv_params = getattr(request, "kv_transfer_params", None)
        shared_prefix_id: int | None = None
        shared_prefix_len: int | None = None
        if isinstance(kv_params, dict):
            pid = kv_params.get("shared_prefix_id")
            plen = kv_params.get("shared_prefix_len")
            if pid is not None:
                try:
                    shared_prefix_id = int(pid)
                except (TypeError, ValueError):
                    shared_prefix_id = None
            if plen is not None:
                try:
                    shared_prefix_len = int(plen)
                except (TypeError, ValueError):
                    shared_prefix_len = None

        if shared_prefix_id is not None and shared_prefix_len is not None:
            logger.debug(
                "Loom prefix prepare_load: req_id=%s prefix_id=%d prefix_len=%d start_block_idx=%d num_blocks=%d num_external_tokens=%d",
                request.request_id,
                shared_prefix_id,
                shared_prefix_len,
                start_block_idx,
                num_blocks,
                num_external_tokens,
            )
            src_spec = self.manager.prepare_load_prefix(
                prefix_id=shared_prefix_id,
                start_block_idx=start_block_idx,
                num_blocks=num_blocks,
                extra=kv_params,
            )
            block_hashes = ()
        else:
            assert len(request.block_hashes) // self.block_size_factor >= num_blocks
            block_hashes = self._get_block_hashes(
                request, start_idx=start_block_idx, end_idx=num_blocks
            )
            src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(block_ids[num_computed_gpu_blocks:])

        if shared_prefix_id is not None and shared_prefix_len is not None:
            src_block_ids = getattr(src_spec, "block_ids", None)
            src_summary = f"{type(src_spec).__name__}"
            if isinstance(src_block_ids, (list, tuple)) and len(src_block_ids) > 0:
                src_summary = (
                    f"{type(src_spec).__name__}(n={len(src_block_ids)} first={src_block_ids[0]} last={src_block_ids[-1]})"
                )
            elif src_block_ids is not None:
                try:
                    n_src = len(src_block_ids)
                except TypeError:
                    n_src = None
                if n_src is not None and n_src > 0:
                    try:
                        first = src_block_ids[0]
                        last = src_block_ids[-1]
                    except Exception:
                        first = None
                        last = None
                    src_summary = (
                        f"{type(src_spec).__name__}(n={n_src} first={first} last={last})"
                    )

            dst_block_ids = getattr(dst_spec, "block_ids", None)
            try:
                dst_len = 0 if dst_block_ids is None else len(dst_block_ids)
            except TypeError:
                dst_len = 0
            logger.debug(
                "Loom prefix load plan: req_id=%s src=%s dst_gpu_blocks=%d",
                request.request_id,
                src_summary,
                dst_len,
            )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        if block_hashes:
            self._reqs_being_loaded[request.request_id].update(block_hashes)
        else:
            self._reqs_being_loaded.setdefault(request.request_id, set())
        self._next_stored_block_idx[request.request_id] = num_blocks

        stats = self._timing.get(request.request_id)
        if stats is not None and stats.get("load_enqueue_ts") is None:
            stats["load_enqueue_ts"] = time.perf_counter()

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> LoomConnectorMetadata:
        self._refresh_request_phases()

        # MVP-0 observability: verify forced recompute ratio is taking effect.
        self._loom_step_counter += 1
        if self._loom_recompute_log_every_steps > 0 and (
            self._loom_step_counter % self._loom_recompute_log_every_steps == 0
        ):
            req_ids = list(scheduler_output.num_scheduled_tokens)
            total = len(req_ids)
            if total > 0:
                forced = sum(1 for req_id in req_ids if self._should_force_recompute(req_id))
                logger.info(
                    "Loom recompute stats: forced=%d total=%d ratio=%.3f (cfg=%r)",
                    forced,
                    total,
                    forced / total,
                    ("auto" if self._loom_recompute_auto else self._loom_recompute_ratio),
                )
        num_prefill = 0
        num_decode = 0
        for req_id in scheduler_output.num_scheduled_tokens:
            phase = self._request_phases.get(req_id)
            if phase == RequestPhase.DECODE:
                num_decode += 1
            else:
                num_prefill += 1
        # if scheduler_output.num_scheduled_tokens:
        #     logger.debug(
        #         "Loom request phase stats: prefill=%d decode=%d total=%d",
        #         num_prefill,
        #         num_decode,
        #         num_prefill + num_decode,
        #     )

        if self._loom_load_only:
            reqs_to_store = {}
        else:
            reqs_to_store = self.policy.get_reqs_to_store(
                scheduler_output,
                requests=self._requests,
                request_block_ids=self._request_block_ids,
                next_stored_block_idx=self._next_stored_block_idx,
                reqs_being_stored=self._reqs_being_stored,
                get_block_hashes=self._get_block_hashes,
                request_phases=self._request_phases,
            )
            if self._loom_disable_store_for_recompute and reqs_to_store:
                reqs_to_store = {
                    req_id: spec
                    for req_id, spec in reqs_to_store.items()
                    if not self._should_force_recompute(req_id)
                }

        meta = LoomConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=reqs_to_store,
            reqs_to_regen={},
        )
        self._reqs_to_load = {}

        # NOTE (orozery): we should move this logic to update_connector_output
        # once KVConnectorOutput allows us to report completed transfers
        if not self._loom_load_only:
            for req_id in scheduler_output.preempted_req_ids or ():
                block_hashes = self._reqs_being_stored.get(req_id)
                if block_hashes:
                    self.manager.complete_store(block_hashes)
                    block_hashes.clear()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        for req_id in connector_output.finished_sending or []:
            block_hashes = self._reqs_being_stored.pop(req_id, None)
            if block_hashes:
                self.manager.complete_store(block_hashes)

        for req_id in connector_output.finished_recving or []:
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                self.manager.complete_load(block_hashes)

            stats = self._timing.get(req_id)
            if stats is not None and stats.get("load_done_ts") is None:
                stats["load_done_ts"] = time.perf_counter()
                stats["num_loaded_blocks"] = len(block_hashes) if block_hashes else 0

    def request_finished(
        self,
        request: Request,
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
        req_id = request.request_id
        now = time.perf_counter()
        stats = self._timing.pop(req_id, None)
        if stats is not None:
            stats["finish_ts"] = now
            load_enqueue_ts = stats.get("load_enqueue_ts")
            load_done_ts = stats.get("load_done_ts")
            decode_start_ts = stats.get("decode_start_ts")
            finish_ts = stats.get("finish_ts")
            if isinstance(load_enqueue_ts, float) and isinstance(load_done_ts, float):
                stats["load_ms"] = (load_done_ts - load_enqueue_ts) * 1e3
            if isinstance(decode_start_ts, float) and isinstance(finish_ts, float):
                stats["decode_wall_ms"] = (finish_ts - decode_start_ts) * 1e3
            # logger.info("loom_timing %s", json.dumps({"req_id": req_id, **stats}, sort_keys=True))

        self._requests.pop(req_id, None)
        self._request_phases.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        self._loom_force_recompute.pop(req_id, None)

        if self._loom_load_only:
            return False, None

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(block_hashes=event.block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=event.block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                    lora_name=None,
                )
