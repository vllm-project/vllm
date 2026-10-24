# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, cast

import torch
from packaging.version import Version
from safetensors.torch import load_file, save_file

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hidden_state_window import (
    HiddenStateWindow,
    block_window_ranges,
    extraction_layout,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Extract data from KV cache."""
    block_size = kv_cache.shape[2]
    return kv_cache[slot_mapping // block_size, :, slot_mapping % block_size][
        :num_tokens
    ]


def load_hidden_states(path: str) -> dict[str, torch.Tensor]:
    """Load hidden states written by ExampleHiddenStatesConnector.

    Blocks (without polling) until the async write is complete by
    acquiring a shared flock on the companion lock file.  The kernel
    puts the caller to sleep until the writer releases its exclusive lock.

    Args:
        path: The file path returned in kv_transfer_params["hidden_states_path"].

    Returns:
        Dict with "hidden_states" and "token_ids" tensors.

    """
    lock_path = path + ".lock"
    with open(lock_path) as lf:
        fcntl.flock(lf, fcntl.LOCK_SH)  # sleeps until writer releases LOCK_EX
        if os.path.exists(path + ".error"):
            with open(path + ".error") as error_file:
                raise RuntimeError("Hidden-state export failed: " + error_file.read())
        data = load_file(path, device="cpu")
    return data


def cleanup_hidden_states(path: str, keep_hidden_states: bool = False) -> None:
    """Clean up hidden states file and lock file after loading.

    If keep_hidden_states is True, only removes the lock file
    and keeps the hidden states file.
    """
    error_path = path + ".error"
    if os.path.exists(error_path):
        os.remove(error_path)
    lock_path = path + ".lock"
    if os.path.exists(lock_path):
        os.remove(lock_path)
    if not keep_hidden_states and os.path.exists(path):
        os.remove(path)


@dataclass
class PendingSave:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    block_ids: list[int]
    positions: torch.Tensor | None = None
    response_positions: torch.Tensor | None = None
    layer_ids: torch.Tensor | None = None
    window_bounds: torch.Tensor | None = None
    target_token_ids: torch.Tensor | None = None


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[PendingSave] = field(default_factory=list)
    # req_id → filename for newly scheduled requests — the worker pre-creates
    # lock files for these so the lock exists before the client receives the
    # output path.
    new_req_filenames: dict[str, str] = field(default_factory=dict)
    block_windows: list[tuple[int, int, int]] = field(default_factory=list)
    cancelled_req_ids: set[str] = field(default_factory=set)


class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """Simple debug implementation of a HiddenStatesConnector.

    Simply extracts the hidden states from the kv cache and stores them to disk.
    Must be used in conjunction with the `extract_hidden_states` spec decoding method.
    """

    def get_finished_count(self) -> int:
        """Number of ``finished_sending`` notifications expected per request.

        Only TP rank 0 writes hidden states to disk, so each request yields a
        single notification. The default (``None``) makes ``KVOutputAggregator``
        wait for every worker in the world size; since ranks 1..N-1 never submit
        a copy, the request would never be reported as finished and its blocks
        would be leaked forever (the scheduler then spins on
        ``has_finished_requests()``).
        """
        return 1

    @classmethod
    def _find_cache_kv_group_id(cls, kv_cache_config: "KVCacheConfig | None") -> int:
        """Index of the KV cache group holding the extracted hidden states.

        Located by spec type so it resolves on both scheduler and worker side.
        """
        if kv_cache_config is None:
            return 0

        from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

        groups = kv_cache_config.kv_cache_groups
        group_ids = [
            gid
            for gid, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        ]
        if len(group_ids) == 1:
            return group_ids[0]
        if not group_ids and len(groups) == 1:
            return 0
        raise ValueError(
            "Could not uniquely identify the extract-hidden-states KV cache "
            f"group among {len(groups)} groups; the hidden-states layer must be "
            "isolated in its own group (MLA verifiers are unsupported)."
        )

    @staticmethod
    def _get_cache_block_size(
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig | None",
        cache_kv_group_id: int,
    ) -> int:
        """Block size of the hidden-states group, read from its own spec.

        cache_config.block_size is bumped to a common multiple for hybrid
        verifiers; the page-aligned hidden-states group keeps a smaller one.
        """
        if kv_cache_config is None:
            return vllm_config.cache_config.block_size
        cache_group = kv_cache_config.kv_cache_groups[cache_kv_group_id]
        return cache_group.kv_cache_spec.block_size

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
        # Read the hidden-states group and its block size from the group spec;
        # cache_config.block_size is bumped (wrong) for hybrid verifiers.
        self._cache_kv_group_id = self._find_cache_kv_group_id(kv_cache_config)
        self._block_size = self._get_cache_block_size(
            vllm_config, kv_cache_config, self._cache_kv_group_id
        )
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers: list[str] = []  # set by self.register_kv_caches
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

        if Version(version("safetensors")) < Version("0.8.0"):
            logger.warning(
                "safetensors < 0.8.0 holds the GIL during save_file, which "
                "serializes the writer thread pool and hurts throughput. "
                "Upgrade to safetensors >= 0.8.0 for better performance."
            )

        assert self._vllm_config.speculative_config is not None, (
            "ExampleHiddenStatesConnector only works when using "
            "'extract_hidden_states' speculative method"
        )
        self._window_enabled = self._kv_transfer_config.get_from_extra_config(
            "enable_window_capture", False
        )
        if type(self._window_enabled) is not bool:
            raise ValueError("enable_window_capture must be a boolean")
        self._max_window_rows = self._kv_transfer_config.get_from_extra_config(
            "max_window_rows", vllm_config.model_config.max_model_len
        )
        if type(self._max_window_rows) is not int or self._max_window_rows < 1:
            raise ValueError("max_window_rows must be a positive integer")
        self._export_budget = self._kv_transfer_config.get_from_extra_config(
            "max_inflight_export_bytes", 64 * 1024 * 1024
        )
        if type(self._export_budget) is not int or self._export_budget < 1:
            raise ValueError("max_inflight_export_bytes must be a positive integer")
        if self._window_enabled:
            self._validate_window_config(vllm_config)
        self._layout, self._layer_ids = extraction_layout(vllm_config)
        self.num_hidden_states = len(self._layer_ids)
        if self._window_enabled:
            row_bytes = (
                self.num_hidden_states
                * vllm_config.model_config.get_hidden_size()
                * cast(torch.dtype, vllm_config.model_config.dtype).itemsize
            )
            if row_bytes > self._export_budget:
                raise ValueError("Export budget must fit at least one hidden-state row")
        self._capture_metrics = dict(
            copied_bytes=0,
            pinned_live_bytes_upper_bound_peak=0,
            export_submit_seconds=0.0,
            copy_wait_seconds=0.0,
            serialization_seconds=0.0,
        )
        self._metrics_lock = threading.Lock()
        self._worker_filenames: dict[str, str] = {}
        self._rejected_req_ids: set[str] = set()
        self._request_windows: dict[str, HiddenStateWindow | None] = {}
        self._request_block_counts: dict[str, int] = {}
        self._request_prompt_lengths: dict[str, int] = {}
        self._cancelled_req_ids: set[str] = set()
        self._window_ranges: torch.Tensor | None = None
        self._export_queue: deque[tuple[Future, int]] = deque()

        # Scheduler-side state
        self._pending_saves: dict[str, PendingSave] = {}
        self._request_filenames: dict[str, str] = {}

        # Worker-side state (set by register_kv_caches).
        self._kv_cache: torch.Tensor | None = None

        # Only TP rank 0 writes hidden states to disk; other TP ranks no-op.
        # Set in register_kv_caches (after distributed init).
        self._is_tp_rank_zero: bool = True

        # Async write infrastructure (worker-side).
        # Dedicated CUDA stream for DtoH copies so they don't block
        # the default stream (model forward). Thread pool for disk writes.
        self._copy_stream: torch.cuda.Stream | None = None  # lazy init
        self._executor = ThreadPoolExecutor(
            max_workers=self._kv_transfer_config.get_from_extra_config(
                "num_writer_threads", 8
            ),
            thread_name_prefix="vllm-hs-save",
        )
        # Whether to use a filesystem lock when writing files to shared storage.
        # This is necessary for online transfer clients to avoid incomplete reads,
        # but can be disabled for offline tasks that run tasks in batches to completion
        self.allow_custom_save_path = self._kv_transfer_config.get_from_extra_config(
            "allow_custom_save_path", False
        )
        if self.allow_custom_save_path:
            logger.warning(
                "allow_custom_save_path is enabled. API clients can write "
                "hidden states to arbitrary paths on the server filesystem. "
                "Only enable this with trusted clients."
            )
        self.use_lock = self._kv_transfer_config.get_from_extra_config(
            "use_synchronization_lock", True
        )
        # req_id → open fd on the .lock file with LOCK_EX held.
        # Pre-created in wait_for_save when a request first arrives,
        # consumed by _submit_async_write which passes the fd to the
        # thread pool worker for release after writing.
        self._lock_fds: dict[str, int] = {}
        # req_id → in-flight disk-write Future for that req_id.
        self._req_futures: dict[str, Future] = {}
        # req_id → CUDA event marking completion of the DtoH copy. Once
        # this event is complete the request is considered "done sending"
        # by get_finished; clients block on the per-file flock to wait for
        # the disk write itself.
        self._req_copy_events: dict[str, torch.cuda.Event] = {}
        # req_ids reported as finished-generating by the scheduler,
        # accumulated across get_finished calls.
        self._accumulated_finished_req_ids: set[str] = set()

    @staticmethod
    def _validate_window_config(config: "VllmConfig") -> None:
        if (
            config.speculative_config is None
            or config.speculative_config.method != "extract_hidden_states"
        ):
            raise ValueError(
                "Window capture requires extract_hidden_states; real "
                "speculative drafters need an independent capture hook"
            )
        parallel = config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise ValueError("Window capture currently requires PP=CP=1")
        if config.use_v2_model_runner:
            raise ValueError("Window capture currently requires Model Runner V1")
        if config.scheduler_config.async_scheduling:
            raise ValueError("Window capture currently requires async_scheduling=False")
        if config.model_config.is_encoder_decoder:
            raise ValueError("Window capture requires a decoder-only model")
        if config.device_config.device_type != "cuda":
            raise ValueError("Window capture currently requires a CUDA device")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("Window capture requires enable_prefix_caching=False")
        assert config.kv_transfer_config is not None
        if not config.kv_transfer_config.get_from_extra_config(
            "use_synchronization_lock", True
        ):
            raise ValueError("Window capture requires use_synchronization_lock=True")
        _, ids = extraction_layout(config)
        auxiliary = [i for i in ids if i >= 0]
        if auxiliary != sorted(auxiliary):
            raise ValueError(
                "Window capture requires auxiliary layer ids in model order"
            )

    def _reserve_export(self, size: int) -> None:
        while self._export_queue and self._export_queue[0][0].done():
            self._export_queue.popleft()
        while self._export_queue and (
            sum(n for _, n in self._export_queue) + size > self._export_budget
        ):
            future, _ = self._export_queue.popleft()
            wait([future])

    def get_window_capture_metrics(self) -> dict[str, int | float]:
        """Return worker-local counters; call after draining writes for final totals."""
        result = dict(self._capture_metrics)
        if self._kv_cache is not None:
            result["hidden_cache_allocated_bytes"] = (
                self._kv_cache.numel() * self._kv_cache.element_size()
            )
        return result

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        for fd in self._lock_fds.values():
            os.close(fd)
        self._lock_fds.clear()
        for filename in self._worker_filenames.values():
            if os.path.exists(filename + ".lock"):
                os.remove(filename + ".lock")
        self._worker_filenames.clear()
        self._export_queue.clear()

    def _get_copy_stream(self) -> torch.cuda.Stream:
        """Lazily create the copy stream (CUDA must be initialized)."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        if not self.has_connector_metadata():
            return
        meta = self._get_connector_metadata()
        assert isinstance(meta, ExampleHiddenStatesConnectorMetadata)
        if self._window_enabled and meta.block_windows and self._is_tp_rank_zero:
            assert self._window_ranges is not None
            updates = torch.tensor(
                meta.block_windows, dtype=torch.long, device=self._window_ranges.device
            )
            self._window_ranges[updates[:, 0]] = updates[:, 1:]
        for req_id in meta.cancelled_req_ids if self._is_tp_rank_zero else ():
            fd = self._lock_fds.pop(req_id, None)
            if fd is not None:
                os.close(fd)
            filename = self._worker_filenames.pop(req_id, None)
            if filename is not None and os.path.exists(filename + ".lock"):
                os.remove(filename + ".lock")

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # Store-only connector — nothing to load

    def wait_for_save(self) -> None:
        """Pre-create lock files for newly arrived requests.

        This runs on the worker BEFORE the scheduler returns the output
        path to the client, guaranteeing that the lock file exists (and
        LOCK_EX is held) by the time the client tries to open it.
        """
        if not self._is_tp_rank_zero:
            return
        if not self.use_lock or not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, ExampleHiddenStatesConnectorMetadata):
            return
        for req_id, filename in metadata.new_req_filenames.items():
            if req_id in self._lock_fds:
                continue
            lock_path = filename + ".lock"
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._lock_fds[req_id] = lock_fd
            self._worker_filenames[req_id] = filename

    def _on_write_done(self, req_id: str, future: Future) -> None:
        """Surface any exception from the disk-write thread and drop the
        completed future from the in-flight tracking dict."""
        self._req_futures.pop(req_id, None)
        exc = future.exception()
        if exc is not None:
            logger.error("Hidden-states write failed for req_id=%s: %r", req_id, exc)
        else:
            timings = future.result()
            if timings:
                with self._metrics_lock:
                    for key, value in timings.items():
                        self._capture_metrics[key] += value

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # Delay tp rank0 initialization until after distributed init
        self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        # Filter layers to only include CacheOnlyAttentionLayers
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )
        self._kv_cache = kv_caches[self.cache_layers[0]]
        if self._window_enabled:
            self._window_ranges = torch.zeros(
                (self._kv_cache.shape[0], 2),
                device=self._kv_cache.device,
                dtype=torch.long,
            )
            layers[
                self.cache_layers[0]
            ].hidden_state_window_ranges = self._window_ranges

        # Block size must match the indexed buffer, else reads hit the wrong
        # slots. Raise (not assert) so the check survives `python -O`.
        # Views are [num_blocks, num_heads, block_size, head_size], matching what
        # extract_from_kv_cache() indexes.
        if self._block_size != self._kv_cache.shape[2]:
            raise ValueError(
                f"Hidden-states block-size mismatch: derived {self._block_size} "
                f"but buffer block size is {self._kv_cache.shape[2]}; read slots "
                "would be wrong (likely a hybrid block-size resolution bug)."
            )

    @staticmethod
    def _write_tensors(
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
        file_metadata: dict[str, str] | None = None,
    ) -> dict[str, float] | None:
        """Thread worker: wait for async DtoH copy, write to disk, release lock.

        ``lock_fd`` is an open file descriptor on the companion ``.lock``
        file with ``LOCK_EX`` already held.  Closing it releases the lock,
        which unblocks any client sleeping on ``LOCK_SH``.
        """
        started = time.perf_counter() if file_metadata else 0.0
        try:
            event.synchronize()
            copied = time.perf_counter() if file_metadata else 0.0
            temporary = filename + ".incomplete"
            try:
                save_file(tensors, temporary, metadata=file_metadata)
                os.replace(temporary, filename)
                if os.path.exists(filename + ".error"):
                    os.remove(filename + ".error")
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
            if file_metadata:
                return {
                    "copy_wait_seconds": copied - started,
                    "serialization_seconds": time.perf_counter() - copied,
                }
            return None
        except Exception as error:
            try:
                with open(filename + ".error", "w") as error_file:
                    error_file.write(str(error))
            except OSError:
                logger.exception("Could not record hidden-state export failure")
            raise
        finally:
            if file_metadata is not None:
                # Drop staging ownership before Future.result unblocks budget waiters.
                tensors.clear()
            if lock_fd is not None:
                os.close(lock_fd)  # releases LOCK_EX

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # Hidden states are already cached by CacheOnlyAttentionLayer during
        # forward. Extraction happens in get_finished once all tokens are done.
        pass

    def _submit_async_write(
        self,
        pending: PendingSave,
    ) -> None:
        """Extract hidden states from KV cache and submit async DtoH + disk write.

        Called from get_finished for each request that has finished generating.
        """
        if not self._is_tp_rank_zero:
            return
        assert self._kv_cache is not None

        started = time.perf_counter() if pending.positions is not None else 0.0
        self._worker_filenames.pop(pending.req_id, None)
        # Compute slot mapping from block_ids
        block_ids_t = torch.tensor(pending.block_ids, dtype=torch.long)
        if pending.positions is not None:
            slot_mapping = (
                block_ids_t[pending.positions // self._block_size] * self._block_size
                + pending.positions % self._block_size
            )
        else:
            block_offsets = torch.arange(self._block_size, dtype=torch.long)
            slot_mapping = (
                block_offsets[None, :] + block_ids_t[:, None] * self._block_size
            ).flatten()

        num_tokens = pending.token_ids.shape[0]
        slot_mapping = slot_mapping[:num_tokens]
        row_bytes = self._kv_cache[0, 0].numel() * self._kv_cache.element_size()
        export_bytes = num_tokens * row_bytes
        bounded = pending.positions is not None
        if bounded:
            if row_bytes > self._export_budget:
                raise ValueError("Export budget must fit at least one hidden-state row")
            self._reserve_export(min(export_bytes, self._export_budget))

        copy_stream = self._get_copy_stream()

        # Ensure the copy stream sees all prior writes on the default stream.
        ready_event = torch.cuda.Event()
        ready_event.record()
        copy_stream.wait_event(ready_event)

        with torch.cuda.stream(copy_stream):
            # Move the CPU slot_mapping to GPU on the copy stream so the
            # implicit H2D inside fancy indexing doesn't sync the default
            # stream. Deliberate bulk transfer in this reference connector.
            with gpu_sync_allowed():
                slot_mapping_gpu = slot_mapping.to(
                    device=self._kv_cache.device, non_blocking=True
                )
            if bounded and export_bytes > self._export_budget:
                # Large windows use bounded staging and pageable final storage.
                rows = max(1, self._export_budget // row_bytes)
                pinned_hs = torch.empty(
                    (num_tokens, *self._kv_cache.shape[2:]), dtype=self._kv_cache.dtype
                )
                staging = torch.empty(
                    (rows, *self._kv_cache.shape[2:]),
                    dtype=self._kv_cache.dtype,
                    pin_memory=True,
                )
                for begin in range(0, num_tokens, rows):
                    end = min(num_tokens, begin + rows)
                    chunk = extract_from_kv_cache(
                        self._kv_cache, slot_mapping_gpu[begin:end], end - begin
                    )
                    staging[: end - begin].copy_(chunk, non_blocking=True)
                    copy_stream.synchronize()
                    pinned_hs[begin:end].copy_(staging[: end - begin])
                    del chunk
            else:
                hidden_states_gpu = extract_from_kv_cache(
                    self._kv_cache, slot_mapping_gpu, num_tokens
                )
                pinned_hs = torch.empty_like(
                    hidden_states_gpu, device="cpu", pin_memory=True
                )
                pinned_hs.copy_(hidden_states_gpu, non_blocking=True)

        # Record completion of this copy on the copy stream.
        copy_done = torch.cuda.Event()
        copy_done.record(copy_stream)

        # token_ids is already on CPU (created in request_finished).
        assert not pending.token_ids.is_cuda, (
            "Expected token_ids on CPU, got CUDA tensor"
        )
        tensors = {
            "hidden_states": pinned_hs,
            "token_ids": pending.token_ids.clone(),
        }

        if pending.positions is not None:
            tensors.update(
                hidden_positions=pending.positions,
                response_positions=pending.response_positions,
                layer_ids=pending.layer_ids,
                hidden_window=pending.window_bounds,
                target_token_ids=pending.target_token_ids,
                final_is_normalized=torch.tensor(self._layout != "auxiliary"),
            )

        # Submit to thread pool for disk write.
        prior = self._req_futures.get(pending.req_id)
        assert prior is None, "Found another KV transfer request with same req_id!"

        os.makedirs(os.path.dirname(pending.filename), exist_ok=True)

        # Use the pre-created lock fd from wait_for_save (already holds
        # LOCK_EX). Falls back to creating one here if use_lock is True
        # but no pre-created fd exists (shouldn't happen in normal flow).
        lock_fd = self._lock_fds.get(pending.req_id)
        if lock_fd is None and self.use_lock:
            lock_path = pending.filename + ".lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._lock_fds[pending.req_id] = lock_fd

        future = self._executor.submit(
            self._write_tensors,
            tensors,
            copy_done,
            pending.filename,
            lock_fd,
            (
                {
                    "request_id": pending.req_id,
                    "layout": self._layout,
                    "coordinate": "response_prediction",
                    "version": "1",
                }
                if bounded
                else None
            ),
        )
        self._lock_fds.pop(pending.req_id, None)
        self._req_copy_events[pending.req_id] = copy_done
        self._req_futures[pending.req_id] = future
        future.add_done_callback(partial(self._on_write_done, pending.req_id))
        if bounded:
            reserved = min(export_bytes, self._export_budget)
            self._export_queue.append((future, reserved))
            with self._metrics_lock:
                self._capture_metrics["copied_bytes"] += export_bytes
                self._capture_metrics["pinned_live_bytes_upper_bound_peak"] = max(
                    self._capture_metrics["pinned_live_bytes_upper_bound_peak"],
                    sum(n for _, n in self._export_queue),
                )
                self._capture_metrics["export_submit_seconds"] += (
                    time.perf_counter() - started
                )

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.

        """
        try:
            self._check_window_request(request)
        except ValueError as error:
            logger.warning(
                "Skipping hidden-state capture for %s: %s", request.request_id, error
            )
            self._rejected_req_ids.add(request.request_id)
        return 0, False

    def _check_window_request(self, request: "Request") -> None:
        window = HiddenStateWindow.from_params(request.kv_transfer_params or {})
        if window is None:
            return
        if not self._window_enabled:
            raise ValueError("Set enable_window_capture=True on the connector")
        if (
            request.prompt_token_ids is None
            or request.prompt_embeds is not None
            or request.resumable
        ):
            raise ValueError("Window capture requires a non-resumable token prompt")
        window.absolute_bounds(request.num_prompt_tokens)
        if window.end > self._max_window_rows:
            raise ValueError(f"Window end must not exceed {self._max_window_rows}")
        if request.sampling_params is None:
            raise ValueError("Window capture requires sampling parameters")
        if request.sampling_params.stop:
            raise ValueError(
                "Window capture does not yet support stop strings; "
                "use stop_token_ids or EOS"
            )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # Usually used to handle allocation of new blocks for requests that are loading
        # tokens from connector's external kv cache. We never load from external cache
        # so this is a no-op.
        assert num_external_tokens == 0, "This connector is store-only"

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
        meta = ExampleHiddenStatesConnectorMetadata()
        meta.cancelled_req_ids = self._cancelled_req_ids
        self._cancelled_req_ids = set()

        # Transfer pending saves into metadata (scheduler → worker bridge)
        meta.pending_saves = list(self._pending_saves.values())
        self._pending_saves.clear()

        # Resolve save paths for new requests and tell the worker so it can
        # pre-create lock files before the client receives the output path.
        for new_req in scheduler_output.scheduled_new_reqs:
            default_path = os.path.join(
                self._storage_path, f"{new_req.req_id}.safetensors"
            )
            kv_params = (
                new_req.sampling_params.extra_args.get("kv_transfer_params")
                if new_req.sampling_params and new_req.sampling_params.extra_args
                else None
            ) or {}
            custom_path = kv_params.get("hidden_states_path")
            if custom_path is not None and not self.allow_custom_save_path:
                logger.warning(
                    "Request %s provided hidden_states_path but "
                    "allow_custom_save_path is disabled. Ignoring "
                    "custom path and using default.",
                    new_req.req_id,
                )
                custom_path = None
            filename = custom_path or default_path
            self._request_filenames[new_req.req_id] = filename
            meta.new_req_filenames[new_req.req_id] = filename
            if self._window_enabled:
                window = (
                    None
                    if new_req.req_id in self._rejected_req_ids
                    else HiddenStateWindow.from_params(kv_params)
                )
                prompt_len = len(new_req.prompt_token_ids or [])
                self._request_windows[new_req.req_id] = window
                self._request_prompt_lengths[new_req.req_id] = prompt_len
                blocks = new_req.block_ids[self._cache_kv_group_id]
                bounds = window.absolute_bounds(prompt_len) if window else None
                meta.block_windows.extend(
                    block_window_ranges(blocks, self._block_size, bounds)
                )
                self._request_block_counts[new_req.req_id] = len(blocks)

        if self._window_enabled:
            cached = scheduler_output.scheduled_cached_reqs
            for req_id, groups in zip(cached.req_ids, cached.new_block_ids):
                if groups is None:
                    continue
                first = (
                    0
                    if req_id in cached.resumed_req_ids
                    else self._request_block_counts[req_id]
                )
                blocks = groups[self._cache_kv_group_id]
                window = self._request_windows[req_id]
                bounds = (
                    window.absolute_bounds(self._request_prompt_lengths[req_id])
                    if window
                    else None
                )
                meta.block_windows.extend(
                    block_window_ranges(blocks, self._block_size, bounds, first)
                )
                self._request_block_counts[req_id] = first + len(blocks)
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called exactly once when a request has finished, before its blocks are
        freed.

        Returns True to delay block freeing until get_finished extracts
        the hidden states from the KV cache.
        """
        req_id = request.request_id
        filename = self._request_filenames.pop(req_id, None)
        window = self._request_windows.pop(req_id, None)
        rejected = req_id in self._rejected_req_ids
        self._rejected_req_ids.discard(req_id)
        self._request_block_counts.pop(req_id, None)
        self._request_prompt_lengths.pop(req_id, None)
        if filename is None:
            # Request was aborted before it was ever scheduled, so
            # build_connector_meta never recorded a filename for it. There are
            # no hidden states to save and no blocks worth delaying.
            self._pending_saves.pop(req_id, None)
            return False, None
        if rejected:
            self._cancelled_req_ids.add(req_id)
            return False, {"hidden_state_capture_skip_reason": "invalid_window"}
        if self._window_enabled and request.status in (
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_ERROR,
            RequestStatus.FINISHED_IGNORED,
        ):
            self._cancelled_req_ids.add(req_id)
            return False, None
        if window is not None:
            start, end = window.accepted_bounds(
                request.num_prompt_tokens, len(request.output_token_ids)
            )
            if end - start < window.min_rows:
                self._cancelled_req_ids.add(req_id)
                return False, {"hidden_state_capture_skip_reason": "insufficient_rows"}
            positions = torch.arange(start, end, dtype=torch.long)
            response_positions = positions - request.num_prompt_tokens + 1
            tokens = list(request.all_token_ids)[start:end]
            targets = list(request.output_token_ids)[
                window.start : window.start + end - start
            ]
            self._pending_saves[req_id] = PendingSave(
                req_id,
                filename,
                torch.tensor(tokens, dtype=torch.long),
                list(block_ids),
                positions,
                response_positions,
                torch.tensor(self._layer_ids),
                torch.tensor([window.start, window.end]),
                torch.tensor(targets, dtype=torch.long),
            )
            return True, {
                "hidden_states_path": filename,
                "hidden_states_layout": self._layout,
                "hidden_state_capture_rows": end - start,
            }
        kv_params = request.kv_transfer_params or {}
        if kv_params.get("include_output_tokens", False):
            # Exclude the final token — it was the model's output, never an
            # input to a forward pass, so its hidden state is not in the cache.
            token_ids = torch.tensor(list(request.all_token_ids)[:-1])
        elif request.prompt_token_ids is not None:
            token_ids = torch.tensor(request.prompt_token_ids)
        else:
            logger.warning(
                "Request %s has no prompt_token_ids (prompt_embeds only). "
                "Saved token_ids will be empty.",
                req_id,
            )
            token_ids = torch.tensor([], dtype=torch.long)
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
        )
        return True, {"hidden_states_path": filename}

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Extract hidden states and poll DtoH-copy completion.

        On the worker side, connector metadata carries pending saves from the
        scheduler. For each one we extract from the KV cache and launch an
        async DtoH copy + thread-pool disk write.

        We then poll accumulated finished req_ids: a request is "done sending"
        once its DtoH copy event is complete. The subsequent disk write may
        still be in flight; clients block on the per-file flock to wait for it.
        """
        # Extract and submit async writes for newly finished requests
        new_pending_req_ids: set[str] = set()
        if self.has_connector_metadata():
            connector_metadata = self._get_connector_metadata()
            if isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata):
                for pending in connector_metadata.pending_saves:
                    new_pending_req_ids.add(pending.req_id)
                    try:
                        self._submit_async_write(pending)
                    except Exception:
                        if self._copy_stream is not None:
                            self._copy_stream.synchronize()
                        fd = self._lock_fds.pop(pending.req_id, None)
                        try:
                            with open(pending.filename + ".error", "w") as error_file:
                                error_file.write("Capture or D2H submission failed")
                        except OSError:
                            logger.exception("Could not record capture failure")
                        finally:
                            if fd is not None:
                                os.close(fd)
                        logger.exception(
                            "Hidden-state export failed for %s", pending.req_id
                        )

        # Only track requests whose block release was delayed for an export.
        # finished_req_ids also contains requests that request_finished() chose
        # not to delay, such as windows with too few accepted rows.
        eligible_req_ids = new_pending_req_ids | self._req_copy_events.keys()
        self._accumulated_finished_req_ids.update(
            finished_req_ids & eligible_req_ids
        )

        # Poll for completed DtoH copies

        done_sending: set[str] = set()
        for req_id in list(self._accumulated_finished_req_ids):
            event = self._req_copy_events.get(req_id)
            if event is not None and not event.query():
                continue  # DtoH copy still in flight
            self._req_copy_events.pop(req_id, None)
            self._accumulated_finished_req_ids.discard(req_id)
            # Only report requests we actually tracked for sending.
            if event is not None:
                done_sending.add(req_id)
            # Clean up any leftover lock fds (e.g. aborted requests that never
            # went through _submit_async_write).
            lock_fd = self._lock_fds.pop(req_id, None)
            if lock_fd is not None:
                os.close(lock_fd)

        return done_sending or None, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not 0 <= self._cache_kv_group_id < len(block_ids):
            # No per-group block table (e.g. a request that was never
            # scheduled); treat as a no-op instead of indexing out of bounds.
            return False, None
        return self.request_finished(request, block_ids[self._cache_kv_group_id])

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """Get the required KV cache layout for this connector.

        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.

        """
        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        # LBNHC means we have (num_tokens, num_heads)
        # LBHNC means we have (num_heads, num_tokens)
        # For now, we only support LBNHC layout since this keeps the
        # hidden states for each token together in memory.
        # LBHNC is primarily used when sharding heads across devices.
        return "LBNHC"
