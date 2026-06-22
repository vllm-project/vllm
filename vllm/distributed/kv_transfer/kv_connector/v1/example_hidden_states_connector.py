# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

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
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

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
    block_size = kv_cache.shape[1]
    return kv_cache[slot_mapping // block_size, slot_mapping % block_size][:num_tokens]


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
        data = load_file(path, device="cpu")
    return data


def cleanup_hidden_states(path: str, keep_hidden_states: bool = False) -> None:
    """Clean up hidden states file and lock file after loading.

    If keep_hidden_states is True, only removes the lock file
    and keeps the hidden states file.
    """
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


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[PendingSave] = field(default_factory=list)
    # req_id → filename for newly scheduled requests — the worker pre-creates
    # lock files for these so the lock exists before the client receives the
    # output path.
    new_req_filenames: dict[str, str] = field(default_factory=dict)


class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """
    Simple debug implementation of a HiddenStatesConnector.

    Simply extracts the hidden states from the kv cache and stores them to disk.
    Must be used in conjunction with the `extract_hidden_states` spec decoding method.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """
        Indicates whether this connector prefers KV blocks that hold KV data for all
        layers, which can speed up KV data transfers. Defaults to False.
        """
        # Must be False so that drafter kv cache isn't merged with verifier's
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
        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        # Scheduler-side state
        self._pending_saves: dict[str, PendingSave] = {}
        self._request_filenames: dict[str, str] = {}

        # Worker-side state (set by register_kv_caches).
        self._kv_cache: torch.Tensor | None = None

        # Identify which KV cache group holds the hidden-states layer.
        self._hs_group_idx: int = 0
        if self._kv_cache_config is not None:
            for i, group in enumerate(self._kv_cache_config.kv_cache_groups):
                if any("cache_only_layers" in n for n in group.layer_names):
                    self._hs_group_idx = i
                    break
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

    def _get_copy_stream(self) -> torch.cuda.Stream:
        """Lazily create the copy stream (CUDA must be initialized)."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass  # Store-only connector — nothing to load

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

    def _on_write_done(self, req_id: str, future: Future) -> None:
        """Surface any exception from the disk-write thread and drop the
        completed future from the in-flight tracking dict."""
        self._req_futures.pop(req_id, None)
        exc = future.exception()
        if exc is not None:
            logger.error("Hidden-states write failed for req_id=%s: %r", req_id, exc)

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

        # Find the KV cache group index for hidden states
        if self._kv_cache_config is not None:
            for i, group in enumerate(self._kv_cache_config.kv_cache_groups):
                if self.cache_layers[0] in group.layer_names:
                    self._hs_group_idx = i
                    break

    @staticmethod
    def _write_tensors(
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
    ) -> None:
        """Thread worker: wait for async DtoH copy, write to disk, release lock.

        ``lock_fd`` is an open file descriptor on the companion ``.lock``
        file with ``LOCK_EX`` already held.  Closing it releases the lock,
        which unblocks any client sleeping on ``LOCK_SH``.
        """
        try:
            event.synchronize()
            save_file(tensors, filename)
        finally:
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

        # Compute slot mapping from block_ids
        block_ids_t = torch.tensor(pending.block_ids, dtype=torch.long)
        num_blocks = block_ids_t.shape[0]
        block_offsets = torch.arange(0, self._block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids_t.reshape((num_blocks, 1)) * self._block_size
        )
        slot_mapping = slot_mapping.flatten()

        num_tokens = pending.token_ids.shape[0]

        copy_stream = self._get_copy_stream()

        # Ensure the copy stream sees all prior writes on the default stream.
        ready_event = torch.cuda.Event()
        ready_event.record()
        copy_stream.wait_event(ready_event)

        with torch.cuda.stream(copy_stream):
            # Move the CPU slot_mapping to GPU on the copy stream so the
            # implicit H2D inside fancy indexing doesn't sync the default
            # stream.
            slot_mapping_gpu = slot_mapping.to(
                device=self._kv_cache.device, non_blocking=True
            )
            hidden_states_gpu = extract_from_kv_cache(
                self._kv_cache, slot_mapping_gpu, num_tokens
            )
            # Async DtoH copy into pinned host memory.
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

        # Submit to thread pool for disk write.
        prior = self._req_futures.get(pending.req_id)
        assert prior is None, "Found another KV transfer request with same req_id!"

        os.makedirs(os.path.dirname(pending.filename), exist_ok=True)

        # Use the pre-created lock fd from wait_for_save (already holds
        # LOCK_EX). Falls back to creating one here if use_lock is True
        # but no pre-created fd exists (shouldn't happen in normal flow).
        lock_fd = self._lock_fds.pop(pending.req_id, None)
        if lock_fd is None and self.use_lock:
            lock_path = pending.filename + ".lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

        future = self._executor.submit(
            self._write_tensors, tensors, copy_done, pending.filename, lock_fd
        )
        self._req_copy_events[pending.req_id] = copy_done
        self._req_futures[pending.req_id] = future
        future.add_done_callback(partial(self._on_write_done, pending.req_id))

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
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
        # This connector is store-only, so we don't need to load any tokens
        return 0, False

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

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        Returns True to delay block freeing until get_finished extracts
        the hidden states from the KV cache.
        """
        req_id = request.request_id
        filename = self._request_filenames.pop(req_id)
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
        if self.has_connector_metadata():
            connector_metadata = self._get_connector_metadata()
            if isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata):
                for pending in connector_metadata.pending_saves:
                    self._submit_async_write(pending)

        # Poll for completed DtoH copies
        self._accumulated_finished_req_ids.update(finished_req_ids)

        done_sending: set[str] = set()
        for req_id in list(self._accumulated_finished_req_ids):
            event = self._req_copy_events.get(req_id)
            if event is None or event.query():
                self._req_copy_events.pop(req_id, None)
                done_sending.add(req_id)
                self._accumulated_finished_req_ids.discard(req_id)
                # Clean up any leftover lock fds (e.g. aborted requests
                # that never went through _submit_async_write).
                lock_fd = self._lock_fds.pop(req_id, None)
                if lock_fd is not None:
                    os.close(lock_fd)

        return done_sending or None, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[self._hs_group_idx])

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
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
        # NHD means we have (num_tokens, num_heads)
        # HND means we have (num_heads, num_tokens)
        # For now, we only support NHD layout since this keeps the
        # hidden states for each token together in memory.
        # HND is primarily used when sharding heads across devices.
        return "NHD"
