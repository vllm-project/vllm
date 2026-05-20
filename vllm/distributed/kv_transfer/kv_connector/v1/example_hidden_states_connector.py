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
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

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
class ReqMeta:
    # Request ID
    req_id: str
    # Request filename
    filename: str
    # Request tokens
    token_ids: torch.Tensor
    # Whether this request is a new request or partially computed already
    new_req: bool

    @staticmethod
    def make_meta(
        req_id: str,
        filename: str,
        token_ids: list[int],
        new_req: bool,
    ) -> "ReqMeta":
        return ReqMeta(
            req_id=req_id,
            filename=filename,
            token_ids=torch.tensor(token_ids),
            new_req=new_req,
        )


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        filename: str,
        token_ids: list[int],
        new_req: bool = True,
    ) -> None:
        self.requests.append(ReqMeta.make_meta(req_id, filename, token_ids, new_req))


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

        self._request_filenames: dict[str, str] = {}
        self._active_requests: dict[str, NewRequestData] = {}
        self._req_blocks: dict[str, list[int]] = {}

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
        self.use_lock = self._kv_transfer_config.get_from_extra_config(
            "use_synchronization_lock", True
        )
        # (tensors_dict, copy_done_event, filename, req_id) queued by
        # save_kv_layer, submitted to thread pool by wait_for_save.
        self._pending_copies: list[
            tuple[dict[str, torch.Tensor], torch.cuda.Event, str, str]
        ] = []
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

    def wait_for_save(self):
        """Submit pending async copies to the thread pool for disk write.

        For each pending write we acquire an exclusive flock on a
        companion ``.lock`` file **before** submitting to the thread pool.
        The thread worker releases the lock after the data file is fully
        written.  Clients call :func:`load_hidden_states` which takes a
        shared flock — the kernel sleeps the client until the writer is
        done.  Because ``wait_for_save`` runs before the worker returns
        output to the scheduler, the lock file is guaranteed to exist
        (and be held) by the time the client receives the path.

        The lock can be disabled via the "use_synchronization_lock" extra config.
        """
        for tensors, event, filename, req_id in self._pending_copies:
            prior = self._req_futures.get(req_id)
            assert prior is None, "Found another KV transfer request with same req_id!"

            lock_fd = None
            if self.use_lock:
                # Create/open the lock file and acquire an exclusive lock.
                # The lock is held by this fd; the thread worker will close
                # the fd after writing, which releases the lock.
                lock_path = filename + ".lock"
                lock_fd = os.open(
                    lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644
                )
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

            future = self._executor.submit(
                self._write_tensors, tensors, event, filename, lock_fd
            )
            self._req_copy_events[req_id] = event
            self._req_futures[req_id] = future
            future.add_done_callback(partial(self._on_write_done, req_id))
        self._pending_copies.clear()

    def _on_write_done(self, req_id: str, future: Future) -> None:
        """Surface any exception from the disk-write thread and drop the
        completed future from the in-flight tracking dict."""
        self._req_futures.pop(req_id, None)
        exc = future.exception()
        if exc is not None:
            logger.error("Hidden-states write failed for req_id=%s: %r", req_id, exc)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
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
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Launches an async DtoH copy on a dedicated CUDA stream.  The
        actual disk write is deferred to wait_for_save() which submits
        it to a thread pool.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata), (
            "ExampleHiddenStatesConnector only supports CacheOnlyAttentionBackend"
        )

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata)

        os.makedirs(self._storage_path, exist_ok=True)

        copy_stream = self._get_copy_stream()

        # Ensure the copy stream sees all prior writes on the default stream.
        ready_event = torch.cuda.Event()
        ready_event.record()
        copy_stream.wait_event(ready_event)

        slot_mapping = attn_metadata.slot_mapping
        offset = 0
        for request in connector_metadata.requests:
            num_tokens = request.token_ids.shape[0]
            with torch.cuda.stream(copy_stream):
                req_slot_mapping = slot_mapping[offset : offset + num_tokens]
                offset += num_tokens

                # Move the CPU slot_mapping to GPU on the copy stream to avoid
                # implicit H2D that would sync the default stream.
                slot_mapping_gpu = req_slot_mapping.to(
                    device=kv_layer.device, non_blocking=True
                )
                hidden_states_gpu = extract_from_kv_cache(
                    kv_layer, slot_mapping_gpu, num_tokens
                )
                # Async DtoH copy into pinned host memory.
                pinned_hs = torch.empty_like(
                    hidden_states_gpu, device="cpu", pin_memory=True
                )
                pinned_hs.copy_(hidden_states_gpu, non_blocking=True)

            # Record completion of this copy on the copy stream.
            copy_done = torch.cuda.Event()
            copy_done.record(copy_stream)

            # token_ids is already on CPU (created in ReqMeta.make_meta).
            assert not request.token_ids.is_cuda, (
                "Expected token_ids on CPU, got CUDA tensor"
            )
            tensors = {
                "hidden_states": pinned_hs,
                "token_ids": request.token_ids.clone(),
            }
            self._pending_copies.append(
                (tensors, copy_done, request.filename, request.req_id)
            )

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
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            filename = os.path.join(self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
            )
            self._request_filenames[new_req.req_id] = filename
            self._active_requests[new_req.req_id] = new_req
            self._req_blocks[new_req.req_id] = list(new_req.block_ids[0])

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)
        _ = self._active_requests.pop(req_id, None)
        _ = self._req_blocks.pop(req_id, None)

        return True, {"hidden_states_path": req_filename}

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Poll DtoH-copy completion for requests that finished generating.

        The scheduler passes finished_req_ids to tell the worker which
        requests are done generating.  We accumulate these across calls
        and return a request as "finished sending" once its DtoH copy
        event is complete (or if it never had a pending copy).  The
        subsequent disk write may still be in flight; clients block on
        the per-file flock to wait for it.
        """
        self._accumulated_finished_req_ids.update(finished_req_ids)

        done_sending: set[str] = set()
        for req_id in list(self._accumulated_finished_req_ids):
            event = self._req_copy_events.get(req_id)
            if event is None or event.query():
                self._req_copy_events.pop(req_id, None)
                done_sending.add(req_id)
                self._accumulated_finished_req_ids.discard(req_id)

        return done_sending or None, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[0])

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
