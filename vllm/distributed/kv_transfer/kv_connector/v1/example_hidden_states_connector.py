# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import torch
from packaging.version import Version
from safetensors.torch import load_file, save_file

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.kv_transfer import is_hs_store_cpu
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import token_ids_to_tensor
from vllm.v1.attention.backend import AttentionMetadata, CommonAttentionMetadata
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
    # Aborted request: drop the CPU pages without writing to disk.
    discard: bool = False


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[PendingSave] = field(default_factory=list)
    # req_id → filename for newly scheduled requests — the worker pre-creates
    # lock files for these so the lock exists before the client receives the
    # output path.
    new_req_filenames: dict[str, str] = field(default_factory=dict)


class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """Simple debug implementation of a HiddenStatesConnector.

    By default the hidden states are gathered from the device KV cache when a
    request finishes. With ``hs_store_device=cpu`` they are instead streamed
    to per-request CPU pages on TP rank 0 during decoding, keeping the
    extract_hidden_states layer out of the device KV pool. When the engine
    prefix cache is enabled, a token-hash index of completed blocks lets
    requests backfill the rows skipped by prefix-cache hits (text-only
    requests; multimodal / prompt-embeds requests are excluded and logged).
    Must be used in conjunction with the `extract_hidden_states` spec decoding method.
    """

    def get_finished_count(self) -> int:
        """Number of `finished_sending` notifications expected per request.

        Only TP rank 0 writes hidden states to disk, so each request yields a
        single notification. The default (`None`) makes `KVOutputAggregator`
        wait for every worker in the world size; since ranks 1..N-1 never submit
        a copy, the request would never be reported as finished and its blocks
        would be leaked forever (the scheduler then spins on
        `has_finished_requests()`).
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
        # With hs_store_device=cpu there is no device HiddenStateCacheSpec
        # group: the proposer streams hidden states to per-request CPU pages
        # via save_hs_batch instead.
        self._hs_store_cpu = is_hs_store_cpu(self._vllm_config)
        if self._hs_store_cpu:
            self._cache_kv_group_id = 0
            self._block_size = vllm_config.cache_config.block_size
        else:
            self._cache_kv_group_id = self._find_cache_kv_group_id(kv_cache_config)
            self._block_size = self._get_cache_block_size(
                vllm_config, kv_cache_config, self._cache_kv_group_id
            )
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers: list[str] = []  # set by self.register_kv_caches
        # CPU-pages state (worker-side, TP rank 0 only): per-request hidden
        # state buffers plus a reusable pinned staging buffer.
        self._cpu_pages: dict[str, torch.Tensor] = {}
        self._hs_staging: torch.Tensor | None = None
        self._logged_cpu_pages = False
        # CPU-pages prefix index (worker-side, TP rank 0 only): block-hash ->
        # cloned full-block hidden states, LRU-evicted by token budget. Lets a
        # request whose prefix hit the engine prefix cache copy the cached
        # rows into its page instead of leaving holes. The on/off flag is
        # resolved in register_kv_caches, not here: platforms may disable
        # enable_prefix_caching during model load (e.g. MLA attention
        # backends without prefix-cache support), and register_kv_caches runs
        # after model load, so it sees the value the scheduler runs with.
        self._hs_prefix_index_on = False
        self._hs_hash_blocks: OrderedDict = OrderedDict()
        self._hs_hash_tokens = 0
        self._hs_prefix_max_tokens = int(
            self._kv_transfer_config.get_from_extra_config(
                "hs_prefix_cache_max_tokens", 131072
            )
        )
        # Per-request chain hashes of full blocks (prefix-hash index keys).
        self._req_block_hashes: dict[str, list] = {}
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)
        if self._hs_store_cpu:
            logger.info(
                "Hidden states are stored on TP rank 0 CPU pages, "
                "not in the device KV cache"
            )

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

        # Only TP rank 0 writes hidden states to disk; other TP ranks no-op.
        # Resolved here so save_hs_batch is safe before register_kv_caches;
        # register_kv_caches refreshes it after distributed init.
        try:
            self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0
        except Exception:
            self._is_tp_rank_zero = True

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

        if self._hs_store_cpu:
            # No device hidden-states tensor to bind: the proposer streams
            # hidden states to CPU pages via save_hs_batch.
            self.cache_layers = []
            self._kv_cache = None
            # Resolve the CPU prefix-index switch here, not in __init__:
            # model load may flip cache_config.enable_prefix_caching off
            # afterwards (e.g. MLA attention backends without prefix-cache
            # support), and register_kv_caches runs after model load, so this
            # matches the value the scheduler actually runs with.
            self._hs_prefix_index_on = bool(
                self._vllm_config.cache_config.enable_prefix_caching
            ) and (self._hs_prefix_max_tokens > 0)
            if self._hs_prefix_index_on:
                logger.info(
                    "Hidden-states CPU prefix index enabled: "
                    "max_tokens=%s, block_size=%s",
                    self._hs_prefix_max_tokens,
                    self._block_size,
                )
            else:
                logger.info(
                    "Hidden-states CPU prefix index disabled "
                    "(engine prefix cache off or max_tokens=0)"
                )
            logger.info(
                "Hidden-states CPU pages: skipped device KV group binding (tp0=%s)",
                self._is_tp_rank_zero,
            )
            return

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

    def _hs_prefix_enabled(self) -> bool:
        # getattr guards keep this safe before register_kv_caches runs.
        return bool(
            getattr(self, "_hs_store_cpu", False)
            and getattr(self, "_hs_prefix_index_on", False)
            and getattr(self, "_is_tp_rank_zero", False)
        )

    def _req_hash_scope_ok(self, req_state: Any) -> bool:
        """Whether this request's hidden-state rows are fully determined by
        its token ids.

        The CPU prefix index is keyed on token-id block hashes. Requests with
        multimodal features or prompt embeddings compute hidden states that
        depend on more than the token ids (the engine's own block hashes
        include mm/embeds extra keys), so for them a token-hash lookup hit
        cannot prove row provenance. They must never publish into or
        backfill from the index: a text-hash collision across different
        images would silently backfill plausible-but-wrong rows.
        """
        return not (
            req_state.mm_features
            or req_state.prompt_embeds is not None
            or req_state.prompt_is_token_ids is not None
        )

    def _req_state(self, req_id: str, requests: Any) -> Any | None:
        if requests is None:
            return None
        return requests.get(req_id)

    def _warn_prefix_scope_gap(self, req_id: str, hit_tokens: int, reason: str) -> None:
        logger.warning(
            "Hidden-states CPU prefix index: req_id=%s skipped backfill of "
            "%s prefix-hit tokens (%s). Those rows are never streamed and "
            "stay unfilled in the saved hidden states; disable prefix "
            "caching if this request's data must be exact.",
            req_id,
            hit_tokens,
            reason,
        )

    def _block_extra_keys(self, req_state: Any) -> tuple[Any, ...] | None:
        # Mirror the engine's LoRA extra key; other engine keys (mm, salt,
        # embeds) are out of scope -- requests carrying them are excluded by
        # _req_hash_scope_ok.
        extra = []
        lora = getattr(req_state, "lora_request", None)
        if lora is not None:
            extra.append(getattr(lora, "lora_name", None) or str(lora))
        return tuple(extra) if extra else None

    def _ensure_req_hashes(self, req_id: str, requests: Any, n_blocks: int) -> list:
        """Chain-hash this request's full blocks, cached per request.

        Fail-safe: without request state nothing is hashed, so nothing can
        be published or backfilled (holes remain, but no wrong data).
        """
        hashes = self._req_block_hashes.setdefault(req_id, [])
        if n_blocks <= 0 or len(hashes) >= n_blocks:
            return hashes[: max(n_blocks, 0)]
        st = self._req_state(req_id, requests)
        if st is None:
            return hashes
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core import kv_cache_utils as _kvu
        from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

        ids = list(st.prompt_token_ids or [])
        ids.extend(st.output_token_ids or [])
        bs = int(self._block_size)
        algo = getattr(
            self._vllm_config.cache_config, "prefix_caching_hash_algo", "sha256"
        )
        hash_fn = get_hash_fn_by_name(algo)
        if getattr(_kvu, "NONE_HASH", None) is None:
            # hash_block_tokens reads the module-level NONE_HASH sentinel,
            # which the worker process may never have initialized.
            init_none_hash(hash_fn)
        extra = self._block_extra_keys(st)
        while len(hashes) < n_blocks:
            i = len(hashes)
            lo = i * bs
            hi = lo + bs
            if hi > len(ids):
                break  # only full blocks are hashed
            prev = hashes[-1] if hashes else None
            hashes.append(hash_block_tokens(hash_fn, prev, ids[lo:hi], extra))
        return hashes

    def _evict_hs_hash_blocks(self) -> None:
        # Called only when the index is on, so a positive token budget is
        # guaranteed; LRU by insertion order via move_to_end on hits.
        while self._hs_hash_tokens > self._hs_prefix_max_tokens and (
            self._hs_hash_blocks
        ):
            _, old = self._hs_hash_blocks.popitem(last=False)
            self._hs_hash_tokens -= int(old.shape[0])

    def _fill_prefix_from_hash(
        self,
        req_id: str,
        page: torch.Tensor,
        n_prefix: int,
        requests: Any,
    ) -> int:
        """Copy cached full blocks into a fresh page's prefix-hit region.

        ``n_prefix`` is the request's num_computed_tokens on its first
        streamed step, i.e. the rows the engine skipped via a prefix-cache
        hit. Returns the number of rows copied.
        """
        if not self._hs_prefix_enabled() or n_prefix <= 0:
            return 0
        st = self._req_state(req_id, requests)
        if st is None:
            self._warn_prefix_scope_gap(
                req_id, n_prefix, "request state unavailable on this rank"
            )
            return 0
        if not self._req_hash_scope_ok(st):
            self._warn_prefix_scope_gap(
                req_id,
                n_prefix,
                "multimodal or prompt-embeds inputs are outside the token-hash scope",
            )
            return 0
        bs = int(self._block_size)
        n_blocks = n_prefix // bs
        if n_blocks <= 0:
            return 0
        hashes = self._ensure_req_hashes(req_id, requests, n_blocks)
        copied = 0
        for i, h in enumerate(hashes[:n_blocks]):
            blk = self._hs_hash_blocks.get(h)
            if blk is None:
                break
            lo = i * bs
            hi = lo + bs
            if page.shape[0] < hi or blk.shape[0] != bs:
                break
            page[lo:hi].copy_(blk)
            self._hs_hash_blocks.move_to_end(h)
            copied += 1
        if copied < n_blocks:
            logger.warning(
                "Hidden-states CPU prefix index: req_id=%s backfilled %s/%s "
                "prefix blocks; the rest are missing (evicted or not yet "
                "published), so rows from token %s on stay unfilled",
                req_id,
                copied,
                n_blocks,
                copied * bs,
            )
        elif copied:
            logger.info(
                "Hidden-states CPU prefix index: backfilled %s/%s prefix "
                "blocks req_id=%s",
                copied,
                n_blocks,
                req_id,
            )
        return copied * bs

    def _publish_full_blocks(
        self,
        req_id: str,
        page: torch.Tensor,
        start: int,
        need: int,
        requests: Any,
    ) -> None:
        """Publish this step's newly completed full blocks into the index."""
        if not self._hs_prefix_enabled() or need <= start:
            return
        st = self._req_state(req_id, requests)
        if st is None or not self._req_hash_scope_ok(st):
            # Rows not token-determined: keep them out of the index entirely
            # instead of polluting it for future lookups.
            return
        bs = int(self._block_size)
        first = start // bs
        last = need // bs
        if last <= first:
            return
        hashes = self._ensure_req_hashes(req_id, requests, last)
        for i in range(first, min(last, len(hashes))):
            h = hashes[i]
            if h in self._hs_hash_blocks:
                self._hs_hash_blocks.move_to_end(h)
                continue
            lo = i * bs
            hi = lo + bs
            if page.shape[0] < hi:
                break
            if page.is_floating_point() and bool(page[lo:hi].isnan().any()):
                # Partially-filled block: prefix-hit rows the index could
                # not backfill keep the NaN sentinel. Publishing it would
                # spread those holes to every future request that hits
                # the same hash.
                continue
            self._hs_hash_blocks[h] = page[lo:hi].contiguous().clone()
            self._hs_hash_tokens += bs
        self._evict_hs_hash_blocks()

    def save_hs_batch(
        self,
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        req_ids: list[str] | None,
        requests: dict[str, Any] | None = None,
    ) -> None:
        """Copy this step's hidden states into per-request CPU pages (TP0).

        ``hidden_states`` is [num_tokens, num_layers, hidden]. The logical
        write offset is ``num_computed_tokens`` per request so chunked prefill
        concatenates. Rows skipped by an engine prefix-cache hit are backfilled
        from the CPU prefix index when possible (see _fill_prefix_from_hash);
        rows that cannot be proven (multimodal / prompt-embeds requests) or
        were evicted stay NaN-filled (a sentinel the writer thread audits, so
        holes are detectable downstream instead of looking like valid data).
        """
        if not self._hs_store_cpu:
            return
        if not self._is_tp_rank_zero:
            return
        if req_ids is None or common_attn_metadata is None:
            return
        if hidden_states.numel() == 0:
            return

        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0:
            return

        qsl = common_attn_metadata.query_start_loc_cpu
        if getattr(qsl, "is_cuda", False):
            qsl = qsl.cpu()
        qsl_list = qsl[: num_reqs + 1].tolist()

        # num_computed_tokens per request is the page's logical write
        # offset. Prefer the scheduler-fed host value
        # (req_state.num_computed_tokens, updated right before forward):
        # it equals seq_lens - query_len without pulling the device
        # seq_lens tensor to host on every step. Fall back to deriving
        # it from seq_lens only when request state is unavailable.
        nct_list = self._host_num_computed_tokens(req_ids, requests, num_reqs)
        if nct_list is None:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            if getattr(seq_lens, "is_cuda", False):
                seq_lens = seq_lens.cpu()
            seq_list = seq_lens.tolist()
            nct_list = [
                int(seq_list[i]) - int(qsl_list[i + 1] - qsl_list[i])
                for i in range(num_reqs)
            ]
        self._save_hs_to_pages(
            hidden_states,
            int(common_attn_metadata.num_actual_tokens),
            qsl_list,
            nct_list,
            req_ids,
            requests,
        )

    def save_hs_batch_tokens(
        self,
        hidden_states: torch.Tensor,
        req_ids: list[str] | None,
        query_start_loc_cpu: Any,
        seq_lens: Any,
        num_actual_tokens: int,
        requests: dict[str, Any] | None = None,
    ) -> None:
        """``save_hs_batch`` variant for runners without CommonAttentionMetadata.

        The V2 GPU model runner batches requests with its own ``InputBatch``
        (per-batch ``req_ids`` / ``query_start_loc`` / ``seq_lens``), so the
        per-request bookkeeping is passed explicitly instead of through
        attention metadata. Semantics match ``save_hs_batch`` exactly: the
        write offset prefers host-side ``num_computed_tokens`` from the
        request states and falls back to deriving it from ``seq_lens``.
        """
        if not self._hs_store_cpu:
            return
        if not self._is_tp_rank_zero:
            return
        if req_ids is None or hidden_states.numel() == 0:
            return

        num_reqs = len(req_ids)
        if num_reqs <= 0:
            return

        qsl = query_start_loc_cpu
        if getattr(qsl, "is_cuda", False):
            qsl = qsl.cpu()
        qsl_list = qsl[: num_reqs + 1].tolist()

        nct_list = self._host_num_computed_tokens(req_ids, requests, num_reqs)
        if nct_list is None:
            seq = seq_lens
            if getattr(seq, "is_cuda", False):
                seq = seq.cpu()
            seq_list = seq[:num_reqs].tolist()
            nct_list = [
                int(seq_list[i]) - int(qsl_list[i + 1] - qsl_list[i])
                for i in range(num_reqs)
            ]
        self._save_hs_to_pages(
            hidden_states,
            int(num_actual_tokens),
            qsl_list,
            nct_list,
            req_ids,
            requests,
        )

    @staticmethod
    def _host_num_computed_tokens(
        req_ids: list[str],
        requests: dict[str, Any] | None,
        num_reqs: int,
    ) -> list[int] | None:
        """Per-request num_computed_tokens from the host request states.

        Returns None when any request lacks state (or requests is None),
        so the caller falls back to deriving the offsets from the device
        seq_lens.
        """
        if requests is None:
            return None
        nct_list: list[int] = []
        for i in range(num_reqs):
            st = requests.get(req_ids[i]) if i < len(req_ids) else None
            nct = getattr(st, "num_computed_tokens", None)
            if nct is None:
                return None
            nct_list.append(int(nct))
        return nct_list

    def _save_hs_to_pages(
        self,
        hidden_states: torch.Tensor,
        num_actual_tokens: int,
        qsl_list: list[int],
        nct_list: list[int],
        req_ids: list[str],
        requests: dict[str, Any] | None,
    ) -> None:
        """Splice this step's rows into per-request CPU pages.

        ``nct_list[i]`` is request i's num_computed_tokens (its logical
        write offset), so chunked prefill concatenates. Rows skipped by
        an engine prefix-cache hit are backfilled from the CPU prefix
        index when possible (see _fill_prefix_from_hash); rows that cannot
        be proven (multimodal / prompt-embeds requests) or were evicted
        stay NaN-filled (a sentinel the writer thread audits, so holes are
        detectable downstream instead of looking like valid data).
        """
        num_reqs = len(req_ids)

        num_actual = min(int(num_actual_tokens), int(hidden_states.shape[0]))
        hs = hidden_states[:num_actual]
        if hs.numel() == 0:
            return

        _, num_layers, hidden_size = hs.shape
        dtype = hs.dtype
        chunk = int(
            self._kv_transfer_config.get_from_extra_config(
                "hs_gather_chunk_tokens", 4096
            )
        )
        if chunk <= 0:
            chunk = num_actual

        if not self._logged_cpu_pages:
            logger.info(
                "Streaming hidden states DtoH to TP0 CPU pages "
                "(chunk=%s tokens, layers=%s, hidden=%s, dtype=%s)",
                chunk,
                num_layers,
                hidden_size,
                dtype,
            )
            self._logged_cpu_pages = True

        if hs.is_cuda:
            copy_stream = self._get_copy_stream()
            ready_event = torch.cuda.Event()
            ready_event.record()
            copy_stream.wait_event(ready_event)

            staging = self._hs_staging
            if (
                staging is None
                or staging.shape[0] < num_actual
                or staging.dtype != dtype
                or staging.shape[1] != num_layers
                or staging.shape[2] != hidden_size
            ):
                try:
                    staging = torch.empty(
                        (num_actual, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                except Exception:
                    staging = torch.empty(
                        (num_actual, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                    )
                self._hs_staging = staging

            with torch.cuda.stream(copy_stream):
                for start in range(0, num_actual, chunk):
                    end = min(start + chunk, num_actual)
                    staging[start:end].copy_(hs[start:end], non_blocking=True)
                    chunk_done = torch.cuda.Event()
                    chunk_done.record(copy_stream)
                    chunk_done.synchronize()
        else:
            # Host-side hidden states (CPU-only platforms, unit tests): no
            # copy stream or pinned staging needed, stream from the tensor.
            staging = hs

        for i in range(num_reqs):
            q0 = int(qsl_list[i])
            q1 = min(int(qsl_list[i + 1]), num_actual)
            n = q1 - q0
            if n <= 0:
                continue
            req_id = req_ids[i]
            start = max(int(nct_list[i]), 0)
            need = start + n
            cap = need
            if requests is not None and req_id in requests:
                try:
                    cap = max(cap, int(requests[req_id].num_prompt_tokens), need)
                except Exception:
                    cap = need
            page = self._cpu_pages.get(req_id)
            page_is_new = page is None
            if page is None or page.shape[0] < need:
                try:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                except Exception:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                    )
                # Sentinel-fill: rows the engine skipped (prefix-cache hits
                # the CPU index cannot prove) or the growth gap of a dropped
                # streaming step would otherwise keep uninitialized memory
                # that looks like plausible data in the saved file. NaN is
                # detectable downstream (isfinite) and audited on save.
                if new_page.is_floating_point():
                    new_page.fill_(float("nan"))
                if page is not None:
                    new_page[: page.shape[0]].copy_(page)
                self._cpu_pages[req_id] = new_page
                page = new_page
            if page_is_new and start > 0:
                # First streamed step with rows the engine skipped via a
                # prefix-cache hit: copy them from the CPU prefix index.
                self._fill_prefix_from_hash(req_id, page, start, requests)
            page[start:need].copy_(staging[q0:q1])
            self._publish_full_blocks(req_id, page, start, need, requests)

    @staticmethod
    def _write_tensors(
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
        req_id: str | None = None,
        audit_nan: bool = False,
    ) -> None:
        """Thread worker: wait for async DtoH copy, write to disk, release lock.

        `lock_fd` is an open file descriptor on the companion `.lock`
        file with `LOCK_EX` already held.  Closing it releases the lock,
        which unblocks any client sleeping on `LOCK_SH`.

        With `audit_nan` (CPU-pages path), count rows still holding the
        NaN sentinel before writing: they were never filled (prefix-cache
        rows the index could not prove, or a dropped streaming step) and the
        log makes the hole auditable instead of looking like valid data.
        """
        try:
            event.synchronize()
            if audit_nan:
                hs = tensors["hidden_states"]
                if hs.is_floating_point():
                    hole_rows = int(
                        torch.isnan(hs).any(dim=tuple(range(1, hs.dim()))).sum()
                    )
                    if hole_rows:
                        logger.warning(
                            "req_id=%s: %s/%s hidden-state rows were never "
                            "filled (NaN sentinel); prefix-cache rows missed "
                            "by the CPU index or a dropped streaming step - "
                            "treat those rows as invalid",
                            req_id,
                            hole_rows,
                            hs.shape[0],
                        )
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
            self._cpu_pages.pop(pending.req_id, None)
            self._req_block_hashes.pop(pending.req_id, None)
            return

        if self._hs_store_cpu:
            # CPU-pages path: the hidden states were streamed by save_hs_batch;
            # finishing is a plain concat + save_file, no device KV gather.
            if pending.discard:
                self._cpu_pages.pop(pending.req_id, None)
                self._req_block_hashes.pop(pending.req_id, None)
                return
            finalized = self._finalize_cpu_page(pending.req_id, pending.token_ids)
            if finalized is None:
                return
            hidden_states_cpu, token_ids = finalized
            copy_done = torch.cuda.Event()
            copy_done.record()
            tensors = {
                "hidden_states": hidden_states_cpu,
                "token_ids": token_ids,
            }
            os.makedirs(os.path.dirname(pending.filename) or ".", exist_ok=True)
            lock_fd = self._lock_fds.pop(pending.req_id, None)
            if lock_fd is None and self.use_lock:
                lock_path = pending.filename + ".lock"
                lock_fd = os.open(
                    lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644
                )
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            future = self._executor.submit(
                self._write_tensors,
                tensors,
                copy_done,
                pending.filename,
                lock_fd,
                pending.req_id,
                self._hs_store_cpu,
            )
            self._req_copy_events[pending.req_id] = copy_done
            self._req_futures[pending.req_id] = future
            future.add_done_callback(partial(self._on_write_done, pending.req_id))
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
            # stream. Deliberate bulk transfer in this reference connector.
            with gpu_sync_allowed():
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

    def _finalize_cpu_page(
        self, req_id: str, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Pop a finished request's CPU page and align it with its token ids.

        Returns ``(hidden_states, token_ids)`` trimmed to the same length, or
        ``None`` if nothing was ever streamed for the request. Unfilled rows
        keep their NaN sentinel (audited by the writer thread) instead of
        leaking stale memory into the saved file.
        """
        page = self._cpu_pages.pop(req_id, None)
        self._req_block_hashes.pop(req_id, None)
        if page is None:
            logger.error("No CPU hidden states for req_id=%s", req_id)
            return None
        num_tokens = int(token_ids.shape[0])
        if page.shape[0] < num_tokens:
            # A streaming step was dropped (the proposers raise on that
            # condition, but stay defensive): keep both tensors the same
            # length so row i still corresponds to token i. The saved file
            # is then a valid prefix of the sequence, and the loss is loud.
            logger.error(
                "CPU hidden-states page too short for req_id=%s: %s < %s; "
                "saving the first %s tokens only",
                req_id,
                page.shape[0],
                num_tokens,
                page.shape[0],
            )
            num_tokens = int(page.shape[0])
            token_ids = token_ids[:num_tokens]
        # The page is popped (sole owner), so hand the view to the writer
        # thread without another full-size clone.
        return page[:num_tokens].contiguous(), token_ids

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
        """Called exactly once when a request has finished, before its blocks are
        freed.

        Returns True to delay block freeing until get_finished extracts
        the hidden states from the KV cache.
        """
        req_id = request.request_id
        filename = self._request_filenames.pop(req_id, None)
        if filename is None:
            # Request was aborted before it was ever scheduled, so
            # build_connector_meta never recorded a filename for it. There are
            # no hidden states to save and no blocks worth delaying.
            self._pending_saves.pop(req_id, None)
            return False, None
        kv_params = request.kv_transfer_params or {}
        if kv_params.get("include_output_tokens", False):
            # Exclude the final token — it was the model's output, never an
            # input to a forward pass, so its hidden state is not in the cache.
            token_ids = token_ids_to_tensor(request._all_token_ids)[:-1]
        elif request.prompt_token_ids is not None:
            token_ids = token_ids_to_tensor(request.prompt_token_ids)
        else:
            logger.warning(
                "Request %s has no prompt_token_ids (prompt_embeds only). "
                "Saved token_ids will be empty.",
                req_id,
            )
            token_ids = torch.tensor([], dtype=torch.long)
        discard = request.status == RequestStatus.FINISHED_ABORTED
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
            discard=discard,
        )
        if self._hs_store_cpu:
            # With CPU pages the device blocks never hold hidden states, so
            # they can be freed immediately instead of waiting for the KV
            # gather. Aborted requests drop their CPU pages without a disk
            # write, hence no hidden-states path is reported.
            delay = False
            params = None if discard else {"hidden_states_path": filename}
        else:
            # Device mode keeps the original behavior: delay block freeing
            # until the KV gather completes, and report the hidden-states
            # path for every finished request, aborted or not.
            delay = True
            params = {"hidden_states_path": filename}
        return delay, params

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

        if self._hs_store_cpu:
            # request_finished returns delay_free=False in CPU-pages mode, so
            # the scheduler already freed the request and dropped its id before
            # this poll runs; reporting it as finished_sending would trip the
            # scheduler's "req_id in self.requests" assertion. There are no
            # device blocks to release, so stay silent on the KV-xfer channel.
            return None, None
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
        if self._hs_store_cpu:
            # No hidden-states KV group / block ids in CPU-pages mode.
            return self.request_finished(request, [])
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
