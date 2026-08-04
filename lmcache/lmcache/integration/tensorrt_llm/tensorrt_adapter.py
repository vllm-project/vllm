# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM KV Cache Connector adapter for LMCache (in-process mode).

Implements ``LMCacheKvConnectorScheduler`` and ``LMCacheKvConnectorWorker`` —
the two classes TRT-LLM's ``kv_connector_config`` requires — backed by
an in-process LMCache engine singleton.

Lifecycle (per TRT-LLM connector ABC):
    * scheduler.get_num_new_matched_tokens → engine.lookup(tokens)
    * scheduler.build_connector_meta → LMCacheConnectorMetadata(loads, saves)
    * worker.register_kv_caches → builds engine via _get_or_create_engine,
      calls gpu_connector.register_kv_caches(kv_cache_tensor)
    * worker.start_load_kv → engine.retrieve(tokens, block_ids)
    * worker.wait_for_save → engine.store(tokens, block_ids)
"""

# Standard
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

# Third Party
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
import torch

# First Party
from lmcache import torch_dev
from lmcache.integration.tensorrt_llm.utils import (
    ENGINE_NAME,
    create_trtllm_metadata,
    lmcache_get_config,
)
from lmcache.logging import init_logger
from lmcache.utils import EngineType, mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.gpu_connector import CreateGPUConnector

logger = init_logger(__name__)


def _get_or_create_engine(
    llm_args: TorchLlmArgs,
    kv_cache_tensor: torch.Tensor,
) -> LMCacheEngine:
    """Return the LMCache engine singleton, creating it on first call.

    Called by the worker's ``register_kv_caches``. On subsequent calls
    (e.g. after a model reload) the existing engine is reused and the
    GPU connector is re-registered with the new KV tensor.
    """
    existing = LMCacheEngineBuilder.get(ENGINE_NAME)
    if existing is not None:
        gpu_connector = existing.gpu_connector
        if gpu_connector is not None:
            gpu_connector.register_kv_caches(kv_cache_tensor)  # type: ignore[attr-defined]
        logger.info("LMCache TRT-LLM: reusing existing engine")
        return existing

    config = lmcache_get_config()

    # Third Party
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(str(llm_args.model))
    head_dim = getattr(
        hf_config,
        "head_dim",
        hf_config.hidden_size // hf_config.num_attention_heads,
    )
    num_kv_heads = getattr(
        hf_config, "num_key_value_heads", hf_config.num_attention_heads
    )
    num_kv_heads = num_kv_heads // llm_args.tensor_parallel_size

    metadata = create_trtllm_metadata(
        llm_args,
        kv_cache_tensor,
        config,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    gpu_connector = CreateGPUConnector(config, metadata, EngineType.TRTLLM)
    gpu_connector.register_kv_caches(kv_cache_tensor)  # type: ignore[attr-defined]

    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        config,
        metadata,
        gpu_connector,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )
    engine.post_init()

    logger.info(
        "LMCache TRT-LLM: created engine (chunk_size=%d, tensor_shape=%s, dtype=%s)",
        config.chunk_size,
        list(kv_cache_tensor.shape),
        kv_cache_tensor.dtype,
    )
    return engine


def destroy_engine() -> None:
    """Destroy the engine singleton. Safe to call if not created."""
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        logger.info("LMCache TRT-LLM: engine destroyed")


@dataclass
class _BlockSpec:
    """Tokens and block IDs for a single load or store operation."""

    tokens: List[int]
    block_ids: List[int]


@dataclass
class LMCacheConnectorMetadata:
    """Metadata passed from the scheduler to the worker each forward step."""

    loads: dict = field(default_factory=dict)
    saves: dict = field(default_factory=dict)


class LMCacheKvConnectorScheduler(KvCacheConnectorScheduler):
    """Scheduler-side connector hook.

    Queries the LMCache engine for cached token counts and emits
    per-request load/save block specs for the worker to act on.
    """

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size: int = self._llm_args.kv_cache_config.tokens_per_block
        # request_id -> (all_tokens, num_matched) — set by
        # get_num_new_matched_tokens, consumed by build_connector_meta.
        self._pending: dict = {}
        # Engine is created by the worker in register_kv_caches, which
        # may run concurrently with scheduler init. Resolved lazily.
        self._engine: Optional[LMCacheEngine] = None

    def get_num_new_matched_tokens(
        self,
        request: LlmRequest,
        num_computed_tokens: int,
    ) -> Tuple[int, bool]:
        """Return how many additional tokens LMCache can provide beyond
        ``num_computed_tokens`` (which TRT-LLM matched via GPU block reuse).

        Args:
            request: The incoming request with its full token sequence.
            num_computed_tokens: Tokens already matched on device
                (block-aligned).

        Returns:
            ``(new_matched, is_async)``. ``is_async`` is always
            ``False`` in this adapter.
        """
        t0 = time.perf_counter()

        if not self._engine:
            self._engine = LMCacheEngineBuilder.get(ENGINE_NAME)
        if not self._engine:
            self._pending[request.request_id] = ([], 0)
            return 0, False

        # TRT-LLM should always pass block-aligned positions.
        if num_computed_tokens % self._block_size != 0:
            self._pending[request.request_id] = ([], 0)
            return 0, False

        all_tokens = list(request.get_tokens(0))

        max_block_aligned = (len(all_tokens) // self._block_size) * self._block_size
        if num_computed_tokens >= max_block_aligned:
            self._pending[request.request_id] = (all_tokens, 0)
            logger.debug(
                "LMCache TRT-LLM scheduler: req %d short-circuit "
                "(TRT matched %d of %d block-aligned tokens) %.3fms",
                request.request_id,
                num_computed_tokens,
                max_block_aligned,
                (time.perf_counter() - t0) * 1000,
            )
            return 0, False

        t1 = time.perf_counter()
        cached_tokens = self._engine.lookup(tokens=all_tokens)
        t2 = time.perf_counter()

        new_matched = max(0, cached_tokens - num_computed_tokens)
        new_matched = (new_matched // self._block_size) * self._block_size

        self._pending[request.request_id] = (all_tokens, new_matched)

        logger.debug(
            "LMCache TRT-LLM scheduler: req %d lookup=%.3fms total=%.3fms "
            "trt_matched=%d lmcache_cached=%d new_matched=%d",
            request.request_id,
            (t2 - t1) * 1000,
            (time.perf_counter() - t0) * 1000,
            num_computed_tokens,
            cached_tokens,
            new_matched,
        )
        return new_matched, False

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> LMCacheConnectorMetadata:
        """Build per-request load/save specs from the pending lookup
        results. The runtime binds the returned metadata to the worker
        via ``bind_connector_meta`` before the forward pass starts.
        """
        meta = LMCacheConnectorMetadata()

        for req in scheduler_output.new_requests:
            if req.request_id not in self._pending:
                continue

            all_tokens, num_matched = self._pending[req.request_id]
            block_ids: List[int] = list(req.new_block_ids)
            num_computed_blocks = req.computed_position // self._block_size

            if num_matched > 0:
                meta.loads[req.request_id] = _BlockSpec(
                    tokens=all_tokens, block_ids=block_ids
                )

            save_start = max(num_computed_blocks, num_matched // self._block_size)
            num_full_new_blocks = len(req.new_tokens) // self._block_size
            if (
                save_start < len(block_ids)
                and num_full_new_blocks > 0
                and save_start < num_computed_blocks + num_full_new_blocks
            ):
                meta.saves[req.request_id] = _BlockSpec(
                    tokens=all_tokens, block_ids=block_ids
                )

        self._pending.clear()
        return meta

    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        """Return whether async saving is in progress.

        Always ``False`` — saves are synchronous in this adapter.
        """
        return False

    def update_state_after_alloc(
        self, request: LlmRequest, block_ids: List[int]
    ) -> None:
        """No-op — block IDs are captured in ``build_connector_meta``
        from ``scheduler_output.new_requests``.
        """
        pass


class LMCacheKvConnectorWorker(KvCacheConnectorWorker):
    """Worker-side connector hook.

    Performs GPU↔CPU KV transfers via the LMCache engine.
    """

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size: int = self._llm_args.kv_cache_config.tokens_per_block
        # Cached after register_kv_caches to avoid per-call singleton lookup.
        self._engine: Optional[LMCacheEngine] = None
        self._load_stream: Optional[torch_dev.Stream] = None
        self._store_stream: Optional[torch_dev.Stream] = None

    @property
    def _meta(self) -> Optional[LMCacheConnectorMetadata]:
        """Typed accessor for ``self._metadata`` set by the base class."""
        return self._metadata  # type: ignore[return-value]

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor) -> None:
        """Register the KV cache tensor and create the LMCache engine.

        Called once by the runtime after KV cache allocation. Caches the
        engine and its load/store streams for fast access on every step.
        """
        self._engine = _get_or_create_engine(
            llm_args=self._llm_args,
            kv_cache_tensor=kv_cache_tensor,
        )
        gpu_conn = self._engine.gpu_connector
        if gpu_conn is not None:
            self._load_stream = gpu_conn.load_stream  # type: ignore[attr-defined]
            self._store_stream = gpu_conn.store_stream  # type: ignore[attr-defined]

    def start_load_kv(self, stream: torch_dev.Stream) -> None:
        """Load KV blocks from LMCache into the GPU paged cache.

        Retrieves all pending blocks on the load stream, then syncs the
        forward-pass stream against it. The cross-layer format loads
        every layer in a single kernel — no per-layer overlap to exploit.
        """
        meta = self._meta
        if meta is None or not meta.loads or self._engine is None:
            return

        t0 = time.perf_counter()
        for spec in meta.loads.values():
            if not spec.tokens or not spec.block_ids:
                continue
            self._engine.retrieve(tokens=spec.tokens, block_ids=spec.block_ids)

        if self._load_stream is not None:
            stream.wait_stream(self._load_stream)

        logger.debug(
            "LMCache TRT-LLM worker: start_load_kv retrieve=%.3fms num_loads=%d",
            (time.perf_counter() - t0) * 1000,
            len(meta.loads),
        )

    def wait_for_layer_load(self, layer_idx: int, stream: torch_dev.Stream) -> None:
        """No-op — cross-layer loads complete in :meth:`start_load_kv`."""
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch_dev.Stream) -> None:
        """No-op — saves are batched in :meth:`wait_for_save`."""
        pass

    def wait_for_save(self, stream: torch_dev.Stream) -> None:
        """Store newly computed KV blocks from GPU to LMCache's CPU cache.

        Waits on the forward-pass stream, runs ``engine.store`` for each
        request with new blocks, and synchronizes the store stream
        before returning.
        """
        meta = self._meta
        if meta is None or not meta.saves or self._engine is None:
            return

        t0 = time.perf_counter()
        if self._store_stream is not None:
            self._store_stream.wait_stream(stream)
        t1 = time.perf_counter()

        for spec in meta.saves.values():
            if not spec.tokens or not spec.block_ids:
                continue
            self._engine.store(tokens=spec.tokens, block_ids=spec.block_ids)
        t2 = time.perf_counter()

        if self._store_stream is not None:
            self._store_stream.synchronize()
        t3 = time.perf_counter()

        logger.debug(
            "LMCache TRT-LLM worker: wait_for_save stream_wait=%.3fms "
            "store=%.3fms sync=%.3fms num_saves=%d",
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            len(meta.saves),
        )

    def get_finished(
        self,
        finished_gen_req_ids: List[int],
        started_loading_req_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        """All ops are synchronous here — nothing is ever pending."""
        return [], []
