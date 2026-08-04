# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM KV Cache Connector adapter for LMCache (multi-process mode).

Implements ``LMCacheMPKvConnectorScheduler`` and
``LMCacheMPKvConnectorWorker`` — the two classes TRT-LLM's
``kv_connector_config`` requires — backed by a standalone LMCache server
reached over ZMQ. Provides process isolation and shared caching across
multiple TRT-LLM instances on the same node.

The KV pool tensor is shared with the server via :class:`RawCudaIPCWrapper`
because TRT-LLM's pool is allocated outside PyTorch's caching allocator
(``at::for_blob`` over ``cudaMalloc``), which makes
``UntypedStorage._share_cuda_()`` raise. The wrapper bypasses that path.
"""

# Standard
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
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
import zmq

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType, check_interprocess_event_support
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.platform.cuda.ipc_wrapper import RawCudaIPCWrapper

logger = init_logger(__name__)

DEFAULT_SERVER_URL = "ipc:///tmp/lmcache.sock"
DEFAULT_MQ_TIMEOUT: float = 300.0


def _get_server_url(llm_args: "TorchLlmArgs") -> str:
    """Resolve the server URL: connector-config field > env var > default."""
    cfg = llm_args.kv_connector_config
    if cfg is not None and cfg.server_url is not None:
        return cfg.server_url
    return os.environ.get("LMCACHE_SERVER_URL", DEFAULT_SERVER_URL)


def _send_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list,
) -> MessagingFuture:
    return mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )


@dataclass
class _BlockSpec:
    tokens: List[int]
    block_ids: List[int]


@dataclass
class LMCacheMPConnectorMetadata:
    loads: dict = field(default_factory=dict)
    saves: dict = field(default_factory=dict)


class LMCacheMPKvConnectorScheduler(KvCacheConnectorScheduler):
    """TRT-LLM scheduler that routes lookup requests to an LMCache MP server."""

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size: int = self._llm_args.kv_cache_config.tokens_per_block
        # request_id -> (all_tokens, num_matched).
        self._pending: dict = {}

        self._zmq_context = zmq.Context()
        self._mq_client = MessageQueueClient(
            _get_server_url(self._llm_args), self._zmq_context
        )
        self._mq_timeout = float(
            os.environ.get("LMCACHE_MQ_TIMEOUT", DEFAULT_MQ_TIMEOUT)
        )

        future = _send_request(self._mq_client, RequestType.GET_CHUNK_SIZE, [])
        self._chunk_size = future.result(timeout=self._mq_timeout)
        logger.info(
            "LMCache MP scheduler: connected to server at %s (chunk_size=%d)",
            _get_server_url(self._llm_args),
            self._chunk_size,
        )

        # Third Party
        import tensorrt_llm

        self._rank = tensorrt_llm.mpi_rank()
        tp_size = llm_args.tensor_parallel_size
        pp_size = llm_args.pipeline_parallel_size
        self._world_size = tp_size * pp_size
        self._model_name = str(getattr(llm_args, "model", "unknown_model"))

    def _create_key(
        self,
        token_ids: List[int],
        start: int,
        end: int,
        request_id: int,
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=str(request_id),
        )

    def get_num_new_matched_tokens(
        self,
        request: LlmRequest,
        num_computed_tokens: int,
    ) -> Tuple[int, bool]:
        """Return how many additional tokens the LMCache server can provide.

        Submits a ``LOOKUP`` to the server and queries
        ``QUERY_PREFETCH_STATUS`` by ``request_id`` to read the result.
        ``LOOKUP`` returns ``None`` on the server protocol — the prefetch
        is tracked server-side keyed by ``request_id``.
        """
        t0 = time.perf_counter()

        if num_computed_tokens % self._block_size != 0:
            self._pending[request.request_id] = ([], 0)
            return 0, False

        all_tokens = list(request.get_tokens(0))

        max_block_aligned = (len(all_tokens) // self._block_size) * self._block_size
        if num_computed_tokens >= max_block_aligned:
            self._pending[request.request_id] = (all_tokens, 0)
            return 0, False

        aligned_end = (len(all_tokens) // self._chunk_size) * self._chunk_size
        key = self._create_key(
            all_tokens, start=0, end=aligned_end, request_id=request.request_id
        ).no_worker_id_version()

        t1 = time.perf_counter()

        try:
            _send_request(self._mq_client, RequestType.LOOKUP, [key, 1]).result(
                timeout=self._mq_timeout
            )
            result = _send_request(
                self._mq_client,
                RequestType.QUERY_PREFETCH_STATUS,
                [str(request.request_id)],
            ).result(timeout=self._mq_timeout)
            cached_tokens = result * self._chunk_size if result is not None else 0
        except Exception as e:
            logger.warning("LMCache MP scheduler: lookup failed: %s", e)
            self._pending[request.request_id] = (all_tokens, 0)
            return 0, False

        t2 = time.perf_counter()

        new_matched = max(0, cached_tokens - num_computed_tokens)
        new_matched = (new_matched // self._block_size) * self._block_size

        # ``LOOKUP`` acquires read locks on chunks in [0, cached_tokens).
        # TRT-LLM already has [0, num_computed_tokens), so those chunks
        # will never be retrieved — release their locks (chunk-aligned)
        # to avoid holding them until TTL expiry.
        overlap_end = min(cached_tokens, num_computed_tokens)
        overlap_end = (overlap_end // self._chunk_size) * self._chunk_size
        if overlap_end > 0:
            free_key = self._create_key(
                all_tokens,
                start=0,
                end=overlap_end,
                request_id=request.request_id,
            ).no_worker_id_version()
            try:
                _send_request(
                    self._mq_client,
                    RequestType.FREE_LOOKUP_LOCKS,
                    [free_key, 1],
                )
            except Exception as e:
                logger.warning("LMCache MP scheduler: free_lookup_locks failed: %s", e)

        self._pending[request.request_id] = (all_tokens, new_matched)

        logger.debug(
            "LMCache MP scheduler: req %d lookup=%.3fms total=%.3fms "
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
    ) -> LMCacheMPConnectorMetadata:
        """Build per-request load/save specs from pending lookup results."""
        meta = LMCacheMPConnectorMetadata()

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
        """Notify the server so it can clean up per-request state.

        Saves are synchronous in this adapter, so we never need to defer
        deallocation — return ``False``. We still call ``END_SESSION`` to
        release the server-side token-hash/session state for the request.
        """
        try:
            _send_request(
                self._mq_client,
                RequestType.END_SESSION,
                [str(request.request_id)],
            )
        except Exception as e:
            logger.warning("LMCache MP scheduler: end_session failed: %s", e)
        return False

    def update_state_after_alloc(
        self, request: LlmRequest, block_ids: List[int]
    ) -> None:
        """No-op — block IDs are captured in :meth:`build_connector_meta`."""
        pass


class LMCacheMPKvConnectorWorker(KvCacheConnectorWorker):
    """TRT-LLM worker that routes store/retrieve to an LMCache MP server."""

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size: int = self._llm_args.kv_cache_config.tokens_per_block

        self._zmq_context = zmq.Context()
        self._mq_client = MessageQueueClient(
            _get_server_url(self._llm_args), self._zmq_context
        )
        self._mq_timeout = float(
            os.environ.get("LMCACHE_MQ_TIMEOUT", DEFAULT_MQ_TIMEOUT)
        )

        self._instance_id = os.getpid()
        self._registered = False

        # Third Party
        import tensorrt_llm

        self._rank = tensorrt_llm.mpi_rank()
        tp_size = llm_args.tensor_parallel_size
        pp_size = llm_args.pipeline_parallel_size
        self._world_size = tp_size * pp_size
        self._model_name = str(getattr(llm_args, "model", "unknown_model"))

        future = _send_request(self._mq_client, RequestType.GET_CHUNK_SIZE, [])
        self._chunk_size = future.result(timeout=self._mq_timeout)

    def _create_key(
        self,
        token_ids: List[int],
        request_id: int,
    ) -> IPCCacheServerKey:
        aligned_end = (len(token_ids) // self._chunk_size) * self._chunk_size
        return IPCCacheServerKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._rank,
            token_ids=tuple(token_ids),
            start=0,
            end=aligned_end,
            request_id=str(request_id),
        )

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor) -> None:
        """Register the KV pool with the LMCache server via raw CUDA IPC.

        TRT-LLM provides a 4-D pool tensor
        ``[NB, NL, 2, NH * BS * HS]``. The server reshapes it to 6-D
        ``[NB, NL, 2, NH, BS, HS]`` from the ``layout_hints`` so format
        detection lands on ``NB_NL_TWO_NH_BS_HS``.
        """
        if self._registered:
            logger.info("LMCache MP worker: KV caches already registered")
            return

        # Third Party
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(self._model_name)
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        num_kv_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        )
        tp_size = self._llm_args.tensor_parallel_size
        num_kv_heads = num_kv_heads // tp_size

        _, _, _, block_size_flat = kv_cache_tensor.shape
        tokens_per_block = block_size_flat // (num_kv_heads * head_dim)

        wrapped = [RawCudaIPCWrapper(kv_cache_tensor)]

        layout_hints = {
            "kv_layout": "HND",
            "num_kv_heads": num_kv_heads,
            "tokens_per_block": tokens_per_block,
            "head_dim": head_dim,
        }

        future = _send_request(
            self._mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                self._instance_id,
                wrapped,
                self._model_name,
                self._world_size,
                EngineType.TRTLLM,
                layout_hints,
                [],
            ],
        )
        try:
            future.result(timeout=self._mq_timeout)
            self._registered = True
            logger.info(
                "LMCache MP worker: registered KV caches "
                "(tensor_shape=%s, NH=%d, BS=%d, HS=%d)",
                list(kv_cache_tensor.shape),
                num_kv_heads,
                tokens_per_block,
                head_dim,
            )
        except TimeoutError:
            logger.error(
                "LMCache MP worker: KV cache registration timed out after %ss",
                self._mq_timeout,
            )

    def start_load_kv(self, stream: torch_dev.Stream) -> None:
        """Send ``RETRIEVE`` requests for each pending load."""
        meta: Optional[LMCacheMPConnectorMetadata] = self._metadata
        if meta is None or not meta.loads:
            return

        t0 = time.perf_counter()
        # Not all backends support interprocess Events (CUDA IPC specific)
        check_interprocess_event_support()
        event = torch_dev.Event(interprocess=True)
        event.record(stream)

        for req_id, spec in meta.loads.items():
            if not spec.tokens or not spec.block_ids:
                continue

            key = self._create_key(spec.tokens, req_id)
            try:
                _send_request(
                    self._mq_client,
                    RequestType.RETRIEVE,
                    [
                        key,
                        self._instance_id,
                        [spec.block_ids],
                        event.ipc_handle(),
                        0,  # skip_first_n_tokens
                    ],
                ).result(timeout=self._mq_timeout)
            except Exception as e:
                logger.warning(
                    "LMCache MP worker: retrieve failed for req %d: %s",
                    req_id,
                    e,
                )

        logger.debug(
            "LMCache MP worker: start_load_kv retrieve=%.3fms num_loads=%d",
            (time.perf_counter() - t0) * 1000,
            len(meta.loads),
        )

    def wait_for_layer_load(self, layer_idx: int, stream: torch_dev.Stream) -> None:
        """No-op — server synchronizes via CUDA IPC events."""
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch_dev.Stream) -> None:
        """No-op — saves are batched in :meth:`wait_for_save`."""
        pass

    def wait_for_save(self, stream: torch_dev.Stream) -> None:
        """Send ``STORE`` requests for each pending save."""
        meta: Optional[LMCacheMPConnectorMetadata] = self._metadata
        if meta is None or not meta.saves:
            return

        t0 = time.perf_counter()
        # Not all backends support interprocess Events (CUDA IPC specific)
        check_interprocess_event_support()
        event = torch_dev.Event(interprocess=True)
        event.record(stream)

        for req_id, spec in meta.saves.items():
            if not spec.tokens or not spec.block_ids:
                continue

            key = self._create_key(spec.tokens, req_id)
            try:
                _send_request(
                    self._mq_client,
                    RequestType.STORE,
                    [
                        key,
                        self._instance_id,
                        [spec.block_ids],
                        event.ipc_handle(),
                    ],
                ).result(timeout=self._mq_timeout)
            except Exception as e:
                logger.warning(
                    "LMCache MP worker: store failed for req %d: %s",
                    req_id,
                    e,
                )

        logger.debug(
            "LMCache MP worker: wait_for_save store=%.3fms num_saves=%d",
            (time.perf_counter() - t0) * 1000,
            len(meta.saves),
        )

    def get_finished(
        self,
        finished_gen_req_ids: List[int],
        started_loading_req_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        """All operations are synchronous — nothing is ever pending."""
        return [], []
