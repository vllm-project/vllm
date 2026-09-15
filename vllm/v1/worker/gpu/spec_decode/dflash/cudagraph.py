# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Mapping

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import maybe_prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class BoundedContextCudaGraph:
    """Small fixed-shape context-KV graph family with eager fallback."""

    is_speculator_graph_manager = True

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        hidden_size: int,
        max_num_tokens: int,
    ) -> None:
        self.max_num_tokens = max_num_tokens
        self.context_states = torch.zeros(
            max_num_tokens, hidden_size, dtype=dtype, device=device
        )
        self.device = self.context_states.device
        self.context_positions = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        )
        self.context_slot_mapping = torch.full(
            (max_num_tokens,),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=device,
        )
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Different token-count graphs replay in arbitrary order. They cannot
        # share a private pool: PyTorch only permits shared-pool graphs when
        # replay order always matches capture order.
        self.graph_pools: dict[int, tuple[int, int]] = {}
        self.capture_stream: torch.cuda.Stream | None = None
        self.replay_count = 0
        self.fallback_count = 0

    def clear(self) -> None:
        """Release graphs captured against a temporary or superseded KV cache."""
        self.graphs.clear()
        self.graph_pools.clear()
        self.capture_stream = None
        self.replay_count = 0
        self.fallback_count = 0

    @staticmethod
    def _shared_slot_mapping(
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> torch.Tensor | None:
        """Canonicalize one shared slot row without synchronizing the GPU."""
        if isinstance(context_slot_mapping, torch.Tensor):
            return context_slot_mapping
        if not context_slot_mapping or any(
            slot_mapping is None for slot_mapping in context_slot_mapping
        ):
            return None
        slot_mappings = [
            slot_mapping
            for slot_mapping in context_slot_mapping
            if slot_mapping is not None
        ]
        first = slot_mappings[0]
        if all(
            slot_mapping.data_ptr() == first.data_ptr()
            and slot_mapping.shape == first.shape
            and slot_mapping.stride() == first.stride()
            for slot_mapping in slot_mappings[1:]
        ):
            return first
        return None

    @torch.inference_mode()
    def capture(
        self,
        forward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    ) -> None:
        """Warm up and capture every context shape up to the configured bound."""
        self.clear()
        self.context_slot_mapping.fill_(PAD_SLOT_ID)
        for num_tokens in range(1, self.max_num_tokens + 1):
            logger.info("Warming DSpark context graph shape %d", num_tokens)
            forward_fn(
                self.context_states[:num_tokens],
                self.context_positions[:num_tokens],
                self.context_slot_mapping[:num_tokens],
            )
            torch.accelerator.synchronize()
        # This projection/cache sequence has no collectives. Using vLLM's
        # distributed graph_capture context here would open a second AITER
        # collective-capture lifecycle after the query graph and overwrite its
        # registered graph-address state. A dedicated plain HIP stream avoids
        # touching that global collective registry.
        source_stream = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream = capture_stream
        capture_stream.wait_stream(source_stream)
        with torch.cuda.stream(capture_stream):
            for num_tokens in range(1, self.max_num_tokens + 1):
                logger.info("Capturing DSpark context graph shape %d", num_tokens)
                graph = torch.cuda.CUDAGraph()
                graph_pool = current_platform.graph_pool_handle()
                with torch.cuda.graph(
                    graph,
                    pool=graph_pool,
                    stream=current_stream(),
                ):
                    forward_fn(
                        self.context_states[:num_tokens],
                        self.context_positions[:num_tokens],
                        self.context_slot_mapping[:num_tokens],
                    )
                self.graphs[num_tokens] = graph
                self.graph_pools[num_tokens] = graph_pool
                torch.accelerator.synchronize()
        source_stream.wait_stream(capture_stream)

    @torch.inference_mode()
    def replay(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
        *,
        eligible: bool,
    ) -> bool:
        """Replay a captured shape when inputs satisfy the bounded contract."""
        num_tokens = context_states.shape[0]
        shared_slot_mapping = self._shared_slot_mapping(context_slot_mapping)
        if (
            not eligible
            or num_tokens not in self.graphs
            or shared_slot_mapping is None
            or context_states.ndim != 2
            or context_positions.shape != (num_tokens,)
            or shared_slot_mapping.shape != (num_tokens,)
            or context_states.shape[1:] != self.context_states.shape[1:]
            or context_states.dtype != self.context_states.dtype
            or context_states.device != self.device
            or context_positions.dtype != torch.int64
            or context_positions.device != self.device
            or shared_slot_mapping.dtype != torch.int64
            or shared_slot_mapping.device != self.device
        ):
            self.fallback_count += 1
            return False

        self.context_states[:num_tokens].copy_(context_states)
        self.context_positions[:num_tokens].copy_(context_positions)
        self.context_slot_mapping[:num_tokens].copy_(shared_slot_mapping)
        self.graphs[num_tokens].replay()
        self.replay_count += 1
        if self.replay_count == 1:
            logger.info("Replayed first DSpark context graph at %d tokens", num_tokens)
        return True


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        input_batch.dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            input_batch.num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


class DFlashCudaGraphManager(CudaGraphManager):
    """DFlash CudaGraphManager for the parallel-drafting query forward,
    building its own attention metadata from scratch."""

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = _prepare_dflash_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
                causal=causal,
            )
            attn_metadata, slot_mappings = attn_state

            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)
