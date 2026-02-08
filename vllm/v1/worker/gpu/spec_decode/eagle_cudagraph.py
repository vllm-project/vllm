# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    capture_graphs,
    get_cudagraph_sizes,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers


class EagleCudaGraphManager:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None

        # NOTE(woosuk): For Eagle, we only use CUDA graphs for decode.
        self.cudagraph_mode = self.compilation_config.cudagraph_mode.decode_mode()

        # only need to capture uniform decode cudagraph sizes (the 2nd return value)
        _, self.cudagraph_sizes = get_cudagraph_sizes(
            self.compilation_config.cudagraph_capture_sizes,
            self.max_num_reqs,
            self.max_num_tokens,
            self.cudagraph_mode,
            uniform_decode_query_len=1,
            uniform_decode_cudagraph=True,
        )

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def get_cudagraph_size(self, num_tokens: int) -> int | None:
        return self.cudagraph_sizes.get(num_tokens)

    def capture_graph(
        self,
        num_tokens: int,
        capture_cudagraph_mode: CUDAGraphMode,
        generate_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
        uniform_decode: bool = True,
    ) -> None:
        if capture_cudagraph_mode == CUDAGraphMode.PIECEWISE:
            capture_fn = self._capture_piecewise_graph
        elif capture_cudagraph_mode == CUDAGraphMode.FULL:
            capture_fn = self._capture_full_graph
        else:
            raise ValueError(
                f"Unexpected cudagraph_mode for capture: {capture_cudagraph_mode}"
            )

        num_reqs = min(num_tokens, self.max_num_reqs)
        attn_metadata, slot_mappings = prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_metadata_builders,
            self.max_model_len,
            kv_cache_config,
            uniform_decode_query_len=1,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)

        # Warm up.
        generate_fn(
            num_reqs,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            CUDAGraphMode.NONE,
        )

        # Capture the graph.
        capture_fn(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            generate_fn=generate_fn,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
        )

    def _capture_full_graph(
        self,
        num_reqs: int,
        num_tokens: int,
        generate_fn: Callable,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens_across_dp: torch.Tensor,
    ) -> None:
        assert num_tokens not in self.graphs
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.pool):
            generate_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                CUDAGraphMode.NONE,
            )
        self.graphs[num_tokens] = graph

    def _capture_piecewise_graph(
        self,
        num_reqs: int,
        num_tokens: int,
        generate_fn: Callable,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens_across_dp: torch.Tensor,
    ) -> None:
        generate_fn(
            num_reqs,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            CUDAGraphMode.PIECEWISE,
        )

    @torch.inference_mode()
    def capture(
        self,
        generate_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        if self.cudagraph_mode == CUDAGraphMode.NONE:
            return

        capture_graphs(
            self.cudagraph_sizes,
            self.device,
            self.capture_graph,
            capture_cudagraph_mode=self.cudagraph_mode,
            desc=f"Capturing eagle CUDA graphs ({self.cudagraph_mode.name})",
            generate_fn=generate_fn,
            input_buffers=input_buffers,
            block_tables=block_tables,
            attn_metadata_builders=attn_metadata_builders,
            kv_cache_config=kv_cache_config,
            uniform_decode=True,
        )

    def run(self, num_tokens: int) -> None:
        assert num_tokens in self.graphs
        self.graphs[num_tokens].replay()
