# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

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
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None

        cudagraph_mode: CUDAGraphMode
        if self.compilation_config.cudagraph_mode is None:
            cudagraph_mode = CUDAGraphMode.NONE
        else:
            cudagraph_mode = self.compilation_config.cudagraph_mode
            if cudagraph_mode == CUDAGraphMode.FULL:
                # NOTE(woosuk): For Eagle, we only use CUDA graphs for decode.
                cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

        self.cudagraph_mode = cudagraph_mode

        self.cudagraph_sizes = get_cudagraph_sizes(
            self.compilation_config.cudagraph_capture_sizes,
            self.max_num_reqs,
            self.max_num_tokens,
            self.cudagraph_mode,
        )

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def get_cudagraph_size(self, num_tokens: int) -> int | None:
        return self.cudagraph_sizes.get(num_tokens)

    def capture_graph(
        self,
        num_tokens: int,
        generate_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        num_reqs = min(num_tokens, self.max_num_reqs)
        attn_metadata = prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_metadata_builders,
            self.max_model_len,
            kv_cache_config,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)

        # Warm up.
        generate_fn(num_tokens, attn_metadata, num_tokens_across_dp)

        # Capture the graph.
        assert num_tokens not in self.graphs
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.pool):
            generate_fn(num_tokens, attn_metadata, num_tokens_across_dp)
        self.graphs[num_tokens] = graph

    @torch.inference_mode()
    def capture(
        self,
        generate_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        capture_graphs(
            self.cudagraph_sizes,
            self.device,
            self.capture_graph,
            generate_fn=generate_fn,
            input_buffers=input_buffers,
            block_tables=block_tables,
            attn_metadata_builders=attn_metadata_builders,
            kv_cache_config=kv_cache_config,
        )

    def run(self, num_tokens: int) -> None:
        assert num_tokens in self.graphs
        self.graphs[num_tokens].replay()
