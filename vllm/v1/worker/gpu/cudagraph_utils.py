# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers


class CudaGraphManager:
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
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None

        if self.compilation_config.cudagraph_mode is None:
            self.cudagraph_mode = CUDAGraphMode.NONE
        else:
            self.cudagraph_mode = self.compilation_config.cudagraph_mode
        if self.compilation_config.cudagraph_capture_sizes is not None:
            cudagraph_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
            # Limit the cudagraph sizes to the max decode batch size.
            self.cudagraph_sizes = [
                x for x in cudagraph_sizes if x <= self.max_num_reqs
            ]
        else:
            self.cudagraph_sizes = []
        self.padded_sizes = self._init_padded_sizes()

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.hidden_states: torch.Tensor | None = None

    def _init_padded_sizes(self) -> dict[int, int]:
        if not self.cudagraph_mode.has_full_cudagraphs():
            # Full cuda graphs are not used.
            return {}
        if not self.cudagraph_sizes:
            return {}

        padded_sizes: dict[int, int] = {}
        for i in range(1, self.cudagraph_sizes[-1] + 1):
            for x in self.cudagraph_sizes:
                if i <= x:
                    padded_sizes[i] = x
                    break
        return padded_sizes

    def needs_capture(self) -> bool:
        return len(self.padded_sizes) > 0

    def get_cudagraph_size(
        self,
        scheduler_output: SchedulerOutput,
        num_tokens_after_padding: int,
    ) -> int | None:
        if not self.cudagraph_mode.has_full_cudagraphs():
            return None
        if self.cudagraph_mode != CUDAGraphMode.FULL:
            # TODO(woosuk): Support uniform decode with multiple tokens (spec decoding).
            all_decode = all(
                x == 1 for x in scheduler_output.num_scheduled_tokens.values()
            )
            if not all_decode:
                # Prefill is included.
                return None
        return self.padded_sizes.get(num_tokens_after_padding)

    def capture_graph(
        self,
        batch_size: int,
        model: nn.Module,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        assert batch_size not in self.graphs

        # Prepare dummy inputs.
        input_ids = input_buffers.input_ids.gpu[:batch_size]
        positions = input_buffers.positions[:batch_size]

        input_buffers.query_start_loc.np[: batch_size + 1] = np.arange(batch_size + 1)
        input_buffers.query_start_loc.np[batch_size:] = batch_size
        input_buffers.query_start_loc.copy_to_gpu()
        # HACK(woosuk): To optimize warmup time, we use 1 (instead of max_model_len)
        # for seq_lens. This leads to a mismatch between seq_lens (GPU) and
        # seq_lens_np (CPU), which might cause issues in some attention backends.
        input_buffers.seq_lens[:batch_size] = 1
        input_buffers.seq_lens[batch_size:] = 0

        input_block_tables = [x[:batch_size] for x in block_tables.input_block_tables]
        slot_mappings = block_tables.slot_mappings[:, :batch_size]

        attn_metadata = build_attn_metadata(
            attn_metadata_builders=attn_metadata_builders,
            num_reqs=batch_size,
            num_tokens=batch_size,
            query_start_loc=input_buffers.query_start_loc,
            seq_lens=input_buffers.seq_lens,
            seq_lens_np=np.full(batch_size, self.max_model_len, dtype=np.int32),
            num_computed_tokens_cpu=None,  # FIXME
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, batch_size)

        # Warm up.
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=batch_size,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            hidden_states = model(
                input_ids=input_ids,
                positions=positions,
            )
            if self.hidden_states is None:
                self.hidden_states = torch.empty_like(hidden_states)

        # Capture the graph.
        graph = torch.cuda.CUDAGraph()
        with (
            patch("torch.cuda.empty_cache", lambda: None),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=batch_size,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
            ),
            torch.cuda.graph(graph, self.pool),
        ):
            hidden_states = model(
                input_ids=input_ids,
                positions=positions,
            )
            self.hidden_states[:batch_size] = hidden_states
        self.graphs[batch_size] = graph

    @torch.inference_mode()
    def capture(
        self,
        model: nn.Module,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        assert self.needs_capture()
        # Capture larger graphs first.
        sizes_to_capture = sorted(self.cudagraph_sizes, reverse=True)
        if is_global_first_rank():
            sizes_to_capture = tqdm(sizes_to_capture, desc="Capturing CUDA graphs")

        with graph_capture(device=self.device):
            for batch_size in sizes_to_capture:
                self.capture_graph(
                    batch_size,
                    model,
                    input_buffers,
                    block_tables,
                    attn_metadata_builders,
                    kv_cache_config,
                )

    def run(self, batch_size: int) -> torch.Tensor:
        assert batch_size in self.graphs
        self.graphs[batch_size].replay()
        assert self.hidden_states is not None
        return self.hidden_states[:batch_size]
