# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from typing import Any

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
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None

        if self.compilation_config.cudagraph_mode is None:
            self.cudagraph_mode = CUDAGraphMode.NONE
        else:
            self.cudagraph_mode = self.compilation_config.cudagraph_mode
        self.cudagraph_sizes = get_cudagraph_sizes(
            self.compilation_config.cudagraph_capture_sizes,
            self.max_num_reqs,
            self.max_num_tokens,
            self.cudagraph_mode,
        )

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.hidden_states: torch.Tensor | None = None

    def needs_capture(self) -> bool:
        return len(self.cudagraph_sizes) > 0

    def get_cudagraph_size(
        self,
        scheduler_output: SchedulerOutput,
        num_tokens_after_padding: int,
    ) -> int | None:
        return get_cudagraph_size(
            num_tokens_after_padding,
            scheduler_output.num_scheduled_tokens.values(),
            self.cudagraph_sizes,
            self.cudagraph_mode,
        )

    def capture_graph(
        self,
        num_tokens: int,
        model: nn.Module,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        num_reqs = min(num_tokens, self.max_num_reqs)
        input_ids = input_buffers.input_ids.gpu[:num_tokens]
        positions = input_buffers.positions[:num_tokens]
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
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
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
        assert num_tokens not in self.graphs
        graph = torch.cuda.CUDAGraph()
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
            ),
            torch.cuda.graph(graph, self.pool),
        ):
            hidden_states = model(
                input_ids=input_ids,
                positions=positions,
            )
            self.hidden_states[:num_tokens] = hidden_states
        self.graphs[num_tokens] = graph

    @torch.inference_mode()
    def capture(
        self,
        model: nn.Module,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        capture_graphs(
            self.cudagraph_sizes,
            self.device,
            self.capture_graph,
            model=model,
            input_buffers=input_buffers,
            block_tables=block_tables,
            attn_metadata_builders=attn_metadata_builders,
            kv_cache_config=kv_cache_config,
        )

    def run(self, num_tokens: int) -> torch.Tensor:
        assert num_tokens in self.graphs
        self.graphs[num_tokens].replay()
        assert self.hidden_states is not None
        return self.hidden_states[:num_tokens]


def get_cudagraph_sizes(
    capture_sizes: list[int] | None,
    max_num_reqs: int,
    max_num_tokens: int,
    cudagraph_mode: CUDAGraphMode,
) -> dict[int, int]:
    if not cudagraph_mode.has_full_cudagraphs():
        return {}
    if not capture_sizes:
        return {}

    capture_sizes = sorted(capture_sizes)
    # Limit the capture sizes to the max number of requests or tokens.
    upper_bound = (
        max_num_reqs
        if cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        else max_num_tokens
    )
    capture_sizes = [x for x in capture_sizes if x <= upper_bound]
    if not capture_sizes:
        return {}

    cudagraph_sizes: dict[int, int] = {}
    for i in range(1, capture_sizes[-1] + 1):
        for x in capture_sizes:
            if i <= x:
                cudagraph_sizes[i] = x
                break
    return cudagraph_sizes


def get_cudagraph_size(
    num_tokens_after_dp_padding: int,
    num_tokens_per_request: Iterable[int],
    cudagraph_sizes: dict[int, int],
    cudagraph_mode: CUDAGraphMode,
) -> int | None:
    size = cudagraph_sizes.get(num_tokens_after_dp_padding)
    if size is None:
        # No CUDA graph for this size.
        return None
    if cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
        all_decode = all(x == 1 for x in num_tokens_per_request)
        if not all_decode:
            # Prefill is included.
            return None
    return size


def capture_graphs(
    cudagraph_sizes: dict[int, int],
    device: torch.device,
    capture_fn: Callable,
    **capture_kwargs,
) -> None:
    # Capture larger graphs first.
    sizes_to_capture = sorted(set(cudagraph_sizes.values()), reverse=True)
    if is_global_first_rank():
        sizes_to_capture = tqdm(sizes_to_capture, desc="Capturing CUDA graphs")

    with graph_capture(device=device):
        for size in sizes_to_capture:
            capture_fn(size, **capture_kwargs)


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_metadata_builders: list[AttentionMetadataBuilder],
    max_model_len: int,
    kv_cache_config: KVCacheConfig,
) -> dict[str, Any]:
    num_tokens_per_req = num_tokens // num_reqs
    query_start_loc = input_buffers.query_start_loc
    query_start_loc.np[: num_reqs + 1] = np.arange(num_reqs + 1) * num_tokens_per_req
    query_start_loc.np[num_reqs:] = num_tokens
    query_start_loc.copy_to_gpu()
    seq_lens_np = np.full(num_reqs, max_model_len, dtype=np.int32)
    # HACK(woosuk): To optimize warmup time, we use 1 (instead of max_model_len)
    # for seq_lens. This leads to a mismatch between seq_lens (GPU) and
    # seq_lens_np (CPU), which might cause issues in some attention backends.
    input_buffers.seq_lens[:num_reqs] = 1
    input_buffers.seq_lens[num_reqs:] = 0

    input_block_tables = [x[:num_reqs] for x in block_tables.input_block_tables]
    slot_mappings = block_tables.slot_mappings[:, :num_tokens]

    attn_metadata = build_attn_metadata(
        attn_metadata_builders=attn_metadata_builders,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=query_start_loc.gpu[: num_reqs + 1],
        query_start_loc_cpu=query_start_loc.cpu[: num_reqs + 1],
        seq_lens=input_buffers.seq_lens,
        seq_lens_np=seq_lens_np,
        num_computed_tokens_cpu=None,  # FIXME
        block_tables=input_block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
    )
    return attn_metadata
