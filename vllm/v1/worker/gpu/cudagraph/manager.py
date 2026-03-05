# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class CudaGraphKey:
    is_full_graph: bool

    num_reqs: int
    total_num_tokens: int
    is_uniform: bool

    # Extra information for the key
    has_lora: bool = False


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_aux_hidden_state_outputs: bool,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.compilation_config = vllm_config.compilation_config
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        self.device = device

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        self.cudagraph_mode = self.compilation_config.cudagraph_mode
        self.decode_len = 1
        if self.speculative_config is not None:
            self.decode_len += self.speculative_config.num_speculative_tokens

        self.num_target_tokens = min(self.max_num_tokens, 2048)

        self.full_graph_keys: np.ndarray | None = None
        self.full_graphs: dict[CudaGraphKey, torch.cuda.CUDAGraph] = {}
        self.pw_graph_keys: np.ndarray | None = None
        # TODO(woosuk): Share the same pool with PW cuda graphs.
        self.pool = torch.cuda.graph_pool_handle()

        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []

    def needs_capture(self) -> bool:
        return self.cudagraph_mode != CUDAGraphMode.NONE

    def capture_graph(
        self,
        cudagraph_key: CudaGraphKey,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        num_reqs = cudagraph_key.num_reqs
        num_tokens = cudagraph_key.total_num_tokens

        model_inputs = {
            "input_ids": input_buffers.input_ids[:num_tokens],
            "positions": input_buffers.positions[:num_tokens],
            # NOTE: Values returned by `prepare_dummy_inputs` will override the
            # default values above.
            **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
        }

        # FIXME
        input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
        input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
        slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, kv_cache_config
        )

        # HACK(woosuk): Special handling for DCP.
        if block_tables.cp_size > 1:
            prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                input_batch.seq_lens,
                num_reqs,
                block_tables.cp_size,
                block_tables.cp_rank,
                block_tables.cp_interleave,
            )
            input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

        attn_metadata = model_state.prepare_attn(
            input_batch,
            input_block_tables,
            slot_mappings,
            attn_groups,
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
            slot_mapping=slot_mappings_by_layer,
        ):
            model_output = model(**model_inputs)
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

        # Allocate output buffers if not already done.
        if self.hidden_states is None:
            self.hidden_states = torch.empty_like(hidden_states)
        if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
            self.aux_hidden_states = [torch.empty_like(x) for x in aux_hidden_states]

        # select and check capture function
        if cudagraph_key.is_full_graph:
            capture_fn = self._capture_full_graph
        else:
            capture_fn = self._capture_piecewise_graph
        capture_fn(
            cudagraph_key=cudagraph_key,
            model=model,
            model_inputs=model_inputs,
            num_tokens_across_dp=num_tokens_across_dp,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings_by_layer,
        )

    def _capture_full_graph(
        self,
        cudagraph_key: CudaGraphKey,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor | None],
        num_tokens_across_dp: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
    ) -> None:
        assert attn_metadata is not None
        num_tokens = cudagraph_key.total_num_tokens

        # Capture the graph.
        assert cudagraph_key not in self.full_graphs
        graph = torch.cuda.CUDAGraph()

        # Sync offloader's copy stream before capture.
        # Ensure any pre-capture prefetches from offloader are complete.
        get_offloader().sync_prev_onload()

        with (
            set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
            ),
            torch.cuda.graph(graph, self.pool),
        ):
            model_output = model(**model_inputs)

            # Join offloader's copy stream after forward to avoid unjoined
            # stream error. The last layer's start_prefetch forks copy_stream,
            # but wait_prefetch only happens in the next forward pass.
            get_offloader().join_after_forward()

            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            # Copy outputs to the output buffers.
            assert self.hidden_states is not None
            self.hidden_states[:num_tokens] = hidden_states
            if self.use_aux_hidden_state_outputs:
                for i, aux_hidden in enumerate(aux_hidden_states):
                    self.aux_hidden_states[i][:num_tokens] = aux_hidden

        # Save the graph.
        self.full_graphs[cudagraph_key] = graph

    def _capture_piecewise_graph(
        self,
        cudagraph_key: CudaGraphKey,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor | None],
        num_tokens_across_dp: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
    ) -> None:
        # create batch descriptor for piecewise cudagraph dispatch key
        batch_descriptor = BatchDescriptor(
            num_tokens=cudagraph_key.total_num_tokens,
            has_lora=cudagraph_key.has_lora,
        )

        # Capture run - CUDAGraphWrapper inside torch.compile will auto capture.
        with set_forward_context(
            attn_metadata=None,  # piecewise no need attn_metadata
            vllm_config=self.vllm_config,
            num_tokens=cudagraph_key.total_num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings,
        ):
            model(**model_inputs)

    @torch.inference_mode()
    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Phase 1: Capture for mixed prefill-decode batches if needed.
        mixed_mode = self.cudagraph_mode.mixed_mode()
        if mixed_mode != CUDAGraphMode.NONE:
            # TODO(woosuk): Make this configurable.
            cudagraph_sizes = [1, 2, 4, 8]
            cudagraph_sizes.extend(range(16, 256, 16))
            cudagraph_sizes.extend(range(256, 1024, 32))
            cudagraph_sizes.extend(range(1024, 8193, 64))
            cudagraph_sizes = [
                x for x in cudagraph_sizes if x <= self.num_target_tokens
            ]
            cudagraph_sizes = sorted(cudagraph_sizes, reverse=True)

            use_full_graph = mixed_mode == CUDAGraphMode.FULL
            graph_keys = []
            for size in cudagraph_sizes:
                key = CudaGraphKey(
                    is_full_graph=use_full_graph,
                    num_reqs=min(size, self.max_num_reqs),
                    total_num_tokens=size,
                    is_uniform=False,
                )
                graph_keys.append(key)

            if is_global_first_rank():
                tqdm_bar = tqdm(
                    graph_keys, desc=f"Capturing CUDA graphs (mixed, {mixed_mode.name})"
                )

            with graph_capture(device=self.device):
                for cudagraph_key in tqdm_bar:
                    self.capture_graph(
                        cudagraph_key=cudagraph_key,
                        model=model,
                        model_state=model_state,
                        input_buffers=input_buffers,
                        block_tables=block_tables,
                        attn_groups=attn_groups,
                        kv_cache_config=kv_cache_config,
                    )

            num_graphs = len(cudagraph_sizes)
            if use_full_graph:
                self.full_graph_keys = np.empty((num_graphs, 3), dtype=np.int32)
                for i, key in enumerate(graph_keys):
                    self.full_graph_keys[i, 0] = key.num_reqs
                    self.full_graph_keys[i, 1] = key.total_num_tokens
                    self.full_graph_keys[i, 2] = key.is_uniform
            else:
                self.pw_graph_keys = np.empty((num_graphs, 1), dtype=np.int32)
                for i, key in enumerate(graph_keys):
                    self.pw_graph_keys[i, 0] = key.total_num_tokens

        # Phase 2: Capture FULL graphs for uniform decode batches if needed.
        # This is only needed if the mixed mode is not FULL.
        decode_mode = self.cudagraph_mode.decode_mode()
        if decode_mode == CUDAGraphMode.FULL and mixed_mode != CUDAGraphMode.FULL:
            # TODO(woosuk): Make this configurable.
            max_num_reqs = min(
                self.max_num_reqs, self.num_target_tokens // self.decode_len
            )
            cudagraph_num_reqs = [1, 2, 4, 8]
            cudagraph_num_reqs.extend(range(16, 256, 16))
            cudagraph_num_reqs.extend(range(256, 1024, 32))
            cudagraph_num_reqs.extend(range(1024, 8193, 64))
            cudagraph_num_reqs = [x for x in cudagraph_num_reqs if x <= max_num_reqs]
            cudagraph_num_reqs = sorted(cudagraph_num_reqs, reverse=True)

            graph_keys = []
            for num_reqs in cudagraph_num_reqs:
                key = CudaGraphKey(
                    is_full_graph=True,
                    num_reqs=num_reqs,
                    total_num_tokens=num_reqs * self.decode_len,
                    is_uniform=True,
                )
                graph_keys.append(key)

            if is_global_first_rank():
                tqdm_bar = tqdm(
                    graph_keys,
                    desc=f"Capturing CUDA graphs (decode, {decode_mode.name})",
                )

            with graph_capture(device=self.device):
                for cudagraph_key in tqdm_bar:
                    self.capture_graph(
                        cudagraph_key=cudagraph_key,
                        model=model,
                        model_state=model_state,
                        input_buffers=input_buffers,
                        block_tables=block_tables,
                        attn_groups=attn_groups,
                        kv_cache_config=kv_cache_config,
                    )

            num_graphs = len(cudagraph_num_reqs)
            self.full_graph_keys = np.empty((num_graphs, 3), dtype=np.int32)
            for i, key in enumerate(graph_keys):
                self.full_graph_keys[i, 0] = key.num_reqs
                self.full_graph_keys[i, 1] = key.total_num_tokens
                self.full_graph_keys[i, 2] = key.is_uniform

    def get_full_cudagraph_key(
        self, num_scheduled_tokens: Sequence[int]
    ) -> CudaGraphKey | None:
        if self.full_graph_keys is None:
            return None

        num_reqs = len(num_scheduled_tokens)
        total_num_tokens = sum(num_scheduled_tokens)
        is_uniform = all(x == self.decode_len for x in num_scheduled_tokens)

        # The CUDA graph size should not be smaller than the input size.
        mask = self.full_graph_keys[:, 0] >= num_reqs
        mask &= self.full_graph_keys[:, 1] >= total_num_tokens
        mask &= self.full_graph_keys[:, 2] == is_uniform

        graph_keys = self.full_graph_keys[mask]
        if graph_keys.shape[0] == 0:
            # No matching CUDA graph.
            return None

        # Find the "smallest" CUDA graph.
        graph_key = graph_keys[graph_keys[:, 1].argmin()]
        return CudaGraphKey(
            is_full_graph=True,
            num_reqs=graph_key[0],
            total_num_tokens=graph_key[1],
            is_uniform=is_uniform,
        )

    def get_pw_cudagraph_key(
        self, num_scheduled_tokens: Sequence[int]
    ) -> CudaGraphKey | None:
        if self.pw_graph_keys is None:
            return None

        total_num_tokens = sum(num_scheduled_tokens)
        mask = self.pw_graph_keys[:, 0] >= total_num_tokens
        graph_keys = self.pw_graph_keys[mask]
        if graph_keys.shape[0] == 0:
            # No matching PW CUDA graph.
            return None

        # Find the "smallest" PW CUDA graph.
        graph_key = graph_keys[graph_keys[:, 0].argmin()]
        return CudaGraphKey(
            is_full_graph=False,
            num_reqs=len(num_scheduled_tokens),
            total_num_tokens=graph_key[0],
            is_uniform=False,
        )

    def get_cudagraph_key(
        self, num_scheduled_tokens: Sequence[int]
    ) -> CudaGraphKey | None:
        full_graph_key = self.get_full_cudagraph_key(num_scheduled_tokens)
        if full_graph_key is not None:
            return full_graph_key
        pw_graph_key = self.get_pw_cudagraph_key(num_scheduled_tokens)
        if pw_graph_key is not None:
            return pw_graph_key
        return None

    def run_fullgraph(
        self, cudagraph_key: CudaGraphKey
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        cudagraph = self.full_graphs.get(cudagraph_key)
        assert cudagraph is not None, f"No cudagraph for {cudagraph_key}"

        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        cudagraph.replay()
        assert self.hidden_states is not None
        num_tokens = cudagraph_key.total_num_tokens
        hidden_states = self.hidden_states[:num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[:num_tokens] for x in self.aux_hidden_states]
