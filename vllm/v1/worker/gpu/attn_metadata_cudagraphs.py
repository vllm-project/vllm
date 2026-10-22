# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the attention metadata builds of FULL decode steps.

A FULL CUDA graph replays the forward only: before every replay the runner
still builds the attention metadata on the host, which refreshes the persistent
buffers the graph reads. For small decode batches that build costs a large
share of the step. ``AttnMetadataCudaGraphs`` records, during CUDA graph
capture, the build of every uniform decode batch shape that a FULL decode graph
serves, and replays it on the steps of that shape, returning the metadata
objects recorded with it. A step of a shape without a recording builds its
metadata eagerly.

A build is recordable when every metadata builder sets
``supports_recorded_decode_build``.
"""

import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from vllm.compilation.cudagraph_pool import capture_pool
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

# (num_reqs, num_reqs_after_padding, num_tokens, num_tokens_after_padding)
GraphKey = tuple[int, int, int, int]


class AttnMetadataCudaGraphs:
    """Records attention metadata builds by batch shape during CUDA graph
    capture, and replays them.

    Each key is built once eagerly, warming lazy allocations, then recorded
    with the metadata it returns. The graphs write the builders' buffers at
    their capture-time addresses, so every capture records them again.
    """

    def __init__(self, vllm_config: VllmConfig, decode_query_len: int) -> None:
        self.vllm_config = vllm_config
        self.decode_query_len = decode_query_len
        self.max_model_len = vllm_config.model_config.max_model_len
        self.graphs: dict[GraphKey, tuple[torch.cuda.CUDAGraph, dict[str, Any]]] = {}
        self.recordable: bool | None = None

    @staticmethod
    def key(input_batch: InputBatch, cudagraph_mode: CUDAGraphMode) -> GraphKey | None:
        """The key of a step whose metadata build can be replayed, or None.

        Only FULL steps of uniform decodes qualify; their key fixes every host
        value the build reads.
        """
        if (
            cudagraph_mode != CUDAGraphMode.FULL
            or input_batch.has_prefill
            or input_batch.max_query_len is not None
        ):
            return None
        query_lens = input_batch.num_scheduled_tokens[: input_batch.num_reqs]
        if not (query_lens == query_lens[0]).all():
            return None
        return (
            input_batch.num_reqs,
            input_batch.num_reqs_after_padding,
            input_batch.num_tokens,
            input_batch.num_tokens_after_padding,
        )

    def capture_keys(self) -> list[GraphKey]:
        """The keys of the uniform decode steps the FULL decode graphs serve.

        As in ``CudaGraphManager``, each capture size rounds up to whole
        requests of the decode query length, and a step runs the smallest graph
        that holds its requests.
        """
        config = self.vllm_config.compilation_config
        query_len = self.decode_query_len
        max_tokens = min(
            self.vllm_config.scheduler_config.max_num_seqs * query_len,
            config.max_cudagraph_capture_size,
        )
        graph_reqs = sorted(
            {
                round_up(size, query_len) // query_len
                for size in config.cudagraph_capture_sizes or ()
                if round_up(size, query_len) <= max_tokens
            }
        )
        keys: list[GraphKey] = []
        num_reqs = 1
        for padded in graph_reqs:
            keys += [
                (n, padded, n * query_len, padded * query_len)
                for n in range(num_reqs, padded + 1)
            ]
            num_reqs = padded + 1
        return keys

    def can_record(self, attn_groups: list[list[AttentionGroup]]) -> bool:
        """Whether every metadata builder of ``attn_groups`` is recordable."""
        if self.recordable is None:
            self.recordable = all(
                group.get_metadata_builder().supports_recorded_decode_build
                for groups in attn_groups
                for group in groups
            )
        return self.recordable

    def capture(
        self, key: GraphKey, build_fn: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        """Record the build of ``key`` in the global CUDA graph pool, which the
        FULL forward graphs and memory profiling use; returns its metadata."""
        build_fn()
        graph = torch.cuda.CUDAGraph()
        with (
            capture_pool(
                current_platform.get_global_graph_pool(), self.vllm_config
            ) as pool,
            torch.cuda.graph(graph, pool),
            warnings.catch_warnings(),
        ):
            # A build that only wraps persistent device buffers records nothing.
            warnings.filterwarnings("ignore", message="The CUDA Graph is empty")
            attn_metadata = build_fn()
        self.graphs[key] = (graph, attn_metadata)
        return attn_metadata

    def replay(self, key: GraphKey) -> dict[str, Any] | None:
        """Replay the build of ``key`` and return its metadata, or None if the
        key was not recorded."""
        entry = self.graphs.get(key)
        if entry is None:
            return None
        graph, attn_metadata = entry
        graph.replay()
        return attn_metadata

    def release(self) -> None:
        """Drop the recorded builds."""
        self.graphs.clear()
        self.recordable = None

    def padded_decode_batch(
        self, key: GraphKey, input_buffers: InputBuffers
    ) -> InputBatch:
        """A dummy uniform decode batch padded like a runtime step of ``key``.
        Like a capture, it takes worst-case sequence lengths, which bound host
        values only, so its recorded build serves every step of the key."""
        num_reqs, num_reqs_padded, num_tokens, num_tokens_padded = key
        input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
        # The padded requests are empty.
        query_start_loc_np = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
        query_start_loc_np[: num_reqs + 1] = input_batch.query_start_loc_np
        return replace(
            input_batch,
            num_reqs_after_padding=num_reqs_padded,
            num_tokens_after_padding=num_tokens_padded,
            query_start_loc=input_buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_np=query_start_loc_np,
            seq_lens=input_buffers.seq_lens[:num_reqs_padded],
            seq_lens_cpu_upper_bound=torch.full(
                (num_reqs_padded,), self.max_model_len, dtype=torch.int32
            ),
            input_ids=input_buffers.input_ids[:num_tokens_padded],
            positions=input_buffers.positions[:num_tokens_padded],
            is_padding=input_buffers.is_padding[:num_tokens_padded],
        )
