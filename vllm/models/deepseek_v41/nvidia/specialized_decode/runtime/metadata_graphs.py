# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the attention metadata builds of FULL decode steps.

A FULL CUDA graph replays the forward only: before every replay the runner
still builds the attention metadata on the host, which refreshes the persistent
buffers the graph reads. For small decode batches that build costs about as
much as the replay. ``AttnMetadataCudaGraphs`` records, during CUDA graph
capture, the build of every uniform decode batch shape that a FULL decode graph
serves, and replays it on the steps of that shape, returning the metadata
objects captured with it. A step of a shape without a recording builds its
metadata eagerly.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from vllm.compilation.cudagraph_pool import capture_pool
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.compilation import (
    decode_cudagraph_capture_max_tokens,
    decode_cudagraph_capture_shapes,
)
from vllm.logger import init_logger
from vllm.models.deepseek_v41.compressor import CompressorMetadataBuilder
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekSparseSWAFlashInferMetadataBuilder,
    DeepseekV4FlashInferSparseMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backends.mla.sparse_indexer import (
    DeepseekV41SparseIndexerMetadataBuilder,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

# Builders whose build() of a uniform decode batch launches only capture-safe
# work (no device-to-host reads, no pageable host-to-device copies) into
# persistent buffers, with host arguments set by the batch shape.
_RECORDABLE_BUILDERS = frozenset(
    {
        CompressorMetadataBuilder,
        DeepseekSparseSWAFlashInferMetadataBuilder,
        DeepseekV41SparseIndexerMetadataBuilder,
        DeepseekV4FlashInferSparseMLAMetadataBuilder,
    }
)

GraphKey = tuple[int, int, int, int]
# Builds the metadata of the attention groups it is given.
BuildFn = Callable[[list[list[AttentionGroup]]], dict[str, Any]]


class AttnMetadataCudaGraphs:
    """Records attention metadata builds by graph key during CUDA graph
    capture, and replays them.

    Each key is built once eagerly, warming lazy allocations, kernel
    compilation and per-stream library state, then recorded with the metadata
    it returns. The graphs write the builders' buffers at their capture-time
    addresses: every capture releases and records them again.
    """

    def __init__(self, vllm_config: VllmConfig, decode_query_len: int) -> None:
        self._vllm_config = vllm_config
        self._decode_query_len = decode_query_len
        self._graphs: dict[GraphKey, tuple[torch.cuda.CUDAGraph, dict[str, Any]]] = {}
        self._side_streams: list[torch.cuda.Stream] = []
        self._recordable: bool | None = None

    @staticmethod
    def key(input_batch: InputBatch, cudagraph_mode: CUDAGraphMode) -> GraphKey | None:
        """The graph key of a step whose metadata build can be recorded, or None.

        Only FULL steps of uniform decodes qualify; their key fixes every host
        value the build reads. Under adaptive verification (``max_query_len``
        set) the device splits the drafts, so the host query lengths are only
        bounds.
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
        """The keys of the decode steps the FULL decode graphs serve, largest
        first.

        The graphs are the ones ``CudaGraphManager`` captures, and a step runs
        the smallest graph that holds its requests. Under dynamic speculative
        decoding only the widest query length is recorded; steps of the
        narrower tiers build their metadata eagerly.
        """
        config = self._vllm_config.compilation_config
        query_len = self._decode_query_len
        max_reqs = self._vllm_config.get_decode_cudagraph_query_lens(query_len)
        graph_reqs = [
            n
            for n, _ in decode_cudagraph_capture_shapes(
                config.decode_cudagraph_capture_request_sizes or (),
                {query_len: max_reqs.get(query_len, 0)},
                decode_cudagraph_capture_max_tokens(config),
            )
        ]
        keys: list[GraphKey] = []
        num_reqs = 1
        for padded in graph_reqs:
            keys += [
                (n, padded, n * query_len, padded * query_len)
                for n in range(num_reqs, padded + 1)
            ]
            num_reqs = padded + 1
        return keys[::-1]

    def can_record(self, attn_groups: list[list[AttentionGroup]]) -> bool:
        """Whether every metadata builder of ``attn_groups`` is recordable."""
        if self._recordable is None:
            self._recordable = all(
                type(group.get_metadata_builder()) in _RECORDABLE_BUILDERS
                for groups in attn_groups
                for group in groups
            )
        return self._recordable

    def capture(
        self,
        key: GraphKey,
        attn_groups: list[list[AttentionGroup]],
        block_tables: Sequence[torch.Tensor],
        build_fn: BuildFn,
    ) -> dict[str, Any]:
        """Record the build of ``key`` in the global CUDA graph pool, which the
        FULL forward graphs and memory profiling use; returns its metadata."""
        self._forked_build(attn_groups, block_tables, build_fn)
        graph = torch.cuda.CUDAGraph()
        with (
            capture_pool(
                current_platform.get_global_graph_pool(), self._vllm_config
            ) as pool,
            torch.cuda.graph(graph, pool),
        ):
            attn_metadata = self._forked_build(attn_groups, block_tables, build_fn)
        self._graphs[key] = (graph, attn_metadata)
        return attn_metadata

    def replay(self, key: GraphKey) -> dict[str, Any] | None:
        """Replay the build of ``key`` and return its metadata, or None if the
        key was not recorded."""
        entry = self._graphs.get(key)
        if entry is None:
            logger.warning_once(
                "DeepSeek-V4.1 specialized decode: a FULL decode step has no "
                "recorded attention-metadata build; building it eagerly."
            )
            return None
        graph, attn_metadata = entry
        graph.replay()
        return attn_metadata

    def _forked_build(
        self,
        attn_groups: list[list[AttentionGroup]],
        block_tables: Sequence[torch.Tensor],
        build_fn: BuildFn,
    ) -> dict[str, Any]:
        """Build the KV-cache groups as concurrent branches on side streams.

        The groups share only read-only inputs, with two exceptions. The sparse
        MLA builder clamps its block table in place, so groups whose block
        tables share storage stay on one branch, in serial order. And the first
        group that needs the token-to-request map builds it into its own buffer,
        which later groups reuse. Each branch builds its own map; the serial
        build's producer is also the first of its branch to need one, so it
        fills the buffer that the FULL forward graphs read.
        """
        branches: dict[int, list[int]] = {}
        for i, groups in enumerate(attn_groups):
            if groups:
                storage = block_tables[i].untyped_storage().data_ptr()
                branches.setdefault(storage, []).append(i)
        if len(branches) < 2:
            return build_fn(attn_groups)
        # Reused, so a warm-up creates per-stream state (e.g. DeepGEMM's) on the
        # streams its capture records.
        while len(self._side_streams) < len(branches):
            self._side_streams.append(torch.cuda.Stream())
        main_stream = current_stream()
        built: dict[str, Any] = {}
        for stream, branch in zip(self._side_streams, branches.values()):
            stream.wait_stream(main_stream)
            with torch.cuda.stream(stream):
                built.update(
                    build_fn(
                        [g if i in branch else [] for i, g in enumerate(attn_groups)]
                    )
                )
        for stream in self._side_streams[: len(branches)]:
            main_stream.wait_stream(stream)
        # The serial build's layer order.
        return {
            name: built[name]
            for groups in attn_groups
            for group in groups
            for name in group.layer_names
        }

    def release(self) -> None:
        """Drop the recorded builds."""
        self._graphs.clear()
        self._recordable = None
