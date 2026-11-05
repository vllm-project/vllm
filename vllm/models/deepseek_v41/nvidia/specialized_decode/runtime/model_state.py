# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model state of the specialized decode.

``DeepseekV41SpecializedDecodeModelState`` is ``DeepseekV41ModelState`` plus:

- the decode executor, built and warmed up once the weights are loaded (the
  model runner creates the model state after loading the target and draft
  models, and before memory profiling);
- CUDA graphs of the attention metadata builds of FULL decode steps
  (``metadata_graphs``), recorded with the inner CUDA graphs and replayed in
  place of the eager build. A step that replays one gathers its engram
  lookback windows on a side stream beside it, so the metadata graph hands
  off to the forward's graph with no eager kernel between them;
- no replay-start upload for decode steps whose starts are known to be zero.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.utils.torch_utils import current_stream
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from ...model_state import DeepseekV41ModelState
from .metadata_graphs import AttnMetadataCudaGraphs, GraphKey
from .warmup import warm_up


def padded_decode_batch(
    key: GraphKey, input_buffers: InputBuffers, max_model_len: int
) -> InputBatch:
    """A dummy uniform decode batch padded like a runtime step of ``key``. Like
    a capture, it takes worst-case sequence lengths, which bound host values
    only, so its recorded build serves every step of the key."""
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
            (num_reqs_padded,), max_model_len, dtype=torch.int32
        ),
        input_ids=input_buffers.input_ids[:num_tokens_padded],
        positions=input_buffers.positions[:num_tokens_padded],
        is_padding=input_buffers.is_padding[:num_tokens_padded],
    )


class DeepseekV41SpecializedDecodeModelState(DeepseekV41ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        language_model = (
            model.get_language_model()
            if hasattr(model, "get_language_model")
            else model
        )
        self.decode = language_model.build_specialized_decode()
        warm_up(self.decode, vllm_config)
        self.graphs = AttnMetadataCudaGraphs(
            vllm_config,
            vllm_config.num_speculative_tokens + self.num_new_sampled_tokens_per_step,
        )
        # Leading rows of _replay_start known to hold zeros.
        self._zero_replay_starts = 0
        # The key whose metadata build capture_inner_cudagraphs records.
        self._recording: GraphKey | None = None
        # The runner's request state, from the first prepare_inputs on; the
        # lookback gather that runs beside a metadata graph reads it.
        self._req_states: RequestState | None = None
        # The gather's side stream and the event recorded after it.
        self._lookback_side: tuple[torch.cuda.Stream, torch.cuda.Event] | None = None
        # The batch whose lookback gather was issued beside its metadata graph,
        # and the event that ends it.
        self._staged_lookback: tuple[InputBatch, torch.cuda.Event] | None = None

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
        # Memory profiling's recordings wrote the builders of its KV cache.
        self.graphs.release()

    def _upload_replay_start(self, values: np.ndarray, num_reqs: int) -> torch.Tensor:
        # Decode steps upload only zeros; the capture's zeroing is the only
        # other writer, and it writes zeros.
        has_replay = bool(values.any())
        if has_replay or num_reqs > self._zero_replay_starts:
            super()._upload_replay_start(values, num_reqs)
            self._zero_replay_starts = 0 if has_replay else num_reqs
        return self._replay_start[:num_reqs]

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        self._req_states = req_states
        return super().prepare_inputs(input_batch, req_states)

    def _stage_lookback(self, input_batch: InputBatch) -> None:
        """Issue the batch's lookback gather on a side stream, beside the
        metadata graph about to replay. It reads the batch's request indices
        and the token history, which the runner has written by now."""
        if self.lookback_token_ids is None or self._req_states is None:
            return
        if self._lookback_side is None:
            self._lookback_side = (torch.cuda.Stream(), torch.cuda.Event())
        self._join_staged_lookback()
        stream, event = self._lookback_side
        stream.wait_stream(current_stream())
        with torch.cuda.stream(stream):
            super()._fill_lookback(input_batch, self._req_states)
        event.record(stream)
        self._staged_lookback = (input_batch, event)

    def _join_staged_lookback(self) -> InputBatch | None:
        """Make the current stream wait for a staged lookback gather, if any;
        returns the batch it gathered."""
        staged, self._staged_lookback = self._staged_lookback, None
        if staged is None:
            return None
        current_stream().wait_event(staged[1])
        return staged[0]

    def _fill_lookback(self, input_batch: InputBatch, req_states: RequestState) -> None:
        # Even a stale gather (a batch whose inputs were never prepared) is
        # joined, so that no two gathers write the window at once.
        if self._join_staged_lookback() is not input_batch:
            super()._fill_lookback(input_batch, req_states)

    def _build_attn_metadata(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        build = super()._build_attn_metadata
        if self._recording is not None:
            return self.graphs.capture(
                self._recording,
                attn_groups,
                block_tables,
                lambda groups: build(
                    input_batch,
                    cudagraph_mode,
                    block_tables,
                    slot_mappings,
                    groups,
                    kv_cache_config,
                    **kwargs,
                ),
            )
        if not for_capture and self.graphs.can_record(attn_groups):
            key = self.graphs.key(input_batch, cudagraph_mode)
            if key is not None:
                # decode_backend=latency_specialized refuses micro-batching.
                assert ubatch_idx == 0
                self._stage_lookback(input_batch)
                attn_metadata = self.graphs.replay(key)
                if attn_metadata is not None:
                    return attn_metadata
        return build(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            **kwargs,
        )

    def capture_inner_cudagraphs(
        self,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        # Record the metadata build of every decode step the FULL decode graphs
        # serve, on dummy batches of that step's shape.
        self.graphs.release()
        if self.graphs.can_record(attn_groups):
            for key in self.graphs.capture_keys():
                self._recording = key
                try:
                    self.prepare_attn(
                        padded_decode_batch(key, input_buffers, self.max_model_len),
                        CUDAGraphMode.FULL,
                        block_tables.get_dummy_block_tables(key[1]),
                        block_tables.get_dummy_slot_mappings(key[3]),
                        attn_groups,
                        kv_cache_config,
                    )
                finally:
                    self._recording = None
        super().capture_inner_cudagraphs(
            input_buffers, block_tables, attn_groups, kv_cache_config, stream
        )
