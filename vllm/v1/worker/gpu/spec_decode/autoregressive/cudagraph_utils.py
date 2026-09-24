# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class SpeculatorBatchDescriptor(BatchExecutionDescriptor):
    num_speculative_tokens: int = 0


class SpeculatorCudaGraphManager(CudaGraphManager):
    """CudaGraphManager for draft prefill and decode.

    Builds fresh dummy inputs and attention metadata for every warmup and
    capture pass so that the contents of the shared persistent buffers
    (e.g. query_start_loc, seq_lens, FA3 scheduler metadata) always match
    the batch descriptor being captured. Reusing metadata built during an
    earlier capture would execute kernels with stale buffer contents.
    """

    def specialize_spec_tokens(
        self, desc: BatchExecutionDescriptor, num_speculative_tokens: int
    ) -> BatchExecutionDescriptor:
        if desc.cg_mode != CUDAGraphMode.FULL or desc in self.graphs:
            return desc
        desc = SpeculatorBatchDescriptor(
            **vars(desc), num_speculative_tokens=num_speculative_tokens
        )
        return (
            desc if desc in self.graphs else replace(desc, cg_mode=CUDAGraphMode.NONE)
        )

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
        specialize_spec_tokens: bool = False,
    ) -> None:
        if specialize_spec_tokens:
            speculative_config = self.vllm_config.speculative_config
            assert speculative_config is not None
            dense_schedule = build_dynamic_sd_schedule_lookup(
                speculative_config.num_speculative_tokens_per_batch_size,
                vllm_max_batch_size=self.max_num_reqs,
                vllm_num_speculative_tokens=self.vllm_config.num_speculative_tokens,
            )
            self._capture_descs = {
                mode: [
                    SpeculatorBatchDescriptor(**vars(desc), num_speculative_tokens=k)
                    for desc in descs
                    for k in sorted(set(dense_schedule) - {0, 1}, reverse=True)
                ]
                for mode, descs in self._capture_descs.items()
            }

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
            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
            )

            forward_kwargs = {}
            if self.decode_query_len == 1:
                forward_kwargs["num_speculative_steps"] = (
                    desc.num_speculative_tokens
                    if isinstance(desc, SpeculatorBatchDescriptor)
                    else self.vllm_config.num_speculative_tokens
                )
            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
                **forward_kwargs,
            )

        super().capture(create_forward_fn, progress_bar_desc)
