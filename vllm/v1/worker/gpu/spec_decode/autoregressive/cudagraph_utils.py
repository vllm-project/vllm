# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
    uniform_dp_token_counts,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class SpeculatorCudaGraphManager(CudaGraphManager):
    """CudaGraphManager for draft prefill and decode.

    Builds fresh dummy inputs and attention metadata for every warmup and
    capture pass so that the contents of the shared persistent buffers
    (e.g. query_start_loc, seq_lens, FA3 scheduler metadata) always match
    the batch descriptor being captured. Reusing metadata built during an
    earlier capture would execute kernels with stale buffer contents.
    """

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp, moe_non_sp_token_counts = uniform_dp_token_counts(
                self.vllm_config.parallel_config, num_tokens
            )
            # Capture runs one shape on every rank, so the agreement is known
            # without a collective.
            dp_sync = (
                DPSyncState(
                    num_tokens_across_dp=num_tokens_across_dp,
                    moe_non_sp_token_counts=moe_non_sp_token_counts,
                    uniform_token_count=None,
                    eager=False,
                    num_reqs=num_reqs,
                )
                if num_tokens_across_dp is not None
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

            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                dp_sync,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)
