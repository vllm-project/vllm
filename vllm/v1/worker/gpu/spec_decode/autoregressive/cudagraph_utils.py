# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from copy import copy

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
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

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
    ) -> None:
        # Autoregressive draft decode always processes exactly one token per
        # request. Its graph shapes must therefore stay fixed at query length 1
        # instead of following the target model's Dynamic-SD verification
        # schedule. Use a shallow config copy only while candidates are built;
        # keep the original config on the manager for capture/runtime behavior.
        candidate_config = vllm_config
        speculative_config = vllm_config.speculative_config
        if (
            decode_query_len == 1
            and speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            candidate_config = copy(vllm_config)
            candidate_config.speculative_config = None

        super().__init__(
            candidate_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
        )
        self.vllm_config = vllm_config

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

            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)
