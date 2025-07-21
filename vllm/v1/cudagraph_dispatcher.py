# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

from vllm.config import (CompilationLevel, CUDAGraphMode, VllmConfig)
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import AttentionCGSupport

logger = init_logger(__name__)


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to switch between multiple cudagraphs.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.cudagraph_mode = self.compilation_config.cudagraph_mode

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        # Verify if correctly piecewise compilation for attention.
        piecewise_compilation = not vllm_config.model_config.enforce_eager\
            and self.compilation_config.level == CompilationLevel.PIECEWISE
        self.piecewise_attn_compilation = piecewise_compilation and\
            self.compilation_config.is_attention_splitting

        self.keys_initialized = False

    def add_cudagraph_key(self, runtime_mode: CUDAGraphMode,
                          batch_descriptor: BatchDescriptor):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], \
            f"Invalid cudagraph runtime mode: {runtime_mode}"
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(self, capture_mixed_batches: bool,
                                    uniform_decode_query_len: int,
                                    attn_cg_support: AttentionCGSupport):
        # This should be called only after attention backend is initialized.

        # Note: we create all valid keys possible for cudagraph but do not 
        # guarantee all keys would be used. For example, we create keys for
        # piecewise cudagraphs when it is piecewise compilation, which is always
        # vaild, but for attention backend support unified routine, we may not
        # trigger capturing/replaying the piecewise cudagraphs depending on
        # CompilationConfig.cudagraph_mode. In addition, if we allow lazy
        # capturing in future PR, some keys may never be triggered.

        if self.compilation_config.level == CompilationLevel.PIECEWISE\
                and len(self.compilation_config.splitting_ops)>0:
            # add piecewise cudagraph keys.
            for bs in self.compilation_config.cudagraph_capture_sizes:
                self.add_cudagraph_key(CUDAGraphMode.PIECEWISE,
                    BatchDescriptor(num_tokens=bs, is_uniform=None))
        
        # if we need capture full cudagraph for mixed prefill-decode batches.
        create_mixed_batch_full_cg = attn_cg_support in [
                            AttentionCGSupport.ALWAYS_UNIFIED,
                            AttentionCGSupport.ALWAYS_SEPARATE] and \
                            capture_mixed_batches
        if self.cudagraph_mode == CUDAGraphMode.FULL:
            # full cudagraph for mix prefill-decode/general batches
            if create_mixed_batch_full_cg:
                for bs in self.compilation_config.cudagraph_capture_sizes:
                    self.add_cudagraph_key(CUDAGraphMode.FULL,
                        BatchDescriptor(num_tokens=bs, is_uniform=False))

            # always create full cudagraph for uniform batches if cudagraph
            # separate routine is enabled.
            if self.compilation_config.cudagraph_separate_routine:
                max_num_tokens = uniform_decode_query_len * \
                    self.vllm_config.scheduler_config.max_num_seqs
                cudagraph_capture_sizes_for_decode = [
                    x for x in self.compilation_config.cudagraph_capture_sizes
                    if x <= max_num_tokens and 
                    x >= uniform_decode_query_len
                ]
                for bs in cudagraph_capture_sizes_for_decode:
                    self.add_cudagraph_key(CUDAGraphMode.FULL,
                        BatchDescriptor(num_tokens=bs, is_uniform=True))
        self.keys_initialized = True

    def dispatch(self, batch_descriptor: BatchDescriptor) -> tuple[
        CUDAGraphMode, Optional[BatchDescriptor]]:
        # if not initialized, just skip dispatching.
        if not self.keys_initialized:
            logger.warning_once("cudagraph dispatching keys are not "
                                "initialized. No cudagraph will be used.")
            return CUDAGraphMode.NONE, None
        
        assert batch_descriptor.is_uniform is not None, \
            "is_uniform is required for cudagraph dispatching."

        # check if key exists for full cudagraph
        if batch_descriptor in self.cudagraph_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.FULL, batch_descriptor
        
        # otherwise, check if key exists for more "general" piecewise cudagraph
        new_batch_descriptor = BatchDescriptor(
            num_tokens=batch_descriptor.num_tokens,
            is_uniform=None)
        if new_batch_descriptor in self.cudagraph_keys[
            CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, new_batch_descriptor
        
        # finally, just return no cudagraphs
        return CUDAGraphMode.NONE, None
