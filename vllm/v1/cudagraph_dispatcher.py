# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

from vllm.config import CompilationLevel, CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.utils import round_up

logger = init_logger(__name__)


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispach keys for multiple set of cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one
    for FULL cudagraph runtime mode. The keys are initialized depending on 
    attention support and what cudagraph mode is set in CompilationConfig. The 
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL, 
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)
    based on the input key. After dispatching (commuicate via forward context), 
    the cudagraph wrappers will trust the dispatch key to do either capturing
    or replaying (if mode matched), or pass through to the underlying runnable 
    without cudagraph (if mode no match or mode is NONE).
    """

    def __init__(self, vllm_config: VllmConfig, is_drafter: bool = False):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.cudagraph_mode = self.compilation_config.cudagraph_mode
        self.is_drafter = is_drafter

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }
        # Placeholder for capture sizes. Should be initialized in
        # self.initialize_cudagraph_keys.
        self.cudagraph_capture_sizes: list[int] = []
        self.uniform_cudagraph_capture_sizes: list[int] = []
        self.uniform_decode_query_len: int = 0

        assert not self.cudagraph_mode.requires_piecewise_compilation() or \
            (self.compilation_config.level == CompilationLevel.PIECEWISE and
             self.compilation_config.splitting_ops_contain_attention()), \
            "Compilation level should be CompilationLevel.PIECEWISE when "\
            "cudagraph_mode piecewise cudagraphs is used, "\
            f"cudagraph_mode={self.cudagraph_mode}, "\
            f"compilation_level={self.compilation_config.level}, "\
            f"splitting_ops={self.compilation_config.splitting_ops}"

        self.keys_initialized = False

    def add_cudagraph_key(self, runtime_mode: CUDAGraphMode,
                          batch_descriptor: BatchDescriptor):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], \
            f"Invalid cudagraph runtime mode: {runtime_mode}"
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode,
                                  uniform_decode_query_len: int):
        # This should be called only after attention backend is initialized.

        # Note: we create all valid keys possible for cudagraph but do not
        # guarantee all keys would be used. For example, we create keys for
        # piecewise cudagraphs when it is piecewise compilation, which is always
        # valid, but for attention backend support unified routine, we may not
        # trigger capturing/replaying the piecewise cudagraphs depending on
        # CompilationConfig.cudagraph_mode. In addition, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            for bs in self.compilation_config.cudagraph_capture_sizes:
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    BatchDescriptor(num_tokens=bs, uniform_decode=False))
            self.cudagraph_capture_sizes = \
                self.compilation_config.cudagraph_capture_sizes

        # if decode cudagraph mode is FULL, and we don't already have mixed
        # mode full cudagraphs then add them here.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL \
            and cudagraph_mode.separate_routine():
            max_num_tokens = uniform_decode_query_len * \
                self.vllm_config.scheduler_config.max_num_seqs
            # for uniform_decode_query_len==1, we use the non-uniform
            # capture sizes, this can be for main model without spec-decode or
            # for the drafter. Otherwise, we use the uniform-projected sizes.
            candidate_sizes = self.compilation_config.cudagraph_capture_sizes\
                if uniform_decode_query_len == 1 else \
                self.compilation_config.cudagraph_capture_sizes_uniform
            cudagraph_capture_sizes_for_decode = [
                x for x in candidate_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs in cudagraph_capture_sizes_for_decode:
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    BatchDescriptor(num_tokens=bs, uniform_decode=True))
            self.uniform_cudagraph_capture_sizes = \
                cudagraph_capture_sizes_for_decode

        self.uniform_decode_query_len = uniform_decode_query_len
        self.keys_initialized = True

    def get_capture_cases(self, uniform_decode: bool) -> list[int]:
        """Return capture sizes for a given whether it is uniform-decode."""
        if not uniform_decode:
            return list(self.cudagraph_capture_sizes)
        else:
            return list(self.uniform_cudagraph_capture_sizes)

    def padded_num_tokens(self, num_tokens: int,
                          uniform_decode: bool) -> tuple[int, bool]:
        """Return num_tokens after padded and whether it is cudagraph padded.
        """
        if self.uniform_decode_query_len == 1 and num_tokens <= \
            self.compilation_config.max_capture_size:
            # common situation within the range of max_capture_size for main
            # model without spec-decode or it is for a drafter.
            # we ignore whether it is uniform-decode since it is always safe
            # to pad.
            return self.vllm_config.pad_for_cudagraph(
                num_tokens, uniform_aligned=False), True

        if self.uniform_decode_query_len > 1 and uniform_decode and \
            num_tokens <= self.compilation_config.max_capture_size_uniform:
            # this is particular for uniform-decode alignment for vaildation
            # phase of spec-decode.
            return self.vllm_config.pad_for_cudagraph(
                num_tokens, uniform_aligned=True), True

        return num_tokens, False

    def plan(
        self,
        num_scheduled_tokens: int,
        num_reqs: int,
        max_query_len: int,
    ) -> tuple[CUDAGraphMode, Optional[BatchDescriptor], int]:
        """Plan cudagraph execution in a single call.

        Returns (runtime_mode, batch_descriptor, num_input_tokens_padded).
        """
        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            num_scheduled_tokens == num_reqs * max_query_len)

        # Compute padded tokens
        cudagraph_padded = False
        if self.cudagraph_mode != CUDAGraphMode.NONE:
            num_input_tokens, cudagraph_padded = self.padded_num_tokens(
                num_scheduled_tokens, uniform_decode)
        else:
            num_input_tokens = num_scheduled_tokens

        if not cudagraph_padded and not self.is_drafter:
            # Eager mode
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)

        # Build initial descriptor and dispatch
        descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                     uniform_decode=uniform_decode)
        runtime_mode, descriptor = self.dispatch(descriptor)
        return runtime_mode, descriptor, num_input_tokens

    def dispatch(
        self, batch_descriptor: BatchDescriptor
    ) -> tuple[CUDAGraphMode, Optional[BatchDescriptor]]:
        """
        Given a batch descriptor, dispatch to a cudagraph mode.
        A new batch descriptor is returned as we might dispatch a uniform batch 
        to a graph that supports a more general batch (uniform to non-uniform).
        """
        # if not initialized, just skip dispatching.
        if not self.keys_initialized:
            logger.warning_once("cudagraph dispatching keys are not "
                                "initialized. No cudagraph will be used.")
            return CUDAGraphMode.NONE, None

        # check if key exists for full cudagraph
        if batch_descriptor in self.cudagraph_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.FULL, batch_descriptor

        # otherwise, check if non-uniform key exists
        non_uniform_key = batch_descriptor.non_uniform
        if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.FULL, non_uniform_key

        # also check if non-uniform key exists for more "general"
        # piecewise cudagraph
        if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, non_uniform_key

        # finally, just return no cudagraphs
        return CUDAGraphMode.NONE, None
