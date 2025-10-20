# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispatch keys for multiple set of
    cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one
    for FULL cudagraph runtime mode. The keys are initialized depending on
    attention support and what cudagraph mode is set in CompilationConfig. The
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL,
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)
    based on the input key. After dispatching (communicated via forward
    context), the cudagraph wrappers will trust the dispatch key to either
    capture or replay (if the mode matches), or pass through to the underlying
    runnable without cudagraph (if the mode does not match or mode is NONE).
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

        not_use_piecewise_compilation = (
            not self.cudagraph_mode.requires_piecewise_compilation()
        )

        assert (
            not_use_piecewise_compilation
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int
    ):
        # This should be called only after attention backend is initialized.

        # LoRA activation cases to specialize the cuda graphs on
        if self.vllm_config.lora_config:
            if self.compilation_config.cudagraph_specialize_lora:
                lora_cases = [True, False]
            else:
                lora_cases = [True]
        else:
            lora_cases = [False]

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            for bs, has_lora in product(
                self.compilation_config.cudagraph_capture_sizes, lora_cases
            ):
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    BatchDescriptor(
                        num_tokens=bs, uniform_decode=False, has_lora=has_lora
                    ),
                )

        # if decode cudagraph mode is FULL, and we don't already have mixed
        # mode full cudagraphs then add them here.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, has_lora in product(cudagraph_capture_sizes_for_decode, lora_cases):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    BatchDescriptor(
                        num_tokens=bs, uniform_decode=True, has_lora=has_lora
                    ),
                )
        self.keys_initialized = True

    def dispatch(
        self, batch_descriptor: BatchDescriptor, use_cascade_attn: bool = False
    ) -> tuple[CUDAGraphMode, BatchDescriptor | None]:
        """
        Given conditions(e.g.,batch descriptor and if using cascade attention),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).
        """
        # if not initialized, just skip dispatching.
        if not self.keys_initialized:
            return CUDAGraphMode.NONE, None

        non_uniform_key = batch_descriptor.non_uniform
        # if a batch use cascade attention, bypass checking full cudagraphs
        if not use_cascade_attn:
            # check if key exists for full cudagraph
            if batch_descriptor in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_descriptor

            # otherwise, check if non-uniform key exists
            if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, non_uniform_key

        # also check if non-uniform key exists for more "general"
        # piecewise cudagraph
        if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, non_uniform_key

        # finally, just return no cudagraphs
        return CUDAGraphMode.NONE, None
