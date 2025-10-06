# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import ceil
from typing import Optional

from typing_extensions import TypeAlias

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger

logger = init_logger(__name__)
CUDAGraphKey: TypeAlias = tuple[int, bool]


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
        self.cudagraph_keys: dict[
            CUDAGraphMode, dict[CUDAGraphKey, BatchDescriptor]
        ] = {
            CUDAGraphMode.PIECEWISE: {},
            CUDAGraphMode.FULL: {},
        }

        not_use_piecewise_compilation = (
            not self.cudagraph_mode.requires_piecewise_compilation()
        )

        assert (
            not_use_piecewise_compilation
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation level should be CompilationLevel.PIECEWISE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.cudagraph_mode}, "
            f"compilation_level={self.compilation_config.level}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        key = (batch_descriptor.num_tokens, batch_descriptor.uniform_decode)
        self.cudagraph_keys[runtime_mode][key] = batch_descriptor

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int
    ):
        # This should be called only after attention backend is initialized.

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        # Add mixed mode keys with proper num_reqs calculation
        if (mixed_mode := cudagraph_mode.mixed_mode()) in (
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        ):
            for bs in self.compilation_config.cudagraph_capture_sizes:
                num_reqs = (
                    self.calculate_num_reqs_for_tokens(
                        bs, uniform_decode_query_len, False
                    )
                    if mixed_mode == CUDAGraphMode.FULL
                    else None
                )
                self.add_cudagraph_key(
                    mixed_mode,
                    BatchDescriptor(
                        num_tokens=bs, uniform_decode=False, num_reqs=num_reqs
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
            for bs in cudagraph_capture_sizes_for_decode:
                num_reqs = self.calculate_num_reqs_for_tokens(
                    bs, uniform_decode_query_len, True
                )
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    BatchDescriptor(
                        num_tokens=bs, uniform_decode=True, num_reqs=num_reqs
                    ),
                )

        self.keys_initialized = True

    def calculate_num_reqs_for_tokens(
        self, num_tokens: int, uniform_decode_query_len: int, uniform_decode: bool
    ) -> int:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs

        if uniform_decode:
            num_reqs = ceil(num_tokens / uniform_decode_query_len)
            return min(num_reqs, max_num_seqs)
        else:
            return min(num_tokens, max_num_seqs)

    def _is_compatible(
        self, batch_descriptor: BatchDescriptor, candidate: BatchDescriptor
    ) -> bool:
        """Check if candidate cudagraph can handle the batch request."""
        if candidate.num_reqs is None:
            return True
        assert batch_descriptor.num_reqs is not None
        return candidate.num_reqs >= batch_descriptor.num_reqs

    def dispatch(
        self, batch_descriptor: BatchDescriptor, use_cascade_attn: bool = False
    ) -> tuple[CUDAGraphMode, Optional[BatchDescriptor]]:
        """
        Given conditions(e.g.,batch descriptor and if using cascade attention),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).
        """
        # if not initialized, just skip dispatching.
        if not self.keys_initialized:
            return CUDAGraphMode.NONE, None

        num_tokens, uniform_decode = (
            batch_descriptor.num_tokens,
            batch_descriptor.uniform_decode,
        )

        candidates = [(CUDAGraphMode.FULL, (num_tokens, uniform_decode))]
        if uniform_decode:
            candidates.append((CUDAGraphMode.FULL, (num_tokens, False)))
        candidates.append((CUDAGraphMode.PIECEWISE, (num_tokens, False)))

        for mode, key in candidates:
            candidate = self.cudagraph_keys[mode].get(key)
            if candidate and self._is_compatible(batch_descriptor, candidate):
                return mode, candidate

        return CUDAGraphMode.NONE, None
