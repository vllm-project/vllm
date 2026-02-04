# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.lora.utils import get_captured_lora_counts

logger = init_logger(__name__)


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

    def __init__(self, vllm_config: VllmConfig, is_mm_encoder: bool = False):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.is_mm_encoder = is_mm_encoder
        self.max_capture_size = (
            self.compilation_config.max_cudagraph_capture_size
            if not is_mm_encoder
            else self.compilation_config.max_mm_encoder_cudagraph_capture_size
        )
        self.capture_sizes = (
            self.compilation_config.cudagraph_capture_sizes
            if not is_mm_encoder
            else self.compilation_config.mm_encoder_cudagraph_capture_sizes
        )
        self.uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False
        self.specialize_lora_count = (
            self.vllm_config.lora_config.specialize_active_lora
            if self.vllm_config.lora_config is not None
            else False
        )
        # Default cudagraph_mode to NONE until initialize_cudagraph_keys is called
        self.cudagraph_mode = CUDAGraphMode.NONE

    def _compute_bs_to_padded_graph_size(self) -> None:
        """Pre-compute the mapping from batch size to padded graph size."""
        self._bs_to_padded_graph_size: list[int] = [0] * (self.max_capture_size + 1)
        for end, start in zip(
            self.capture_sizes + [self.max_capture_size + 1],
            [0] + self.capture_sizes,
        ):
            for bs in range(start, end):
                if bs == start:
                    self._bs_to_padded_graph_size[bs] = start
                else:
                    self._bs_to_padded_graph_size[bs] = end

        # Validate that compile_sizes won't be changed by padding.
        # Only validate when cudagraphs are actually being used.
        if (
            self.compilation_config.compile_sizes
            and self.cudagraph_mode != CUDAGraphMode.NONE
        ):
            for size in self.compilation_config.compile_sizes:
                if size <= self.max_capture_size:
                    padded = self._bs_to_padded_graph_size[size]
                    if padded != size:
                        raise ValueError(
                            f"compile_sizes contains {size} which would be "
                            f"padded to {padded}. All compile_sizes must be "
                            "values that won't be changed by cudagraph padding. "
                            "Use values from cudagraph_capture_sizes."
                        )

    def _get_lora_cases(self) -> list[int]:
        """
        Returns list of has_lora values for CUDA graph capture.
        This is the single source of truth for LoRA capture cases.
        """
        lora_config = self.vllm_config.lora_config
        if lora_config is None:
            # No LoRA configured - single case with no LoRA
            return [0]

        # LoRA is enabled - capture graphs based on cudagraph_specialize_lora
        if self.compilation_config.cudagraph_specialize_lora:
            captured_counts = get_captured_lora_counts(
                lora_config.max_loras, self.specialize_lora_count
            )
            # Specialize: capture separate graphs for with and without LoRA
            return [0] + captured_counts
        else:
            # No specialization: only capture graphs with LoRA active
            return [lora_config.max_loras + 1]

    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = num_tokens_padded // uniform_decode_query_len
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ):
        # This should be called only after attention backend is initialized. So we can
        # get the correct cudagraph mode after backend support is resolved.
        self.cudagraph_mode = cudagraph_mode

        # Early exit if cudagraphs are disabled
        if cudagraph_mode == CUDAGraphMode.NONE:
            self.keys_initialized = True
            return

        self._compute_bs_to_padded_graph_size()

        # Get LoRA cases to capture
        lora_cases = self._get_lora_cases() if not self.is_mm_encoder else [0]
        self.captured_lora_counts = [
            lora_count for lora_count in lora_cases if lora_count
        ]

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            for bs, num_active_loras in product(self.capture_sizes, lora_cases):
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    self._create_padded_batch_descriptor(
                        bs, False, num_active_loras > 0, num_active_loras
                    ).relax_for_mixed_batch_cudagraphs(),
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
                for x in self.capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_decode, lora_cases
            ):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras
                    ),
                )

        self.keys_initialized = True

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        disable_full: bool = False,
        num_active_loras: int = 0,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """
        Given conditions(e.g.,batch descriptor and if using piecewise only),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).

        Args:
            num_tokens: Number of tokens in the batch.
            uniform_decode: Whether the batch is uniform decode (i.e. uniform and query
                length is uniform_decode_query_len).
            has_lora: Whether LoRA is active.
            disable_full: If True, skip FULL cudagraph checks and
                return PIECEWISE or NONE only. (can be used for features like
                cascade attention that are not supported by full cudagraphs)
            num_active_loras: Number of distinct active LoRA adapters.
        """
        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or num_tokens > self.max_capture_size
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        effective_num_active_loras = num_active_loras
        if has_lora and num_active_loras > 0:
            if self.specialize_lora_count:
                # Find the smallest captured `num_active_loras` that is >= the current
                # `num_active_loras`. This is because we only capture graphs for
                # a subset of possible `num_active_loras` values (powers of 2).
                import bisect

                idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
                if idx < len(self.captured_lora_counts):
                    effective_num_active_loras = self.captured_lora_counts[idx]
            else:
                # When not specializing, graphs are captured only with max_loras + 1,
                # so we must use max_loras + 1 for dispatch to find a matching graph.
                effective_num_active_loras = self.vllm_config.lora_config.max_loras + 1

        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, uniform_decode, has_lora, effective_num_active_loras
        )
        relaxed_batch_desc = batch_desc.relax_for_mixed_batch_cudagraphs()

        if not disable_full:
            # check if key exists for full cudagraph
            if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc

            # otherwise, check if the relaxed key exists
            if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, relaxed_batch_desc

        # also check if the relaxed key exists for more "general"
        # piecewise cudagraph
        if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, relaxed_batch_desc

        # finally, just return no cudagraphs and a trivial batch descriptor
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

    def get_capture_descs(self) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]:
        """
        Returns capture descriptors for cudagraph capturing.

        Returns:
            List of (runtime_mode, batch_descriptors) tuples, ordered PIECEWISE
            first then FULL. Batch descriptors are sorted largest-first for
            memory efficiency.
        """
        if not self.keys_initialized or self.cudagraph_mode == CUDAGraphMode.NONE:
            return []

        result = []
        # Return in order: PIECEWISE first, then FULL
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            descs = list(self.cudagraph_keys[mode])
            if descs:
                # Sort by num_tokens descending (largest first)
                descs.sort(key=lambda d: d.num_tokens, reverse=True)
                result.append((mode, descs))

        return result
