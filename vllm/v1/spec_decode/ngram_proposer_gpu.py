# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU-accelerated N-gram proposer using fully async PyTorch tensor operations.

This version uses a fully vectorized approach with unfold and argmax for
finding the first match across all sequences in parallel.
"""

import numpy as np
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    VllmConfig,
)
from vllm.forward_context import set_forward_context
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


@support_torch_compile(
    dynamic_arg_dims={
        "num_tokens_no_spec": 0,
        "token_ids_gpu": [0, 1],
        "combined_mask": 0,
    }
)
class NgramGPUKernel(nn.Module):
    """
    GPU-accelerated N-gram proposer using fully async tensor operations.

    Interface: All inputs are GPU tensors (no lists, no numpy arrays)

    PERFORMANCE OPTIMIZATION WITH TORCH.COMPILE:

    1. Tensor Allocation Strategy:
       - DO: Allocate tensors inside forward() - torch.compile will optimize this
       - DON'T: Pre-allocate buffers as class attributes - breaks compilation
       - WHY: torch.compile fuses allocations into the compiled graph for efficiency

    2. Dynamic Shapes:
       - Batch size (dim 0) and sequence length (dim 1) are marked as dynamic
       - torch.compile generates specialized kernels for different shapes
       - The first call with a new shape will trigger recompilation (cached)

    3. Graph Compilation:
       - Uses fullgraph=True mode for maximum optimization
       - All operations are tensor-based (no Python loops or conditionals)
       - The entire forward pass is compiled into a single CUDA graph

    4. Memory Efficiency:
       - torch.compile's memory planning optimizes temporary allocations
       - Fusion of operations reduces memory bandwidth requirements
       - No manual memory management needed - compiler handles it
    """

    def __init__(
        self, vllm_config: VllmConfig, prefix: str = "", device: torch.device = "cuda"
    ):
        super().__init__()

        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.device = device

    def _find_first_and_extract_all_n_parallel(
        self,
        data: torch.Tensor,
        seq_lengths: torch.Tensor,
        min_pattern_len: int,
        max_pattern_len: int,
        result_len: int,
    ) -> torch.Tensor:
        """
        Process all pattern lengths in parallel, selecting the longest match.
        Completely free of data-dependent control flow, suitable for
        torch.compile optimization.
        """
        batch_size = data.shape[0]
        device = data.device
        max_seq_len = data.shape[1]
        num_patterns = max_pattern_len - min_pattern_len + 1

        all_windows = data.unfold(1, max_pattern_len, 1)  # [B, num_windows, max_n]
        num_windows = all_windows.shape[1]
        window_starts = torch.arange(num_windows, device=device)
        pattern_lengths = torch.arange(
            min_pattern_len, max_pattern_len + 1, device=device
        )
        batch_indices = torch.arange(batch_size, device=device)

        all_first_matches = torch.full(
            (batch_size, num_patterns), -1, dtype=torch.long, device=device
        )

        for i, pattern_len in enumerate(range(min_pattern_len, max_pattern_len + 1)):
            offset = max_pattern_len - pattern_len

            # Extract pattern from the end of each sequence
            pattern_starts = seq_lengths - pattern_len
            pattern_indices = pattern_starts.unsqueeze(1) + torch.arange(
                pattern_len, device=device
            )
            patterns = torch.gather(data, 1, pattern_indices.clamp(min=0))

            # Slice windows and perform matching
            current_windows = all_windows[..., offset:]
            matches = (current_windows == patterns.unsqueeze(1)).all(dim=-1)

            # Validity check: ensure enough space for result extraction
            max_valid_pattern_start = seq_lengths - pattern_len - result_len
            pattern_start_positions = window_starts + offset
            valid_mask = pattern_start_positions <= max_valid_pattern_start.unsqueeze(1)
            final_matches = matches & valid_mask

            # Handle prefix positions that fall before the available windows
            prefix_positions = torch.arange(offset, device=device)
            gather_indices = prefix_positions.view(1, -1, 1) + torch.arange(
                pattern_len, device=device
            ).view(1, 1, -1)
            gather_indices = gather_indices.clamp(min=0, max=max_seq_len - 1)
            expanded_indices = gather_indices.expand(batch_size, -1, -1)
            prefix_tokens = torch.gather(
                data.unsqueeze(1).expand(-1, offset, -1),
                2,
                expanded_indices,
            )
            prefix_matches = (
                prefix_tokens == patterns.unsqueeze(1).expand(-1, offset, -1)
            ).all(dim=-1)
            prefix_valid_mask = prefix_positions <= max_valid_pattern_start.unsqueeze(1)
            prefix_final_matches = prefix_matches & prefix_valid_mask

            combined_matches = torch.cat([prefix_final_matches, final_matches], dim=1)
            start_positions = torch.cat(
                [prefix_positions, pattern_start_positions], dim=0
            )

            # Find first match
            # (if no match, argmax returns 0, but we verify with has_match)
            first_indices = torch.argmax(combined_matches.int(), dim=1)
            has_match = combined_matches[batch_indices, first_indices]
            match_positions = start_positions[first_indices]

            # Store valid match positions
            all_first_matches[:, i] = torch.where(has_match, match_positions, -1)

        # Select the longest valid match,
        # from back to front, prioritizing longer patterns
        best_pattern_idx = (all_first_matches >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_pattern_idx = num_patterns - 1 - best_pattern_idx  # Flip back

        # Extract corresponding results
        best_match_pos = all_first_matches[batch_indices, best_pattern_idx]

        # Handle matched cases - completely avoid data-dependent branching
        has_any_match = best_match_pos >= 0

        best_pattern_lengths = pattern_lengths[best_pattern_idx]

        # Calculate result start positions, invalid positions will be
        # clamped to valid range. We now track true start positions, so the
        # result starts right after the matched n-gram
        result_starts = torch.where(
            has_any_match,
            best_match_pos + best_pattern_lengths,
            torch.zeros_like(best_match_pos),
        )

        # Create gather indices
        result_indices = result_starts.unsqueeze(1) + torch.arange(
            result_len, device=device
        )
        # Ensure indices are within valid range
        result_indices = result_indices.clamp(min=0, max=max_seq_len - 1)

        # Always execute gather (even for invalid data)
        extracted_sequences = torch.gather(data, 1, result_indices)

        # Use where to zero out invalid results
        results = torch.where(
            has_any_match.unsqueeze(1),
            extracted_sequences,
            torch.zeros_like(extracted_sequences),
        )

        return results

    def forward(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size] on GPU
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len] on GPU
        combined_mask: torch.Tensor,  # [batch_size] bool on GPU
    ) -> torch.Tensor:
        """
        Forward pass for N-gram proposal using GPU tensor operations.

        This is the core computation method that will be compiled by torch.compile
        via the @support_torch_compile decorator.

        Args:
            num_tokens_no_spec: Number of tokens for each sequence [batch_size]
            token_ids_gpu: Token IDs [batch_size, max_len]
            combined_mask: Whether each sequence is valid for spec decode [batch_size]
            batch_size: Deprecated parameter, will be inferred from tensor shape

        Returns:
            draft_tokens: [batch_size, k] on GPU
        """

        device = token_ids_gpu.device

        # Infer batch_size from the input tensor shape to maintain dynamic shape
        actual_batch_size = token_ids_gpu.shape[0]

        # Initialize output tensor - torch.compile will optimize this allocation
        # NOTE(patchy): Do NOT pre-allocate this as a buffer
        #               it would break torch.compile
        draft_tokens = torch.zeros(
            (actual_batch_size, self.k), dtype=torch.int32, device=device
        )

        results = self._find_first_and_extract_all_n_parallel(
            token_ids_gpu,
            num_tokens_no_spec,
            min_pattern_len=self.min_n,
            max_pattern_len=self.max_n,
            result_len=self.k,
        )

        # Apply combined mask to results. Expand mask explicitly to avoid
        # relying on broadcasting behavior that can confuse torch.compile.
        mask = combined_mask.unsqueeze(1).expand(-1, self.k)
        draft_tokens = torch.where(mask, results, draft_tokens)

        return draft_tokens

    def load_model(self, *args, **kwargs):
        """No model to load for N-gram proposer."""
        pass


class NgramProposerGPU:
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        compilation_config = CompilationConfig(
            level=3,
            custom_ops=["none"],
            splitting_ops=[],
            compile_sizes=[],
            inductor_compile_config={
                "enable_auto_functionalized_v2": False,
                "max_autotune": True,
                "aggressive_fusion": True,
                "triton.autotune_pointwise": True,
                "coordinate_descent_tuning": True,
                "use_mixed_mm": False,
            },
            use_cudagraph=False,
        )

        self.vllm_config = VllmConfig(compilation_config=compilation_config)

        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.device = device

        self.kernel = NgramGPUKernel(
            vllm_config=vllm_config, prefix="ngram_gpu_kernel", device=device
        )
        self.device = device
        self.kernel.to(device)
        self.kernel.eval()
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        self._dummy_run()

    def _dummy_run(self):
        token_ids, num_tokens, sampled_flags, valid_mask = self._generate_dummy_data(
            batch_size=self.max_num_seqs,
            max_seq_len=min(self.max_model_len, 1024),
            vocab_size=self.vocab_size,
            pattern_len=self.k,
            repetition_rate=0.5,
            device=self.device,
        )

        combined_mask = (
            sampled_flags
            & valid_mask
            & (num_tokens < self.max_model_len)
            & (num_tokens >= self.min_n)
        )

        for _ in range(3):
            with set_forward_context(None, self.vllm_config):
                _ = self.kernel(num_tokens, token_ids, combined_mask)

    def _generate_dummy_data(
        self,
        batch_size: int,
        max_seq_len: int,
        vocab_size: int = 152064,
        pattern_len: int = 3,
        repetition_rate: float = 0.5,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random test data with n-gram repetitions.

        Args:
            batch_size: Number of sequences in the batch
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size for random token generation
            pattern_len: Length of patterns to inject for matching
            repetition_rate: Rate of n-gram repetitions to inject
            device: Device to place tensors on

        Returns:
            token_ids: [batch_size, max_seq_len] tensor
            num_tokens: [batch_size] tensor
            sampled_flags: [batch_size] bool tensor
            valid_mask: [batch_size] bool tensor
        """
        # Generate random token IDs
        token_ids = torch.randint(
            0, vocab_size, (batch_size, max_seq_len), dtype=torch.int32, device=device
        )

        # Generate random sequence lengths
        min_len = max(pattern_len * 2 + 3, max_seq_len // 2)
        num_tokens = torch.randint(
            min_len, max_seq_len, (batch_size,), dtype=torch.int32, device=device
        )

        # Inject n-gram repetitions using the tail pattern of each sequence
        for i in range(batch_size):
            seq_len = num_tokens[i].item()
            if seq_len > pattern_len * 2:
                # Pattern is the last pattern_len tokens of the valid sequence
                src_pos = seq_len - pattern_len
                num_reps = int(seq_len * repetition_rate / pattern_len)
                for _ in range(num_reps):
                    # Place the copied tail pattern somewhere before the tail
                    tgt_pos = torch.randint(0, seq_len - pattern_len, (1,)).item()
                    if tgt_pos == src_pos:
                        continue

                    token_ids[i, tgt_pos : tgt_pos + pattern_len] = token_ids[
                        i, src_pos : src_pos + pattern_len
                    ].clone()

        # All sequences have sampled tokens and are valid
        sampled_flags = torch.ones(batch_size, dtype=torch.bool, device=device)
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        return token_ids, num_tokens, sampled_flags, valid_mask

    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size] on GPU
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len] on GPU
        sampled_flags: torch.Tensor,  # [batch_size] bool on GPU
        valid_mask: torch.Tensor,  # [batch_size] bool on GPU
    ) -> torch.Tensor:
        assert token_ids_gpu.device == self.device
        assert num_tokens_no_spec.device == self.device
        assert sampled_flags.device == self.device
        assert valid_mask.device == self.device

        with set_forward_context(None, self.vllm_config):
            combined_mask = (
                sampled_flags
                & valid_mask
                & (num_tokens_no_spec < self.max_model_len)
                & (num_tokens_no_spec >= self.min_n)
            )

            return self.kernel(
                num_tokens_no_spec,
                token_ids_gpu,
                combined_mask,
            )

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[np.ndarray],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids.shape[0] > 0:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        return torch.tensor(next_token_ids, dtype=torch.int32, device=self.device)

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_indices: torch.Tensor,
        num_discarded_requests: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead.
        It also accounts for the rejected tokens in `sampled_token_ids`.
        This function must use device functions to operate on the inputs, and
        should not introduce any blocking CPU-GPU synchronization.
        """
        # TODO(Ben): Combine this into a custom fused kernel
        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        # Batch convert seq_lens to avoid multiple .item() calls
        seq_lens_list = common_attn_metadata.seq_lens_cpu[:num_reqs].tolist()

        # Now use the pre-converted list to avoid .item() calls in the loop
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i])
                for i in range(num_reqs)
            ]
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        # Mask out the sampled tokens indices that should not be sampled.
        discard_sampled_tokens_req_indices = discard_request_indices[
            :num_discarded_requests
        ]

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu.index_fill_(
            0, discard_sampled_tokens_req_indices, -1
        )

        # Generate a mask for all valid tokens within those requests
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
        )

        # Count the number of valid tokens in each request
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Get the rightmost valid index per row
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Get last valid token from each row
        # (assume undefined state where there is no valid token)
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
        ).squeeze(1)

        # Use last token if valid, pre-computed backup if not
        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )

        return next_token_ids, valid_sampled_tokens_count, valid_sampled_token_ids_gpu

    def load_model(self, *args, **kwargs):
        self.kernel.load_model(*args, **kwargs)
