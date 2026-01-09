# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU-accelerated N-gram proposer using fully async PyTorch tensor operations.

This version uses a fully vectorized approach with unfold and argmax for
finding the first match across all sequences in parallel.
"""

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
)
from vllm.forward_context import set_forward_context
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import InputBatch


@support_torch_compile()
class NgramGPUKernel(nn.Module):
    """
    GPU-accelerated N-gram proposer using fully async tensor operations.

    PERFORMANCE OPTIMIZATION WITH TORCH.COMPILE:

    1. Tensor Allocation Strategy:
       - DO: Allocate tensors inside forward() - torch.compile will optimize this
       - DON'T: Pre-allocate buffers as class attributes - breaks compilation
       - WHY: torch.compile fuses allocations into the compiled graph for efficiency

    2. Dynamic Shapes:
       - Batch size (dim 0) is automatically marked as dynamic in support_torch_compile
       - torch.compile generates specialized kernels for different shapes

    3. Graph Compilation:
       - Note: fullgraph=True does NOT mean the forward pass runs as a single
         CUDA Graph. nsys profiling shows ~5 separate Triton kernel launches.
         torch.compile with fullgraph=True ensures complete Dynamo tracing
         without fallbacks, but Inductor still generates multiple kernels.
       - CUDA Graph capture is disabled (cudagraph_mode=NONE) for this module
         because: (1) the kernel launch overhead is minimal compared to the
         actual computation, and (2) the n-gram matching workload is simple
         enough that CUDA Graph capture would be overkill with little benefit.
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
        token_ids: torch.Tensor,
        seq_lengths: torch.Tensor,
        min_ngram_len: int,
        max_ngram_len: int,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        """
        Find n-gram matches and extract tokens following the match.

        For each sequence, searches for the earliest occurrence of the trailing
        n-gram (the "suffix") earlier in the sequence. When found, extracts
        the tokens that followed that earlier occurrence as draft predictions.
        Tries multiple n-gram lengths and selects the longest match.

        Terminology:
            - suffix: The trailing n-gram at the end of each sequence that we
                      search for earlier in the history
            - search_window: A sliding view over the sequence used to find
                             matches of the suffix
            - match_position: The starting index where the suffix was found
            - draft_tokens: Tokens extracted after the match position

        Args:
            token_ids: Token IDs for each sequence
                Shape: [batch_size, max_seq_len]
            seq_lengths: Actual length of each sequence (excluding padding)
                Shape: [batch_size]
            min_ngram_len: Minimum n-gram size to search for (e.g., 2)
            max_ngram_len: Maximum n-gram size to search for (e.g., 5)
            num_draft_tokens: Number of tokens to extract after match (k)

        Returns:
            Draft token predictions, -1 for invalid/no-match positions
                Shape: [batch_size, num_draft_tokens]
        """
        batch_size = token_ids.shape[0]
        max_seq_len = token_ids.shape[1]
        device = token_ids.device
        num_ngram_sizes = max_ngram_len - min_ngram_len + 1

        # ngram_lengths: All n-gram sizes we'll try
        # Shape: [num_ngram_sizes]
        ngram_lengths = torch.arange(min_ngram_len, max_ngram_len + 1, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        # first_match_positions: Stores the earliest match position for each
        # (sequence, ngram_length) pair. -1 means no match found.
        # Shape: [batch_size, num_ngram_sizes]
        first_match_positions = torch.full(
            (batch_size, num_ngram_sizes), -1, dtype=torch.long, device=device
        )

        for i, ngram_len in enumerate(range(min_ngram_len, max_ngram_len + 1)):
            # Create sliding windows of size ngram_len over each sequence.
            # Window w contains tokens[w : w + ngram_len].
            # Shape: [batch_size, num_windows, ngram_len]
            #   where num_windows = max_seq_len - ngram_len + 1
            # Note: unfold returns a view (O(1)), so calling it per iteration
            # is efficient and avoids complex prefix handling for shorter n-grams.
            search_windows = token_ids.unfold(1, ngram_len, 1)
            num_windows = search_windows.shape[1]

            # Extract the trailing suffix (last ngram_len tokens) from each seq
            # suffix_starts[b] = position where suffix begins in sequence b
            # Shape: suffix_starts [batch_size], suffix [batch_size, ngram_len]
            suffix_starts = seq_lengths - ngram_len
            suffix_indices = suffix_starts.unsqueeze(1) + torch.arange(
                ngram_len, device=device
            )
            suffix = torch.gather(token_ids, 1, suffix_indices.clamp(min=0))

            # Check which windows match the suffix
            # matches[b, w] = True if window w in sequence b matches suffix[b]
            # Shape: [batch_size, num_windows]
            matches = (search_windows == suffix.unsqueeze(1)).all(dim=-1)

            # Validity check: the match position must leave room for at least
            # one token after the suffix to extract as a draft token.
            # Window positions are simply 0, 1, 2, ... num_windows-1
            # max_valid_suffix_start[b] = last valid starting position in seq b
            max_valid_suffix_start = seq_lengths - ngram_len - 1
            window_positions = torch.arange(num_windows, device=device)
            valid_mask = window_positions <= max_valid_suffix_start.unsqueeze(1)
            final_matches = matches & valid_mask

            # Find first (earliest) match position for each sequence
            # (argmax returns 0 if no match, so we verify with has_match)
            first_match_idx = torch.argmax(final_matches.int(), dim=1)
            has_match = final_matches[batch_indices, first_match_idx]

            # Store valid match positions (window index = actual position)
            first_match_positions[:, i] = torch.where(has_match, first_match_idx, -1)

        # Select the longest n-gram that found a valid match
        # (search from back to front to prioritize longer n-grams)
        # Shape: best_ngram_idx [batch_size]
        best_ngram_idx = (first_match_positions >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_ngram_idx = num_ngram_sizes - 1 - best_ngram_idx  # Flip back

        # Get the match position for the best n-gram
        # Shape: best_match_pos [batch_size]
        best_match_pos = first_match_positions[batch_indices, best_ngram_idx]

        # Handle matched cases - completely avoid data-dependent branching
        has_any_match = best_match_pos >= 0

        # best_ngram_lengths[b] = length of the best matching n-gram for seq b
        # Shape: [batch_size]
        best_ngram_lengths = ngram_lengths[best_ngram_idx]

        # Calculate where to start extracting draft tokens
        # draft_start[b] = position right after the matched suffix
        # Shape: draft_start [batch_size]
        draft_start = torch.where(
            has_any_match,
            best_match_pos + best_ngram_lengths,
            torch.zeros_like(best_match_pos),
        )
        tokens_available = seq_lengths - draft_start

        # Create gather indices for extracting draft tokens
        # Shape: draft_indices [batch_size, num_draft_tokens]
        draft_indices = draft_start.unsqueeze(1) + torch.arange(
            num_draft_tokens, device=device
        )
        draft_indices = draft_indices.clamp(min=0, max=max_seq_len - 1)

        # Extract draft tokens (always execute gather, even for invalid positions)
        # Shape: draft_tokens [batch_size, num_draft_tokens]
        draft_tokens = torch.gather(token_ids, 1, draft_indices)

        # Mask out positions beyond what's available in the sequence
        position_indices = torch.arange(num_draft_tokens, device=device).unsqueeze(0)
        valid_positions = position_indices < tokens_available.unsqueeze(1)

        draft_tokens = torch.where(
            valid_positions,
            draft_tokens,
            torch.full_like(draft_tokens, -1),
        )

        # Mask out all positions if no match was found
        draft_tokens = torch.where(
            has_any_match.unsqueeze(1),
            draft_tokens,
            torch.full_like(draft_tokens, -1),
        )

        return draft_tokens

    def forward(
        self,
        num_tokens_no_spec: torch.Tensor,
        token_ids_gpu: torch.Tensor,
        combined_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for N-gram proposal using GPU tensor operations.

        Args:
            num_tokens_no_spec: Number of tokens for each sequence [batch_size]
            token_ids_gpu: Token IDs [batch_size, max_len]
            combined_mask: Whether each sequence is valid for spec decode [batch_size]

        Returns:
            draft_tokens: [batch_size, k] on GPU
            is_empty_draft_tokens: [batch_size] bool on GPU
        """

        device = token_ids_gpu.device

        # Infer batch_size from the input tensor shape to maintain dynamic shape
        actual_batch_size = token_ids_gpu.shape[0]

        # Initialize output tensor - torch.compile will optimize this allocation
        # NOTE(patchy): Do NOT pre-allocate this as a buffer
        #               it would break torch.compile
        draft_tokens = torch.full(
            (actual_batch_size, self.k), -1, dtype=torch.int32, device=device
        )

        results = self._find_first_and_extract_all_n_parallel(
            token_ids_gpu,
            num_tokens_no_spec,
            min_ngram_len=self.min_n,
            max_ngram_len=self.max_n,
            num_draft_tokens=self.k,
        )

        draft_tokens = torch.where(combined_mask.unsqueeze(1), results, -1)

        is_empty_draft_tokens = (draft_tokens == -1).all(dim=1)

        return draft_tokens, is_empty_draft_tokens

    def load_model(self, *args, **kwargs):
        """No model to load for N-gram proposer."""
        pass


class NgramProposerGPU:
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        compilation_config = CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
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
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        model_config = vllm_config.model_config
        speculative_config = vllm_config.speculative_config
        scheduler_config = vllm_config.scheduler_config

        self.vllm_config = VllmConfig(
            compilation_config=compilation_config,
            model_config=model_config,
            speculative_config=speculative_config,
            scheduler_config=scheduler_config,
        )

        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.device = device

        self.kernel = NgramGPUKernel(
            vllm_config=self.vllm_config, prefix="ngram_gpu_kernel", device=device
        )
        self.device = device
        self.kernel.to(device)
        self.kernel.eval()

        self._dummy_run()

    def _dummy_run(self):
        token_ids, num_tokens, sampled_flags, valid_mask = self._generate_dummy_data(
            batch_size=self.max_num_seqs,
            max_seq_len=self.max_model_len,
            pattern_len=self.k,
            device=self.device,
        )

        combined_mask = sampled_flags & valid_mask & (num_tokens >= self.min_n)

        for _ in range(3):
            with set_forward_context(None, self.vllm_config):
                _, _ = self.kernel(num_tokens, token_ids, combined_mask)

    def _generate_dummy_data(
        self,
        batch_size: int,
        max_seq_len: int,
        pattern_len: int,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random test data with n-gram repetitions.

        Args:
            batch_size: Number of sequences in the batch
            max_seq_len: Maximum sequence length
            pattern_len: Length of patterns to inject for matching
            device: Device to place tensors on

        Returns:
            token_ids: [batch_size, max_seq_len] tensor
            num_tokens: [batch_size] tensor
            sampled_flags: [batch_size] bool tensor
            valid_mask: [batch_size] bool tensor
        """
        # Generate random token IDs
        token_ids = torch.zeros(
            batch_size,
            max_seq_len,
            dtype=torch.int32,
            device=device,
        )

        # Generate random sequence lengths
        num_tokens = torch.randint(
            pattern_len, max_seq_len, (batch_size,), dtype=torch.int32, device=device
        )

        # All sequences have sampled tokens and are valid
        sampled_flags = torch.ones(batch_size, dtype=torch.bool, device=device)
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        return token_ids, num_tokens, sampled_flags, valid_mask

    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size]
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len]
        valid_sampled_token_ids_gpu: torch.Tensor,  # [batch_size, num_spec_tokens + 1]
        valid_sampled_tokens_count: torch.Tensor,  # [batch_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propose draft tokens using GPU-accelerated n-gram matching.

        This method:
        1. Scatters newly sampled tokens into token_ids_gpu
        2. Updates num_tokens_no_spec in-place
        3. Computes validity masks for speculative decoding
        4. Runs n-gram matching kernel to propose draft tokens

        Args:
            num_tokens_no_spec: Number of tokens per sequence (modified in-place)
            token_ids_gpu: Token IDs tensor (modified in-place with new tokens)
            valid_sampled_token_ids_gpu: Newly sampled tokens to scatter
            valid_sampled_tokens_count: Count of valid tokens per sequence

        Returns:
            draft_tokens: Proposed draft token IDs [batch_size, k]
            is_empty_draft_tokens: Boolean mask for empty proposals [batch_size]
        """
        assert token_ids_gpu.device == self.device
        assert num_tokens_no_spec.device == self.device

        batch_size = num_tokens_no_spec.shape[0]
        max_new_tokens = valid_sampled_token_ids_gpu.shape[1]  # num_spec_tokens + 1

        # Scatter newly sampled tokens into token_ids_gpu
        offsets = torch.arange(max_new_tokens, device=self.device)
        write_positions = num_tokens_no_spec.unsqueeze(1) + offsets.unsqueeze(0)
        valid_write_mask = offsets.unsqueeze(0) < valid_sampled_tokens_count.unsqueeze(
            1
        )
        scatter_mask = valid_write_mask & (valid_sampled_token_ids_gpu != -1)

        write_positions_long = write_positions.long()
        existing_values = token_ids_gpu.gather(1, write_positions_long)

        tokens_cast = valid_sampled_token_ids_gpu.to(token_ids_gpu.dtype)
        tokens_to_scatter = torch.where(
            scatter_mask,
            tokens_cast,
            existing_values,
        )
        token_ids_gpu.scatter_(1, write_positions_long, tokens_to_scatter)

        # Update num_tokens_no_spec in-place
        num_tokens_no_spec += valid_sampled_tokens_count

        # Compute validity masks
        sampled_flags = valid_sampled_tokens_count > 0
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        with set_forward_context(None, self.vllm_config):
            combined_mask = (
                sampled_flags & valid_mask & (num_tokens_no_spec >= self.min_n)
            )

            with record_function_or_nullcontext("ngram_proposer_gpu: kernel"):
                draft_tokens, is_empty_draft_tokens = self.kernel(
                    num_tokens_no_spec,
                    token_ids_gpu,
                    combined_mask,
                )

            return draft_tokens, is_empty_draft_tokens

    def update_token_ids_ngram(
        self,
        sampled_token_ids: torch.Tensor,
        gpu_input_batch: InputBatch,
        token_ids_gpu: torch.Tensor,
        num_tokens_no_spec: torch.Tensor,
        discard_request_mask: torch.Tensor,
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
        num_reqs = gpu_input_batch.num_reqs

        # Extract backup_next_token_ids from token_ids_gpu using vectorized gather
        # For each request i, get token_ids_gpu[i, num_tokens_no_spec[i] - 1]
        # This is the last valid token before speculative tokens
        backup_indices = (num_tokens_no_spec[:num_reqs] - 1).clamp(min=0).long()
        backup_next_token_ids = torch.gather(
            token_ids_gpu[:num_reqs], dim=1, index=backup_indices.unsqueeze(1)
        ).squeeze(1)

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        # Use discard_request_mask to invalidate sampled tokens for discarded
        # requests (e.g., chunked prefill partial requests that should not be
        # sampled). Expand mask to match [num_reqs, num_tokens] shape.
        # Use masked_fill_ to avoid creating new tensors (no CPU-GPU sync).
        discard_mask_expanded = discard_request_mask[:num_reqs].unsqueeze(1)
        valid_sampled_token_ids_gpu.masked_fill_(discard_mask_expanded, -1)

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

        # Use last token if valid, vectorized backup from token_ids_gpu if not
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            backup_next_token_ids,
        )

        return next_token_ids, valid_sampled_tokens_count, valid_sampled_token_ids_gpu

    def load_model(self, *args, **kwargs):
        self.kernel.load_model(*args, **kwargs)
