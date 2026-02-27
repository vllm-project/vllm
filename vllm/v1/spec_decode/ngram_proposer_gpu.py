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
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


@support_torch_compile()
class NgramGPUKernel(nn.Module):
    """GPU-accelerated N-gram proposer using fully async tensor operations."""

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
        Find suffix n-gram matches and extract following tokens.
        Searches for the earliest prior occurrence of the trailing n-gram,
        tries multiple lengths, and picks the longest valid match.

        Args:
            token_ids: Token IDs for each sequence
            seq_lengths: Actual length of each sequence (excluding padding)
            min_ngram_len: Minimum n-gram size to search for (e.g., 2)
            max_ngram_len: Maximum n-gram size to search for (e.g., 5)
            num_draft_tokens: Number of tokens to extract after match (k)

        Returns:
            Draft token predictions; -1 means invalid/no match.
        """
        batch_size = token_ids.shape[0]
        max_seq_len = token_ids.shape[1]
        device = token_ids.device
        num_ngram_sizes = max_ngram_len - min_ngram_len + 1

        # All n-gram sizes to try.
        ngram_lengths = torch.arange(min_ngram_len, max_ngram_len + 1, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        # Earliest match per (sequence, ngram_len); -1 means no match.
        first_match_positions = torch.full(
            (batch_size, num_ngram_sizes), -1, dtype=torch.long, device=device
        )

        for i, ngram_len in enumerate(range(min_ngram_len, max_ngram_len + 1)):
            # Sliding windows of size ngram_len; unfold is O(1) view.
            search_windows = token_ids.unfold(1, ngram_len, 1)
            num_windows = search_windows.shape[1]

            # Trailing suffix (last ngram_len tokens) for each sequence.
            suffix_starts = seq_lengths - ngram_len
            suffix_indices = suffix_starts.unsqueeze(1) + torch.arange(
                ngram_len, device=device
            )
            suffix = torch.gather(token_ids, 1, suffix_indices.clamp(min=0))

            # Window matches for each sequence.
            matches = (search_windows == suffix.unsqueeze(1)).all(dim=-1)

            # Match must leave room for at least one draft token.
            max_valid_suffix_start = seq_lengths - ngram_len - 1
            window_positions = torch.arange(num_windows, device=device)
            valid_mask = window_positions <= max_valid_suffix_start.unsqueeze(1)
            final_matches = matches & valid_mask

            # Find earliest match (argmax=0 when empty; verify with has_match).
            first_match_idx = torch.argmax(final_matches.int(), dim=1)
            has_match = final_matches[batch_indices, first_match_idx]

            # Store valid match positions (window index = position).
            first_match_positions[:, i] = torch.where(has_match, first_match_idx, -1)

        # Select the longest n-gram with a match.
        best_ngram_idx = (first_match_positions >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_ngram_idx = num_ngram_sizes - 1 - best_ngram_idx  # Flip back

        # Match position for the best n-gram.
        best_match_pos = first_match_positions[batch_indices, best_ngram_idx]

        # Avoid data-dependent branching.
        has_any_match = best_match_pos >= 0

        # Length of the best matching n-gram.
        best_ngram_lengths = ngram_lengths[best_ngram_idx]

        # Start position right after the matched suffix.
        draft_start = torch.where(
            has_any_match,
            best_match_pos + best_ngram_lengths,
            torch.zeros_like(best_match_pos),
        )
        tokens_available = seq_lengths - draft_start

        # Gather indices for draft tokens.
        draft_indices = draft_start.unsqueeze(1) + torch.arange(
            num_draft_tokens, device=device
        )
        draft_indices = draft_indices.clamp(min=0, max=max_seq_len - 1)

        # Extract draft tokens; gather always runs.
        draft_tokens = torch.gather(token_ids, 1, draft_indices)

        # Mask positions beyond available tokens.
        position_indices = torch.arange(num_draft_tokens, device=device).unsqueeze(0)
        valid_positions = position_indices < tokens_available.unsqueeze(1)

        draft_tokens = torch.where(
            valid_positions,
            draft_tokens,
            torch.full_like(draft_tokens, -1),
        )

        # If no match, mask all positions.
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

        # Infer batch size to preserve dynamic shape.
        actual_batch_size = token_ids_gpu.shape[0]

        # Allocate in forward so torch.compile can optimize.
        # NOTE(patchy): Do NOT pre-allocate this as a buffer
        #               it breaks torch.compile
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
        self.device = device

        self.kernel = NgramGPUKernel(
            vllm_config=self.vllm_config, prefix="ngram_gpu_kernel", device=device
        )
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
        token_ids = torch.zeros(
            batch_size,
            max_seq_len,
            dtype=torch.int32,
            device=device,
        )

        num_tokens = torch.randint(
            pattern_len, max_seq_len, (batch_size,), dtype=torch.int32, device=device
        )

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

        Steps: scatter new tokens, update lengths, build masks, run kernel.

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

        # Scatter newly sampled tokens into token_ids_gpu.
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

        # Update num_tokens_no_spec in-place.
        num_tokens_no_spec += valid_sampled_tokens_count

        # Compute validity masks.
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
        sampled_token_ids: torch.Tensor | list[list[int]],
        gpu_input_batch: InputBatch,
        token_ids_gpu: torch.Tensor,
        num_tokens_no_spec: torch.Tensor,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare speculative decoding inputs on device:
        compute next token ids and valid counts, honoring discarded requests
        and rejected tokens, without CPU-GPU sync.
        """
        num_reqs = gpu_input_batch.num_reqs

        if isinstance(sampled_token_ids, list):
            # When disable_padded_drafter_batch=True, sampled_token_ids is
            # an irregular list[list[int]] where sublists may have different
            # lengths (including empty lists for discarded requests).
            # Pad all sublists to the same length with -1 before converting
            # to tensor.
            max_len = max(
                (len(sublist) for sublist in sampled_token_ids),
                default=0,
            )
            # Ensure at least length 1 for tensor creation
            max_len = max(max_len, 1)
            padded_list = [
                sublist + [-1] * (max_len - len(sublist))
                for sublist in sampled_token_ids
            ]
            sampled_token_ids = torch.tensor(
                padded_list, dtype=torch.int32, device=self.device
            )
        assert isinstance(sampled_token_ids, torch.Tensor), (
            "sampled_token_ids should be a torch.Tensor for ngram_gpu"
        )

        # Backup last valid token before speculative tokens.
        backup_indices = (num_tokens_no_spec[:num_reqs] - 1).clamp(min=0).long()
        backup_next_token_ids = torch.gather(
            token_ids_gpu[:num_reqs], dim=1, index=backup_indices.unsqueeze(1)
        ).squeeze(1)

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        # Invalidate sampled tokens for discarded requests.
        discard_mask_expanded = discard_request_mask[:num_reqs].unsqueeze(1)
        valid_sampled_token_ids_gpu.masked_fill_(discard_mask_expanded, -1)

        # Mask valid tokens within each request.
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
        )

        # Count valid tokens per request.
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Rightmost valid index per row.
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Last valid token from each row; undefined if none.
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
        ).squeeze(1)

        # Use last token if valid; otherwise fallback to backup.
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            backup_next_token_ids,
        )

        return next_token_ids, valid_sampled_tokens_count, valid_sampled_token_ids_gpu

    def load_model(self, *args, **kwargs):
        self.kernel.load_model(*args, **kwargs)


def update_scheduler_for_empty_drafts(
    is_empty_draft_tokens_event: torch.cuda.Event,
    is_empty_draft_tokens_cpu: torch.Tensor,
    scheduler_output: "SchedulerOutput",
    req_id_to_index: dict[str, int],
) -> None:
    """Update scheduler_output for requests with empty draft tokens.

    Called between _update_states and _prepare_inputs to delay the sync so
    the async D2H copy can finish and reduce kernel bubbles.

    Args:
        scheduler_output: The scheduler output to update.
        req_id_to_index: A mapping from request IDs to their indices in the batch.
    """
    req_data = scheduler_output.scheduled_cached_reqs

    # Sync the is_empty_draft_tokens copy (should be complete).
    is_empty_draft_tokens_event.synchronize()

    for req_id in req_data.req_ids:
        req_index = req_id_to_index.get(req_id)

        if req_index is None or is_empty_draft_tokens_cpu[req_index].item():
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, []
            )
            num_spec_tokens = len(spec_token_ids)
            scheduler_output.total_num_scheduled_tokens -= num_spec_tokens
            scheduler_output.num_scheduled_tokens[req_id] -= num_spec_tokens
            scheduler_output.scheduled_spec_decode_tokens.pop(req_id, None)


def update_ngram_gpu_tensors_incremental(
    input_batch: InputBatch,
    token_ids_gpu_tensor: torch.Tensor,
    num_tokens_no_spec_gpu: torch.Tensor,
    new_reqs: list[CachedRequestState],
    device: torch.device,
) -> None:
    """Incrementally update token_ids_gpu_tensor and num_tokens_no_spec_gpu
    for ngram GPU proposer.

    Handles three cases: first run, reorder, and new/resumed requests.

    Args:
        new_reqs: List of new or resumed requests that need full tensor copy.
    """
    prev_req_id_to_index = input_batch.prev_req_id_to_index
    curr_req_id_to_index = input_batch.req_id_to_index

    if not curr_req_id_to_index:
        return

    new_req_ids = {req.req_id for req in new_reqs}

    # First run, no previous state
    if prev_req_id_to_index is None:
        for idx in curr_req_id_to_index.values():
            num_tokens = input_batch.num_tokens_no_spec[idx]
            if num_tokens > 0:
                token_ids_gpu_tensor[idx, :num_tokens].copy_(
                    input_batch.token_ids_cpu_tensor[idx, :num_tokens],
                    non_blocking=True,
                )
                num_tokens_no_spec_gpu[idx : idx + 1].copy_(
                    input_batch.num_tokens_no_spec_cpu_tensor[idx : idx + 1],
                    non_blocking=True,
                )
        return

    # Detect index changes for reorder
    reorder_src: list[int] = []
    reorder_dst: list[int] = []

    for req_id, curr_idx in curr_req_id_to_index.items():
        if req_id in new_req_ids:
            continue

        prev_idx = prev_req_id_to_index.get(req_id)
        if prev_idx is not None and prev_idx != curr_idx:
            reorder_src.append(prev_idx)
            reorder_dst.append(curr_idx)

    if reorder_src:
        src_tensor = torch.tensor(reorder_src, dtype=torch.long, device=device)
        dst_tensor = torch.tensor(reorder_dst, dtype=torch.long, device=device)

        temp_token_ids = token_ids_gpu_tensor[src_tensor].clone()
        temp_num_tokens = num_tokens_no_spec_gpu[src_tensor].clone()

        token_ids_gpu_tensor[dst_tensor] = temp_token_ids
        num_tokens_no_spec_gpu[dst_tensor] = temp_num_tokens

    # Full copy for new/resumed requests
    for req_state in new_reqs:
        new_req_idx = curr_req_id_to_index.get(req_state.req_id)
        if new_req_idx is None:
            continue

        num_tokens = input_batch.num_tokens_no_spec[new_req_idx]
        if num_tokens > 0:
            token_ids_gpu_tensor[new_req_idx, :num_tokens].copy_(
                input_batch.token_ids_cpu_tensor[new_req_idx, :num_tokens],
                non_blocking=True,
            )
            num_tokens_no_spec_gpu[new_req_idx : new_req_idx + 1].copy_(
                input_batch.num_tokens_no_spec_cpu_tensor[
                    new_req_idx : new_req_idx + 1
                ],
                non_blocking=True,
            )


def copy_is_empty_draft_tokens(
    is_empty_draft_tokens_cpu: torch.Tensor,
    is_empty_draft_tokens_copy_stream: torch.cuda.Stream,
    is_empty_draft_tokens_event: torch.cuda.Event,
    is_empty_draft_tokens: torch.Tensor | None,
    batch_size: int,
) -> None:
    """Async copy is_empty_draft_tokens to CPU using dedicated stream.

    Uses a separate CUDA stream to overlap D2H copy with kernel execution.
    """
    if is_empty_draft_tokens is None:
        return

    num_reqs_to_copy = min(batch_size, is_empty_draft_tokens.shape[0])
    if num_reqs_to_copy <= 0:
        return

    default_stream = torch.cuda.current_stream()
    with torch.cuda.stream(is_empty_draft_tokens_copy_stream):
        is_empty_draft_tokens_copy_stream.wait_stream(default_stream)
        is_empty_draft_tokens_cpu[:num_reqs_to_copy].copy_(
            is_empty_draft_tokens[:num_reqs_to_copy], non_blocking=True
        )
        is_empty_draft_tokens_event.record()
