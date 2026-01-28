# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer


@triton.jit
def ptd_prepare_inputs_kernel(
    # Input tensors from target model
    target_token_ids_ptr,  # [num_tokens] - verified token IDs
    target_positions_ptr,  # [num_tokens] - verified positions
    target_hidden_ptr,  # [num_tokens, hidden_size] - verified hidden states
    mask_hidden_ptr,  # [hidden_size] - learned mask embedding for draft positions
    next_token_ids_ptr,  # [batch_size] - sampled next tokens per request
    last_token_indices_ptr,  # [batch_size] - index of last verified token per request
    original_slot_mapping_ptr,  # [num_tokens] - KV cache slots for verified tokens
    block_table_ptr,  # [batch_size, max_blocks] - KV cache block table
    in_query_start_loc_ptr,  # [batch_size + 1] - input query boundaries
    out_query_start_loc_ptr,  # [batch_size + 1] - output query boundaries
    # Output tensors for draft model
    out_input_ids_ptr,  # [num_out_tokens] - token IDs for draft
    out_positions_ptr,  # [num_out_tokens] - positions for draft
    out_hidden_ptr,  # [num_out_tokens, hidden_size] - hidden states for draft
    out_slot_mapping_ptr,  # [num_out_tokens] - KV cache slots for draft
    # Constants
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,  # KV cache block size
    max_blocks: tl.constexpr,  # max blocks per sequence
    mask_token_id: tl.constexpr,  # special token ID for draft positions
    max_model_len: tl.constexpr,
    HIDDEN_TILE_SIZE: tl.constexpr,  # tile size for hidden dim parallelism
):
    """
    Prepares inputs for parallel token drafting.

    Parallel drafting generates K draft tokens in a single forward pass by:
    1. Shifting verified tokens left (drop first, append next_token)
    2. Appending K-1 mask tokens for parallel draft positions
    3. Using learned mask_hidden embedding for draft position hidden states

    Grid: (num_out_tokens, num_hidden_tiles)
        - First dim: one program per output token
        - Second dim: tiles over hidden_size for parallel hidden state copy
          (HIDDEN_TILE_SIZE=256 is standard for hidden dim operations in vLLM)

    The kernel handles two types of positions:
        - Verified positions (local_idx <= last_idx): copy from target tensors
        - Draft positions (local_idx > last_idx): use mask_token_id and mask_hidden
    """
    # Program IDs: token_idx iterates over output tokens,
    # hidden_tile_i tiles over the hidden dimension
    token_idx = tl.program_id(0)
    hidden_tile_i = tl.program_id(1)

    # Find which request this token belongs to
    req_idx = 0
    for r in range(batch_size):
        out_start = tl.load(out_query_start_loc_ptr + r)
        out_end = tl.load(out_query_start_loc_ptr + r + 1)
        req_idx = tl.where((token_idx >= out_start) & (token_idx < out_end), r, req_idx)

    in_start = tl.load(in_query_start_loc_ptr + req_idx)
    out_start = tl.load(out_query_start_loc_ptr + req_idx)
    global_last_idx = tl.load(last_token_indices_ptr + req_idx)
    last_idx = global_last_idx - in_start

    local_idx = token_idx - out_start
    is_verified = local_idx <= last_idx

    # Scalar outputs (token_ids, positions, slots) are written only by the first
    # hidden tile (hidden_tile_i == 0) to avoid redundant writes. All tiles
    # participate in copying hidden states since that's the expensive operation.
    if hidden_tile_i == 0:
        if is_verified:
            if local_idx < last_idx:
                out_tok = tl.load(target_token_ids_ptr + in_start + local_idx + 1)
            else:
                out_tok = tl.load(next_token_ids_ptr + req_idx)
        else:
            out_tok = mask_token_id
        tl.store(out_input_ids_ptr + token_idx, out_tok)

        if is_verified:
            out_pos = tl.load(target_positions_ptr + in_start + local_idx)
        else:
            last_pos = tl.load(target_positions_ptr + in_start + last_idx)
            out_pos = last_pos + (local_idx - last_idx)
            out_pos = tl.where(out_pos >= max_model_len, 0, out_pos)
        tl.store(out_positions_ptr + token_idx, out_pos)

        if is_verified:
            slot = tl.load(original_slot_mapping_ptr + in_start + local_idx)
        else:
            last_pos = tl.load(target_positions_ptr + in_start + last_idx)
            raw_draft_pos = last_pos + (local_idx - last_idx)
            is_overflow = raw_draft_pos >= max_model_len
            # Clamp to 0 for block table lookup (but will use -1 for actual slot)
            draft_pos = tl.where(is_overflow, 0, raw_draft_pos)
            block_num = draft_pos // block_size
            block_offset = draft_pos % block_size
            block_id = tl.load(block_table_ptr + req_idx * max_blocks + block_num)
            computed_slot = (block_id * block_size + block_offset).to(tl.int64)
            # Use PADDING_SLOT_ID (-1) for overflow positions to avoid KV cache writes
            # Cast -1 to int64 via arithmetic: 0 - 1 on int64 tensor
            padding_slot_id = computed_slot * 0 - 1
            slot = tl.where(is_overflow, padding_slot_id, computed_slot)
        tl.store(out_slot_mapping_ptr + token_idx, slot)

    # All tiles copy their portion of hidden states
    h_start = hidden_tile_i * HIDDEN_TILE_SIZE
    h_offs = h_start + tl.arange(0, HIDDEN_TILE_SIZE)
    h_mask = h_offs < hidden_size

    if is_verified:
        hidden_vals = tl.load(
            target_hidden_ptr + (in_start + local_idx) * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        )
    else:
        hidden_vals = tl.load(mask_hidden_ptr + h_offs, mask=h_mask, other=0.0)

    tl.store(
        out_hidden_ptr + token_idx * hidden_size + h_offs, hidden_vals, mask=h_mask
    )


class PtdEagleProposer(EagleProposer):
    """Generates draft tokens in a single forward pass using mask tokens."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)

        # Parallel drafting operates in text-only mode
        self.supports_mm_inputs = False

        self.slot_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.draft_token_offsets = torch.arange(
            self.num_speculative_tokens, device=device, dtype=torch.int64
        )

        self.mask_hidden: torch.Tensor | None = None
        self.mask_token_id: int | None = None
        self.block_size = vllm_config.cache_config.block_size

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)

        # Parallel drafting requires mask token id from config
        config = self.draft_model_config.hf_config
        self.mask_token_id = getattr(config, "ptd_token_id", None)
        if self.mask_token_id is None:
            raise ValueError(
                "Parallel drafting requires 'ptd_token_id' in draft model config.json"
            )
        self.mask_token_id = int(self.mask_token_id)

        self.mask_hidden = self.model.mask_hidden

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mm_embed_inputs is not None:
            raise NotImplementedError(
                "Parallel drafting does not support multimodal inputs"
            )

        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )

        if self.attn_metadata_builder is None:
            self.attn_metadata_builder = self._get_attention_metadata_builder()

        draft_len = self.num_speculative_tokens - 1
        input_query_start_loc = common_attn_metadata.query_start_loc

        accepted_lengths = last_token_indices - input_query_start_loc[:batch_size] + 1
        out_lens = accepted_lengths + draft_len

        output_query_start_loc = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        output_query_start_loc[1:] = torch.cumsum(out_lens, dim=0)

        total_out = common_attn_metadata.num_actual_tokens + batch_size * draft_len

        input_query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        accepted_lengths_cpu = (
            input_query_start_loc_cpu[1 : batch_size + 1]
            - input_query_start_loc_cpu[:batch_size]
        )
        output_query_start_loc_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        output_query_start_loc_cpu[1:] = torch.cumsum(
            accepted_lengths_cpu + draft_len, dim=0
        )

        slot_mapping = self._prepare_ptd_inputs(
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            last_token_indices,
            common_attn_metadata.slot_mapping,
            common_attn_metadata.block_table_tensor,
            input_query_start_loc,
            output_query_start_loc,
            total_out,
            batch_size,
        )

        seq_lens = common_attn_metadata.seq_lens
        if num_rejected_tokens_gpu is not None:
            seq_lens = seq_lens - num_rejected_tokens_gpu
        seq_lens = (seq_lens + self.num_speculative_tokens).to(
            common_attn_metadata.seq_lens.dtype
        )

        common_attn_metadata.query_start_loc = output_query_start_loc
        common_attn_metadata.query_start_loc_cpu = output_query_start_loc_cpu
        common_attn_metadata.seq_lens = seq_lens
        common_attn_metadata.num_actual_tokens = total_out
        common_attn_metadata.max_query_len = (
            common_attn_metadata.max_query_len + draft_len
        )
        common_attn_metadata.max_seq_len = common_attn_metadata.max_seq_len + draft_len
        common_attn_metadata.slot_mapping = slot_mapping
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata._num_computed_tokens_cpu = None

        attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )
        per_layer_metadata = {name: attn_metadata for name in self.attn_layer_names}

        num_input, cudagraph_mode = self._get_ptd_cudagraph_config(total_out)

        hidden_states = self._run_ptd_forward(
            num_input, total_out, per_layer_metadata, cudagraph_mode
        )

        ends = output_query_start_loc[1 : batch_size + 1]
        starts = ends - self.num_speculative_tokens
        indices = starts.unsqueeze(1) + self.draft_token_offsets
        hidden_states_selected = hidden_states[indices.flatten()]

        logits = self.model.compute_logits(hidden_states_selected)
        return logits.argmax(dim=-1).view(batch_size, self.num_speculative_tokens)

    def _prepare_ptd_inputs(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        input_query_start_loc: torch.Tensor,
        output_query_start_loc: torch.Tensor,
        total_out: int,
        batch_size: int,
    ) -> torch.Tensor:
        HIDDEN_TILE_SIZE = 256
        num_hidden_tiles = (self.hidden_size + HIDDEN_TILE_SIZE - 1) // HIDDEN_TILE_SIZE

        ptd_prepare_inputs_kernel[(total_out, num_hidden_tiles)](
            target_token_ids,
            target_positions,
            target_hidden_states,
            self.mask_hidden,
            next_token_ids,
            last_token_indices,
            slot_mapping,
            block_table,
            input_query_start_loc,
            output_query_start_loc,
            self.input_ids,
            self.positions,
            self.hidden_states,
            self.slot_buffer,
            batch_size=batch_size,
            hidden_size=self.hidden_size,
            block_size=self.block_size,
            max_blocks=block_table.shape[1],
            mask_token_id=self.mask_token_id,
            max_model_len=self.max_model_len,
            HIDDEN_TILE_SIZE=HIDDEN_TILE_SIZE,
        )
        return self.slot_buffer[:total_out]

    def _get_ptd_cudagraph_config(self, num_tokens: int) -> tuple[int, CUDAGraphMode]:
        num_padded, _ = self._pad_batch_across_dp(num_tokens, num_tokens)

        # Use cudagraph_dispatcher for CUDA graph decisions (compatible with nightly)
        cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_padded
        )
        return batch_desc.num_tokens, cudagraph_runtime_mode

    def _run_ptd_forward(
        self,
        num_input: int,
        num_out: int,
        per_layer_metadata: dict,
        cudagraph_mode: CUDAGraphMode,
    ) -> torch.Tensor:
        with set_forward_context(
            per_layer_metadata,
            self.vllm_config,
            num_tokens=num_input,
            cudagraph_runtime_mode=cudagraph_mode,
        ):
            result = self.model(
                input_ids=self.input_ids[:num_input],
                positions=self._get_positions(num_input),
                hidden_states=self.hidden_states[:num_input],
                inputs_embeds=None,
            )
            hidden_states = result[0] if isinstance(result, tuple) else result
        return hidden_states[:num_out]
