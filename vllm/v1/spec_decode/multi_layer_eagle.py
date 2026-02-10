# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import MultiLayerEagleMetadata

logger = init_logger(__name__)

BLOCK_HIDDEN = 128
BLOCK_TOKENS = 128


class MultiLayerEagleProposer(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)

        self.layer_num: int = getattr(
            self.speculative_config.draft_model_config.hf_text_config, "n_predict", 0
        )
        self.num_speculative_tokens: int = (
            self.speculative_config.num_speculative_tokens
        )
        if self.num_speculative_tokens != self.layer_num:
            logger.warning_once(
                "For multi_layer_eagle, num_speculative_tokens "
                "does not match layer_num, adjusting to layer_num"
            )
            self.num_speculative_tokens = self.layer_num

    def adjust_input(
        self,
        batch_size: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        multi_layer_eagle_metadata: MultiLayerEagleMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

        MAX_SHIFT = self.layer_num
        assert MAX_SHIFT > 0

        prev_token_ids = target_token_ids.clone()
        prev_positions = target_positions.clone()
        prev_hidden_states = target_hidden_states.clone()
        slot_mapping = common_attn_metadata.slot_mapping

        start_token_indices = common_attn_metadata.query_start_loc[:-1]
        end_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        pos_for_shift = (
            target_positions[0] if target_positions.dim() == 2 else target_positions
        )
        start_token_pos = pos_for_shift[start_token_indices]

        shift = torch.minimum(
            end_token_indices - token_indices_to_sample,
            start_token_pos,
        )
        shift = torch.clamp(shift, min=0)

        # Metadata updates (matches the original reference implementation).
        token_indices_to_sample.add_(shift)
        common_attn_metadata.seq_lens.sub_(shift)

        # NOTE: ignore cpu data to avoid device sync
        # common_attn_metadata.seq_lens_cpu.copy_(common_attn_metadata.seq_lens,
        #                                         non_blocking=True)
        # query_lens = common_attn_metadata.query_start_loc[
        #     1:] - common_attn_metadata.query_start_loc[:-1]
        # num_computed_tokens = common_attn_metadata.seq_lens - query_lens.to(
        #     common_attn_metadata.seq_lens.dtype)
        # common_attn_metadata.num_computed_tokens_cpu.copy_(
        #     num_computed_tokens.to(
        #         common_attn_metadata.num_computed_tokens_cpu.dtype),
        #     non_blocking=True,
        # )
        # common_attn_metadata.max_seq_len =
        #       int(common_attn_metadata.seq_lens_cpu.max().item())

        cached_lens = multi_layer_eagle_metadata.cached_len
        shift = torch.minimum(shift, cached_lens)

        _multi_layer_eagle_shift_and_cache(
            batch_size=batch_size,
            max_shift=MAX_SHIFT,
            src_token_ids=target_token_ids,
            dst_token_ids=prev_token_ids,
            src_positions=target_positions,
            dst_positions=prev_positions,
            src_hidden_states=target_hidden_states,
            dst_hidden_states=prev_hidden_states,
            src_slot_mapping=slot_mapping,
            dst_slot_mapping=slot_mapping,
            start_token_indices=start_token_indices,
            end_token_indices=end_token_indices,
            token_indices_to_sample=token_indices_to_sample,
            shift=shift,
            cached_lens=cached_lens,
            cached_prev_token_ids=multi_layer_eagle_metadata.cached_token_ids,
            cached_prev_positions=multi_layer_eagle_metadata.cached_positions,
            cached_prev_hidden_states=multi_layer_eagle_metadata.cached_hidden_states,
            cached_slot_mappings=multi_layer_eagle_metadata.cached_slot_mappings,
            common_attn_metadata=common_attn_metadata,
        )

        return prev_token_ids, prev_positions, prev_hidden_states, common_attn_metadata

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        raise Exception(
            "speculative_config.disable_padded_drafter_batch"
            " is not supported now for MultiLayerEagleProposer."
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens
        )
        if use_cudagraphs:
            cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                num_tokens_dp_padded
            )
            num_input_tokens = batch_desc.num_tokens
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            num_input_tokens = num_tokens_dp_padded
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # Make sure to use EAGLE's own buffer during cudagraph capture.
        if (
            self.attn_layer_names
            and slot_mappings is not None
            and self.attn_layer_names[0] in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        adjust_input_kwargs = {
            "batch_size": 1,
            "target_token_ids": self.input_ids[:num_input_tokens],
            "target_positions": self._get_positions(num_input_tokens),
            "target_hidden_states": self.hidden_states[:num_input_tokens],
            "token_indices_to_sample": torch.tensor(
                [num_input_tokens - 1], dtype=torch.int32, device=self.device
            ),
            "common_attn_metadata": CommonAttentionMetadata(
                query_start_loc=torch.tensor(
                    [0, num_input_tokens], dtype=torch.int32, device=self.device
                ),
                query_start_loc_cpu=torch.tensor(
                    [0, num_input_tokens], dtype=torch.int32, device="cpu"
                ),
                seq_lens=torch.tensor(
                    [num_input_tokens], dtype=torch.int32, device=self.device
                ),
                num_reqs=1,
                num_actual_tokens=num_input_tokens,
                max_query_len=num_input_tokens,
                max_seq_len=self.max_model_len,
                block_table_tensor=torch.tensor(
                    [], dtype=torch.int32, device=self.device
                ),
                slot_mapping=self.arange[:num_input_tokens],
                logits_indices_padded=None,
                num_logits_indices=None,
                causal=True,
                encoder_seq_lens=None,
            ),
            "multi_layer_eagle_metadata": MultiLayerEagleMetadata.make_dummy(
                layer_num=self.layer_num,
                hidden_size=self.hidden_size,
                device=self.device,
            ),
        }
        # NOTE ensure the jit kernel in _adjust_input can be compiled
        self.adjust_input(**adjust_input_kwargs)

        for fwd_idx in range(self.layer_num):
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                model_kwargs = {
                    "input_ids": input_ids,
                    "positions": self._get_positions(num_input_tokens),
                    "hidden_states": self.hidden_states[:num_input_tokens],
                    "inputs_embeds": inputs_embeds,
                    "spec_step_idx": fwd_idx,
                }

                self.model(**model_kwargs)


def _multi_layer_eagle_shift_and_cache(
    *,
    batch_size: int,
    max_shift: int,
    src_token_ids: torch.Tensor,
    dst_token_ids: torch.Tensor,
    src_positions: torch.Tensor,
    dst_positions: torch.Tensor,
    src_hidden_states: torch.Tensor,
    dst_hidden_states: torch.Tensor,
    src_slot_mapping: torch.Tensor,
    dst_slot_mapping: torch.Tensor,
    start_token_indices: torch.Tensor,
    end_token_indices: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    shift: torch.Tensor,
    cached_lens: torch.Tensor,
    cached_prev_token_ids: torch.Tensor,
    cached_prev_positions: torch.Tensor,
    cached_prev_hidden_states: torch.Tensor,
    cached_slot_mappings: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
):
    if batch_size == 0:
        return

    assert max_shift > 0
    assert cached_prev_positions.is_contiguous()
    assert cached_prev_token_ids.is_contiguous()
    assert cached_prev_hidden_states.is_contiguous()
    assert cached_slot_mappings.is_contiguous()
    assert src_hidden_states.is_contiguous()
    assert dst_hidden_states.is_contiguous()

    # If src/dst are the same tensor, shifting is unsafe without a separate src.
    if src_slot_mapping.data_ptr() == dst_slot_mapping.data_ptr():
        src_slot_mapping = src_slot_mapping.clone()

    # Cache extraction for the next call.
    store_start = torch.maximum(
        start_token_indices,
        (token_indices_to_sample + 1 - max_shift),
    )
    store_lens = torch.clamp(
        token_indices_to_sample - store_start + 1,
        min=0,
        max=max_shift,
    )

    # Avoid device sync: query length == (end - start + 1) == diff of
    # query_start_loc (CPU copy).
    max_window_len = int(
        (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        .max()
        .item()
    )
    num_blocks = max(1, (max_window_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS)

    _shift_and_gather_cache_1d_kernel[(batch_size, num_blocks)](
        src_token_ids,
        dst_token_ids,
        cached_prev_token_ids,
        start_token_indices,
        end_token_indices,
        shift,
        cached_lens,
        store_start,
        store_lens,
        MAX_SHIFT=max_shift,
        PADDED_SHIFT=triton.next_power_of_2(max_shift),
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    _shift_and_gather_cache_1d_kernel[(batch_size, num_blocks)](
        src_slot_mapping,
        dst_slot_mapping,
        cached_slot_mappings,
        start_token_indices,
        end_token_indices,
        shift,
        cached_lens,
        store_start,
        store_lens,
        MAX_SHIFT=max_shift,
        PADDED_SHIFT=triton.next_power_of_2(max_shift),
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    _shift_and_gather_cache_1d_kernel[(batch_size, num_blocks)](
        src_positions,
        dst_positions,
        cached_prev_positions,
        start_token_indices,
        end_token_indices,
        shift,
        cached_lens,
        store_start,
        store_lens,
        MAX_SHIFT=max_shift,
        PADDED_SHIFT=triton.next_power_of_2(max_shift),
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    hidden_size = int(dst_hidden_states.shape[1])
    # Hidden blocking avoids extremely large Triton tiles (and huge cubins)
    # when hidden_size is large.
    num_hidden_blocks = max(1, (hidden_size + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN)

    _shift_and_gather_hidden_kernel[(batch_size, num_blocks, num_hidden_blocks)](
        src_hidden_states,
        dst_hidden_states,
        cached_prev_hidden_states,
        start_token_indices,
        end_token_indices,
        shift,
        cached_lens,
        store_start,
        store_lens,
        MAX_SHIFT=max_shift,
        PADDED_SHIFT=triton.next_power_of_2(max_shift),
        HIDDEN_SIZE=hidden_size,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        num_warps=4,
    )

    cached_lens.copy_(store_lens)
    return


@triton.jit
def _shift_and_gather_cache_1d_kernel(
    src_ptr,
    dst_ptr,
    cached_ptr,
    start_ptr,
    end_ptr,
    shift_ptr,
    cached_len_ptr,
    store_start_ptr,
    store_len_ptr,
    MAX_SHIFT: tl.constexpr,
    PADDED_SHIFT: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    # Per-sequence "shift + gather" for packed 1D arrays (token ids, positions,
    # slot mappings, ...).
    #
    # We operate on a packed batch where each sequence (request) occupies a
    # contiguous window [start, end] (inclusive) in a flattened tensor.
    # For the next speculative step, we build a right-shifted version of each
    # window. The shift amount can differ per sequence.
    #
    # For a single sequence (0-based index i within its window):
    #   - Prefix (i < shift):
    #       dst[start + i] = cached[cached_len - shift + i]
    #   - Body   (i >= shift):
    #       dst[start + i] = src[start + i - shift]
    #
    # The vacated prefix is filled from a small per-sequence cache (up to
    # MAX_SHIFT elements) that stores values from previous speculative steps.
    #
    # Example:
    #   cached_tail = [a3, a4]
    #   src_window  = [b0, b1, b2, b3, b4]
    #   shift = 2
    #   -> dst_window = [a3, a4, b0, b1, b2]
    #
    # After dst is produced, we refresh cached_ptr[seq, :] with a suffix of dst
    # (specified by store_start / store_len) so the next call can populate its
    # prefix from cache.
    pid_seq = tl.program_id(0)
    pid_blk = tl.program_id(1)

    start = tl.load(start_ptr + pid_seq).to(tl.int32)
    end = tl.load(end_ptr + pid_seq).to(tl.int32)
    shift = tl.load(shift_ptr + pid_seq).to(tl.int32)
    cached_len = tl.load(cached_len_ptr + pid_seq).to(tl.int32)

    assert cached_len >= shift

    # get dst indices
    base = pid_blk * BLOCK_TOKENS
    k = tl.arange(0, BLOCK_TOKENS)
    offs = base + k
    dst_idx = start + offs

    # get dst mask
    window_len = end - start + 1
    mask = offs < window_len

    # load from cached
    base_cached = cached_ptr + pid_seq * MAX_SHIFT
    cached_idx = cached_len - shift + offs
    cached_mask = offs < shift
    val_cached = tl.load(base_cached + cached_idx, mask=mask & cached_mask, other=0)

    # load from src
    src_idx = start + offs - shift
    val_src = tl.load(src_ptr + src_idx, mask=mask & ~cached_mask, other=0)

    # store to dst
    val = tl.where(cached_mask, val_cached, val_src)
    tl.store(dst_ptr + dst_idx, val, mask=mask)

    # Store into the per-sequence cache.
    #
    # Cache layout: [batch_size, MAX_SHIFT] (flattened). We always write the
    # full MAX_SHIFT region (zero-padded when store_len < MAX_SHIFT) to keep the
    # cache contiguous.
    store_start = tl.load(store_start_ptr + pid_seq).to(tl.int32)
    store_len = tl.load(store_len_ptr + pid_seq).to(tl.int32)
    m = tl.arange(0, PADDED_SHIFT)
    store_mask = m < MAX_SHIFT
    dst_idx = store_start + m
    val = tl.load(dst_ptr + dst_idx, mask=store_mask & (m < store_len), other=0)
    tl.store(base_cached + m, val, mask=store_mask)


@triton.jit
def _shift_and_gather_hidden_kernel(
    src_ptr,
    dst_ptr,
    cached_ptr,
    start_ptr,
    end_ptr,
    shift_ptr,
    cached_len_ptr,
    store_start_ptr,
    store_len_ptr,
    MAX_SHIFT: tl.constexpr,
    PADDED_SHIFT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    # Per-sequence "shift + gather" for hidden states.
    #
    # This kernel implements the same logical transformation as
    # _shift_and_gather_cache_1d_kernel, but operates on hidden states with
    # shape [num_tokens, hidden_size].
    #
    # Layout:
    #   - src_ptr / dst_ptr: packed hidden states [num_tokens, hidden_size]
    #   - cached_ptr: per-sequence cache [batch_size, MAX_SHIFT, hidden_size]
    #
    # For each sequence window [start, end] (inclusive) and its shift value, for
    # 0-based index i within the window:
    #   - Prefix (i < shift):
    #       dst[start + i, :] = cached[seq, cached_len - shift + i, :]
    #   - Body   (i >= shift):
    #       dst[start + i, :] = src[start + i - shift, :]
    #
    # We tile over tokens (BLOCK_TOKENS) and hidden dim (BLOCK_HIDDEN) to avoid
    # extremely large Triton tiles when hidden_size is large. As in the 1D
    # kernel, we refresh cached_ptr[seq, :, :] with a suffix of dst so the next
    # call can populate its prefix from cache.
    pid_seq = tl.program_id(0)
    pid_blk = tl.program_id(1)
    pid_hid = tl.program_id(2)

    start = tl.load(start_ptr + pid_seq).to(tl.int32)
    end = tl.load(end_ptr + pid_seq).to(tl.int32)
    shift = tl.load(shift_ptr + pid_seq).to(tl.int32)
    cached_len = tl.load(cached_len_ptr + pid_seq).to(tl.int32)

    assert cached_len >= shift

    # get dst indices
    base = pid_blk * BLOCK_TOKENS
    k = tl.arange(0, BLOCK_TOKENS)
    tok_offs = base + k
    dst_tok = start + tok_offs
    n = pid_hid * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    dst_ptrs = dst_ptr + dst_tok[:, None] * HIDDEN_SIZE + n[None, :] * 1

    # get dst mask
    window_len = end - start + 1
    tok_mask = tok_offs < window_len
    n_mask = n < HIDDEN_SIZE
    mask = tok_mask[:, None] & n_mask[None, :]

    # load from cached
    base_cached = cached_ptr + pid_seq * HIDDEN_SIZE * MAX_SHIFT
    cached_tok = cached_len - shift + tok_offs
    cached_ptrs = base_cached + cached_tok[:, None] * HIDDEN_SIZE + n[None, :] * 1
    cached_mask = tok_offs < shift
    val_cached = tl.load(cached_ptrs, mask=mask & cached_mask[:, None], other=0)

    # load from src
    src_tok = start + tok_offs - shift
    src_ptrs = src_ptr + src_tok[:, None] * HIDDEN_SIZE + n[None, :] * 1
    val_src = tl.load(src_ptrs, mask=mask & ~cached_mask[:, None], other=0)

    # store to dst
    val = tl.where(cached_mask[:, None], val_cached, val_src)
    tl.store(dst_ptrs, val, mask=mask)

    # store to cached
    store_start = tl.load(store_start_ptr + pid_seq).to(tl.int32)
    store_len = tl.load(store_len_ptr + pid_seq).to(tl.int32)
    m = tl.arange(0, PADDED_SHIFT)
    m_mask = (m < MAX_SHIFT) & (m < store_len)
    store_tok = store_start + m
    dst_ptrs = dst_ptr + store_tok[:, None] * HIDDEN_SIZE + n[None, :] * 1
    store_ptrs = base_cached + m[:, None] * HIDDEN_SIZE + n[None, :] * 1
    mask = m_mask[:, None] & n_mask[None, :]
    val = tl.load(dst_ptrs, mask=mask, other=0)
    tl.store(store_ptrs, val, mask=mask)
