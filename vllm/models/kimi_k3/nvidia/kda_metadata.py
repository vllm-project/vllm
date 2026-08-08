# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 specialization of GDN attention metadata.

The request classification and cudagraph staging intentionally mirror
``GDNAttentionMetadataBuilder``. Kimi-K3 builds the metadata required by its
prefill KDA kernel internally, so this builder omits the shared FLA chunk
metadata construction.
"""

from dataclasses import dataclass
from functools import cache

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec


@cache
def _metadata_launch_pdl() -> bool:
    return current_platform.is_arch_support_pdl()


@triton.jit(do_not_specialize=["num_requests"])
def _get_aligned_state_indices_kernel(
    block_table_ptr,
    seq_lens_ptr,
    state_indices_ptr,
    block_table_stride_0: tl.constexpr,
    block_table_stride_1: tl.constexpr,
    seq_lens_stride: tl.constexpr,
    state_indices_stride_0: tl.constexpr,
    state_indices_stride_1: tl.constexpr,
    num_requests,
    CACHE_BLOCK_SIZE: tl.constexpr,
    NUM_STATE_SLOTS: tl.constexpr,
    BLOCK_STATE_SLOTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid_row = rows < num_requests
    seq_lens = tl.load(
        seq_lens_ptr + rows * seq_lens_stride,
        mask=valid_row,
        other=1,
    )
    # Triton truncates signed division toward zero, unlike PyTorch floor
    # division. Clamping makes both semantics equivalent for seq_lens <= 0.
    first_state_slot = tl.maximum((seq_lens - 1) // CACHE_BLOCK_SIZE, 0)

    state_slots = tl.arange(0, BLOCK_STATE_SLOTS)
    valid_state_slot = state_slots < NUM_STATE_SLOTS
    state_indices = tl.load(
        block_table_ptr
        + rows[:, None] * block_table_stride_0
        + (first_state_slot[:, None] + state_slots[None, :]) * block_table_stride_1,
        mask=valid_row[:, None] & valid_state_slot[None, :],
    )
    tl.store(
        state_indices_ptr
        + rows[:, None] * state_indices_stride_0
        + state_slots[None, :] * state_indices_stride_1,
        state_indices,
        mask=valid_row[:, None] & valid_state_slot[None, :],
    )


def _mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: MambaSpec,
    mamba_cache_mode: str,
) -> torch.Tensor:
    if mamba_cache_mode in ("all", "none"):
        return block_table

    assert block_table.is_cuda and seq_lens.is_cuda
    num_requests = block_table.shape[0]
    num_state_slots = 1 + kv_cache_spec.num_speculative_blocks
    state_indices = torch.empty(
        (num_requests, num_state_slots),
        dtype=block_table.dtype,
        device=block_table.device,
    )
    BLOCK_ROWS = 32
    grid = (triton.cdiv(num_requests, BLOCK_ROWS),)
    _get_aligned_state_indices_kernel[grid](
        block_table,
        seq_lens,
        state_indices,
        block_table.stride(0),
        block_table.stride(1),
        seq_lens.stride(0),
        state_indices.stride(0),
        state_indices.stride(1),
        num_requests,
        CACHE_BLOCK_SIZE=kv_cache_spec.block_size,
        NUM_STATE_SLOTS=num_state_slots,
        BLOCK_STATE_SLOTS=triton.next_power_of_2(num_state_slots),
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=1,
        launch_pdl=_metadata_launch_pdl(),
    )
    return state_indices


@triton.jit(do_not_specialize=["num_spec_decodes", "batch_size"])
def _stage_spec_decode_metadata_kernel(
    state_indices_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    staged_state_indices_ptr,
    staged_query_start_loc_ptr,
    staged_num_accepted_tokens_ptr,
    state_indices_stride_0: tl.constexpr,
    state_indices_stride_1: tl.constexpr,
    staged_state_indices_stride_0: tl.constexpr,
    staged_state_indices_stride_1: tl.constexpr,
    num_spec_decodes,
    batch_size,
    NUM_STATE_SLOTS: tl.constexpr,
    BLOCK_STATE_SLOTS: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    real_request = rows < num_spec_decodes

    state_slots = tl.arange(0, BLOCK_STATE_SLOTS)
    valid_state_slot = state_slots < NUM_STATE_SLOTS
    state_indices = tl.load(
        state_indices_ptr
        + rows[:, None] * state_indices_stride_0
        + state_slots[None, :] * state_indices_stride_1,
        mask=real_request[:, None] & valid_state_slot[None, :],
        other=NULL_STATE_ID,
    )
    tl.store(
        staged_state_indices_ptr
        + rows[:, None] * staged_state_indices_stride_0
        + state_slots[None, :] * staged_state_indices_stride_1,
        state_indices,
        mask=(rows < batch_size)[:, None] & valid_state_slot[None, :],
    )

    query_row = tl.minimum(rows, num_spec_decodes)
    query_start_loc = tl.load(
        query_start_loc_ptr + query_row,
        mask=rows <= batch_size,
    )
    tl.store(
        staged_query_start_loc_ptr + rows,
        query_start_loc,
        mask=rows <= batch_size,
    )

    num_accepted_tokens = tl.load(
        num_accepted_tokens_ptr + rows,
        mask=real_request,
        other=1,
    )
    tl.store(
        staged_num_accepted_tokens_ptr + rows,
        num_accepted_tokens,
        mask=rows < batch_size,
    )


def stage_spec_decode_metadata(
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    staged_state_indices: torch.Tensor,
    staged_query_start_loc: torch.Tensor,
    staged_num_accepted_tokens: torch.Tensor,
    *,
    num_spec_decodes: int,
) -> None:
    """Stage speculative-decode metadata into CUDA-graph buffers."""
    assert state_indices.is_cuda
    assert state_indices.ndim == 2
    batch_size, num_state_slots = staged_state_indices.shape
    BLOCK_ROWS = 32
    grid = (triton.cdiv(batch_size + 1, BLOCK_ROWS),)
    _stage_spec_decode_metadata_kernel[grid](
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        staged_state_indices,
        staged_query_start_loc,
        staged_num_accepted_tokens,
        state_indices.stride(0),
        state_indices.stride(1),
        staged_state_indices.stride(0),
        staged_state_indices.stride(1),
        num_spec_decodes,
        batch_size,
        NUM_STATE_SLOTS=num_state_slots,
        BLOCK_STATE_SLOTS=triton.next_power_of_2(num_state_slots),
        NULL_STATE_ID=NULL_BLOCK_ID,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=1,
        launch_pdl=_metadata_launch_pdl(),
    )


@dataclass
class KimiK3KDAMetadata(GDNAttentionMetadata):
    pass


class KimiK3KDAMetadataBuilder(GDNAttentionMetadataBuilder):
    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> KimiK3KDAMetadata:
        m = common_attn_metadata
        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        assert isinstance(self.kv_cache_spec, MambaSpec)
        # Equivalent PyTorch "align" path:
        #   start = ((seq_lens - 1) // block_size).clamp_(min=0)
        #   offsets = torch.arange(1 + num_speculative_blocks, dtype=torch.int32)
        #   indices = (start[:, None] + offsets).to(torch.int64)
        #   block_table_tensor = torch.gather(block_table, 1, indices)
        block_table_tensor = _mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks_cpu = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            # A nonnegative entry identifies a spec request. If no draft token
            # was scheduled, process the whole batch as non-spec instead.
            if num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item() == 0:
                spec_sequence_masks_cpu = None
                num_spec_decodes = 0
            else:
                num_spec_decodes = spec_sequence_masks_cpu.sum().item()

        if num_spec_decodes == 0:
            # The runner orders ordinary decodes before prefills.
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            assert spec_sequence_masks_cpu is not None
            assert num_accepted_tokens is not None
            query_lens_cpu = query_start_loc_cpu.diff()
            num_query_tokens = query_start_loc_cpu[-1].item()

            # Exclude zero-length cudagraph padding from request-indexed
            # non-spec metadata.
            active_non_spec_mask_cpu = (~spec_sequence_masks_cpu) & (query_lens_cpu > 0)
            non_spec_query_lens_cpu = query_lens_cpu[active_non_spec_mask_cpu]
            num_non_spec_requests = non_spec_query_lens_cpu.numel()
            num_non_spec_tokens = non_spec_query_lens_cpu.sum().item()

            # Query length alone cannot distinguish a true decode from a
            # one-token prefill chunk. Packed decode is safe only when every
            # active non-spec request is a true, one-token decode.
            assert m.is_prefilling is not None
            assert m.is_prefilling.device.type == "cpu"
            non_spec_is_prefilling = m.is_prefilling[active_non_spec_mask_cpu]
            use_prefill = torch.any(
                non_spec_is_prefilling | (non_spec_query_lens_cpu != 1)
            ).item()
            if use_prefill:
                num_prefills = num_non_spec_requests
                num_prefill_tokens = num_non_spec_tokens
                num_decodes = 0
                num_decode_tokens = 0
            else:
                num_prefills = 0
                num_prefill_tokens = 0
                num_decodes = num_non_spec_requests
                num_decode_tokens = num_non_spec_tokens

            num_spec_decode_tokens = num_query_tokens - num_non_spec_tokens

            if num_prefills == 0 and num_decodes == 0:
                spec_token_indx = None
                non_spec_token_indx = None
                # Real requests precede trailing cudagraph padding.
                spec_state_indices_tensor = block_table_tensor[
                    :num_spec_decodes, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padding trails real requests, so this prefix already contains
                # the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            else:
                query_lens = query_start_loc.diff()
                spec_sequence_masks_gpu = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks_gpu,
                    query_lens,
                    output_size=num_query_tokens,
                )
                # Stable partitioning preserves request-local token order in
                # both subgroup tensors.
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                # Spec requests carry one state slot per speculative step;
                # non-spec requests use only their current state slot.
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    active_non_spec_mask_cpu, 0
                ]

                spec_query_lens = query_lens[spec_sequence_masks_cpu]
                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    spec_query_lens,
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                if num_prefills > 0:
                    non_spec_query_lens = query_lens[active_non_spec_mask_cpu]
                    non_spec_query_start_loc = torch.zeros(
                        non_spec_query_lens.size(0) + 1,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    torch.cumsum(
                        non_spec_query_lens,
                        dim=0,
                        out=non_spec_query_start_loc[1:],
                    )
                    non_spec_query_start_loc_cpu = torch.zeros(
                        non_spec_query_lens_cpu.size(0) + 1,
                        dtype=torch.int32,
                    )
                    torch.cumsum(
                        non_spec_query_lens_cpu,
                        dim=0,
                        out=non_spec_query_start_loc_cpu[1:],
                    )
                else:
                    # Packed decode consumes one row per request and does not
                    # use cumulative sequence lengths.
                    non_spec_query_start_loc = None
                    non_spec_query_start_loc_cpu = None

                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

            # A row can report 0 accepted tokens when its sampled tokens were
            # discarded (stale async-scheduling step) while its drafts were
            # still scheduled. Such a row has no valid spec state to resume
            # from, and its recurrent state must not advance this step: null
            # its state slots so the kernels skip both the initial-state read
            # and the final-state write, and clamp the count so the slot
            # index (num_accepted_tokens - 1) stays in bounds. This is done
            # out-of-place because the packed-decode branch above slices
            # block_table_tensor, which can be a view of the runner's
            # persistent block table.
            spec_state_indices_tensor = spec_state_indices_tensor.masked_fill(
                (num_accepted_tokens == 0).unsqueeze(-1), NULL_BLOCK_ID
            )
            num_accepted_tokens = num_accepted_tokens.clamp(min=1)

        # Unlike the shared GDN layer, Kimi-K3's prefill KDA wrapper prepares
        # its own chunk indices. Only causal-convolution metadata is needed here.
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        if num_prefills > 0:
            has_initial_state = m.compute_num_computed_tokens() > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[active_non_spec_mask_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # Prepare per-request tensors for cudagraph replay. num_actual_tokens
        # may be token-padded, while state/query/acceptance metadata is indexed
        # by request.
        batch_size = m.num_reqs
        if (
            self.use_full_cuda_graph
            and num_spec_decodes > 0
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            # Equivalent PyTorch staging:
            #   state[:N].copy_(state_src); state[N:].fill_(NULL_BLOCK_ID)
            #   qsl[:N + 1].copy_(qsl_src); qsl[N + 1:].fill_(qsl_src[-1])
            #   accepted[:N].copy_(accepted_src); accepted[N:].fill_(1)
            stage_spec_decode_metadata(
                state_indices=spec_state_indices_tensor,
                query_start_loc=spec_query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                staged_state_indices=self.spec_state_indices_tensor[:batch_size],
                staged_query_start_loc=self.spec_query_start_loc[: batch_size + 1],
                staged_num_accepted_tokens=self.num_accepted_tokens[:batch_size],
                num_spec_decodes=num_spec_decodes,
            )

            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            self.non_spec_state_indices_tensor[num_decodes:batch_size].fill_(
                NULL_BLOCK_ID
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]

        return KimiK3KDAMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=None,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )


class KimiK3KDAAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_KDA"

    @staticmethod
    def get_builder_cls() -> type[KimiK3KDAMetadataBuilder]:
        return KimiK3KDAMetadataBuilder
