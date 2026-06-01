# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import final

import torch

from vllm.lora.layers import LoRAMapping
from vllm.lora.utils import get_captured_lora_counts
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import round_up

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (
        LoRAKernelMeta,
        fused_moe_lora,
        lora_expand,
        lora_shrink,
    )

from vllm import _custom_ops as ops

from .punica_base import PunicaWrapperBase

# ---------------------------------------------------------------------------
# MoE LoRA kernel selection (CUDA BGMV vs Triton SGMV)
# ---------------------------------------------------------------------------
# Control which kernel is used for prefill vs decode via environment variables:
#
#   VLLM_MOE_LORA_PREFILL_BACKEND = triton | cuda   (default: cuda)
#   VLLM_MOE_LORA_DECODE_BACKEND  = triton | cuda   (default: cuda)
#   VLLM_MOE_LORA_DECODE_THRESHOLD = <int>           (default: 0)
#       When set > 0, use CUDA only when num_tokens <= threshold during
#       decode; otherwise fall back to Triton.  0 means always use the
#       decode backend.
#
# Legacy shortcut (overrides both):
#   VLLM_MOE_LORA_BACKEND = triton   -> forces Triton for both
#   VLLM_MOE_LORA_BACKEND = cuda     -> forces CUDA for both
#
import os as _os

_legacy = _os.environ.get("VLLM_MOE_LORA_BACKEND", "").lower()
if _legacy == "triton":
    _MOE_PREFILL_USE_CUDA = False
    _MOE_DECODE_USE_CUDA = False
elif _legacy == "cuda":
    _MOE_PREFILL_USE_CUDA = True
    _MOE_DECODE_USE_CUDA = True
else:
    _MOE_PREFILL_USE_CUDA = (
        _os.environ.get("VLLM_MOE_LORA_PREFILL_BACKEND", "cuda").lower()
        == "cuda"
    )
    _MOE_DECODE_USE_CUDA = (
        _os.environ.get("VLLM_MOE_LORA_DECODE_BACKEND", "cuda").lower()
        == "cuda"
    )

_MOE_DECODE_THRESHOLD = int(
    _os.environ.get("VLLM_MOE_LORA_DECODE_THRESHOLD", "0")
)

# Debug mode: set VLLM_MOE_LORA_DEBUG=1 to enable bounds checking and logging
_MOE_DEBUG = _os.environ.get("VLLM_MOE_LORA_DEBUG", "0") == "1"

# Need CUDA buffers if either phase uses CUDA
USE_BGMV_MOE_CUDA = _MOE_PREFILL_USE_CUDA or _MOE_DECODE_USE_CUDA


@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)

        self.lora_config = kwargs["lora_config"]
        self.max_loras = self.lora_config.max_loras

        # Compute captured LoRA counts for cudagraph specialization.
        captured_lora_counts = get_captured_lora_counts(
            self.max_loras, self.lora_config.specialize_active_lora
        )

        self.token_mapping_meta = LoRAKernelMeta.make(
            self.max_loras,
            max_num_batched_tokens,
            device=device,
            captured_lora_counts=captured_lora_counts,
        )

        # When speculative decoding is enabled, max_num_samples is
        # max_batches * (num_speculative_decoding_tokens + 1).
        # This line can be optimized by replacing max_num_batched_tokens
        # to  max_batches * (num_speculative_decoding_tokens + 1).
        self.prompt_mapping_meta = LoRAKernelMeta.make(
            self.max_loras,
            max_num_batched_tokens,
            device=device,
            captured_lora_counts=captured_lora_counts,
        )

        # MoE BGMV CUDA kernel buffers
        if USE_BGMV_MOE_CUDA:
            self._init_moe_bgmv_buffers(max_num_batched_tokens, device)

    def _init_moe_bgmv_buffers(self, max_num_batched_tokens, device):
        """Preallocate device buffers for MoE BGMV CUDA kernels.

        All per-call temporaries are allocated here once and reused via
        slicing, eliminating runtime torch.zeros / torch.arange / .to()
        overhead.
        """
        max_slices = 2    # W13 has 2 slices (gate+up), W2 has 1
        max_experts = 512  # Nemotron-Super has 512
        max_top_k = 64
        max_pairs = max_num_batched_tokens * max_top_k

        self._moe_w_ptr_buffer_shrink = torch.zeros(
            max_slices * max_experts, dtype=torch.int64, device=device)
        self._moe_w_ptr_buffer_expand = torch.zeros(
            max_slices * max_experts, dtype=torch.int64, device=device)
        self._moe_w_ptr_max_experts = max_experts
        self._moe_w_ptr_max_slices = max_slices

        self._moe_no_lora_flag_cpu = torch.tensor(
            [False], dtype=torch.bool, device="cpu")

        self._moe_slice_start_loc_buffer = torch.zeros(
            max_slices, dtype=torch.int64, device=device)
        self._moe_slice_start_loc_cpu = torch.zeros(
            max_slices, dtype=torch.int64, pin_memory=True)

        self._moe_seq_indices = torch.arange(
            max_pairs, dtype=torch.int64, device=device)
        self._moe_ones = torch.ones(
            max_pairs, dtype=torch.float32, device=device)
        self._moe_lora_indices_expanded = torch.empty(
            max_pairs, dtype=torch.int64, device=device)

        max_feat_out = 16384
        max_rank = self.lora_config.max_lora_rank
        self._moe_shrink_out_bf16 = torch.zeros(
            max_slices, max_pairs, max_rank,
            dtype=torch.bfloat16, device=device)
        self._moe_shrink_out_fp16 = torch.zeros(
            max_slices, max_pairs, max_rank,
            dtype=torch.float16, device=device)
        self._moe_y_accum = torch.zeros(
            max_num_batched_tokens * max_feat_out,
            dtype=torch.float32, device=device)
        self._moe_y_accum_max_feat = max_feat_out
        self._moe_expert_ids_i64 = torch.empty(
            max_pairs, dtype=torch.int64, device=device)
        self._moe_lora_indices_i64 = torch.full(
            (max_num_batched_tokens,), -1, dtype=torch.int64, device=device)
        self._moe_topk_weights_flat = torch.empty(
            max_pairs, dtype=torch.float32, device=device)
        self._moe_sorted_token_ids_div = torch.empty(
            max_pairs, dtype=torch.int64, device=device)

    def update_metadata(
        self,
        mapping: LoRAMapping,
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
        **kwargs,
    ):
        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

        # Update MoE BGMV no-lora flag
        if USE_BGMV_MOE_CUDA:
            self._moe_no_lora_flag_cpu[0] = torch.all(
                self.token_lora_indices == -1)

    def add_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        lora_shrink(
            x,
            lora_a_stacked,
            y,
            *self.token_mapping_meta.meta_args(
                x.size(0), self.lora_config.specialize_active_lora
            ),
            scale,
        )

    def add_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): If True, add LoRA output to y; if False, write
                LoRA-only output to y (used for dual-stream when base and LoRA
                run on different CUDA streams). Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)  # first dimension is the num slices

        lora_expand(
            x,
            lora_b_stacked,
            y,
            *self.token_mapping_meta.meta_args(
                num_tokens, self.lora_config.specialize_active_lora
            ),
            offset_start=offset_start,
            add_inputs=add_inputs,
        )

        y = y.view_as(y_org)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        lora_expand(
            x.unsqueeze(dim=0),
            (lora_b_stacked,),
            y,
            *self.token_mapping_meta.meta_args(
                x.size(0), self.lora_config.specialize_active_lora
            ),
            offset_start=0,
            add_inputs=add_inputs,
        )

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)
        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        r = lora_b_stacked[0].size(-1)
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        # Note: buffer is zeroed inside the shrink op
        buffer = torch.empty(
            (len(output_slices), x.size(0), r), dtype=torch.float32, device=x.device
        )

        add_inputs = kwargs.pop("add_inputs", True)

        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs,
        )
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            output_slices,
            add_inputs=add_inputs,
            **kwargs,
        )

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        # Note: buffer is zeroed inside the shrink op
        buffer = torch.empty((x.size(0), r), dtype=torch.float32, device=x.device)

        lora_shrink(
            x,
            [lora_a_stacked],
            buffer.unsqueeze(dim=0),
            *self.prompt_mapping_meta.meta_args(
                x.size(0), self.lora_config.specialize_active_lora
            ),
            scale,
        )

        lora_expand(
            buffer.unsqueeze(dim=0),
            [lora_b_stacked],
            y,
            *self.prompt_mapping_meta.meta_args(
                buffer.size(0), self.lora_config.specialize_active_lora
            ),
            add_inputs=True,
        )
        y = y.view_as(y_org)

    def moe_lora_align_block_size(
        self,
        topk_ids: torch.Tensor,
        num_tokens: int,
        block_size: int,
        num_experts: int,
        max_loras: int,
        adapter_enabled: torch.Tensor,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
        naive_block_assignment: bool = False,
        token_lora_mapping: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aligns tokens and experts into block-sized chunks for LoRA-based
        mixture-of-experts (MoE) execution.

        When `token_lora_mapping` is provided, it overrides the global mapping
        read from `self.token_mapping_meta`. This is how EP+LoRA injects the
        per-rank-local token→LoRA map after all-to-all dispatch.
        """
        (
            token_lora_mapping_meta,
            _,
            _,
            _,
            lora_ids,
            _,
            _,
        ) = self.token_mapping_meta.meta_args(
            num_tokens, self.lora_config.specialize_active_lora
        )
        if token_lora_mapping is None:
            token_lora_mapping = token_lora_mapping_meta
        # Under EP the caller passes local_num_experts but topk_ids carries
        # GLOBAL expert indices. The CUDA kernel uses num_experts to size
        # its bucketing table; with EP we must size by global_num_experts
        # so global topk_ids don't overflow. expert_map inside the kernel
        # then translates global→local so the output expert_ids are local
        # (mirrors the non-LoRA moe_align_block_size behavior).
        kernel_num_experts = (
            expert_map.numel() if expert_map is not None else num_experts
        )
        # Block-aligned metadata is needed for the Triton path.
        # The CUDA path needs naive (unpadded) expert_ids.
        # When both prefill and decode use CUDA, always force naive.
        if _MOE_PREFILL_USE_CUDA and _MOE_DECODE_USE_CUDA:
            naive_block_assignment = True

        if naive_block_assignment:
            expert_ids = topk_ids.reshape(-1)
            sorted_ids = None
            num_tokens_post_pad = None
        else:
            max_num_tokens_padded = topk_ids.numel() + kernel_num_experts * (
                block_size - 1
            )
            if pad_sorted_ids:
                max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
            if topk_ids.numel() < kernel_num_experts:
                max_num_tokens_padded = topk_ids.numel() * block_size
            sorted_ids = torch.empty(
                (max_loras * max_num_tokens_padded,),
                dtype=torch.int32,
                device=topk_ids.device,
            )
            max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
            # Expert ids are initialized to -1 so unused (lora, expert)
            # slots don't drive the LoRA Triton kernel into the wrong bucket.
            # The kernel overwrites only active slots.
            expert_ids = torch.full(
                (max_loras * max_num_m_blocks,),
                -1,
                dtype=torch.int32,
                device=topk_ids.device,
            )
            num_tokens_post_pad = torch.empty(
                (max_loras), dtype=torch.int32, device=topk_ids.device
            )

            ops.moe_lora_align_block_size(
                topk_ids,
                token_lora_mapping,
                kernel_num_experts,
                block_size,
                max_loras,
                max_num_tokens_padded,
                max_num_m_blocks,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                adapter_enabled,
                lora_ids,
                expert_map,
            )

        return token_lora_mapping, sorted_ids, expert_ids, num_tokens_post_pad

    # -----------------------------------------------------------------
    # CUDA BGMV MoE LoRA kernel support
    # -----------------------------------------------------------------

    def _ensure_moe_w_ptr(
        self,
        lora_weights: tuple[torch.Tensor, ...],
        w_ptr_buffer_1d: torch.Tensor,
        tag: str,
    ) -> tuple[torch.Tensor, int]:
        """Populate w_ptr and return (contiguous [num_slices, num_experts] view, lora_stride).

        Uses pre-allocated 1D buffer reshaped per call -- graph-capture safe.
        Works directly with the original weight layout [max_loras, num_experts, rank, feat]
        without creating a transposed copy.
        """
        from vllm.utils.flashinfer import _get_submodule

        bgmv_mod = _get_submodule("flashinfer.fused_moe.bgmv_moe")

        num_slices = len(lora_weights)
        num_experts = lora_weights[0].shape[1]

        w_ptr = w_ptr_buffer_1d[:num_slices * num_experts].view(
            num_slices, num_experts)

        lora_stride = 0
        for s in range(num_slices):
            lora_stride = bgmv_mod.fill_w_ptr(
                w_ptr, lora_weights[s], num_experts, s)

        return w_ptr, lora_stride

    def add_lora_fused_moe_cuda(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor | None,
        max_lora_rank: int,
        top_k_num: int,
        adapter_enabled: torch.Tensor,
        mul_routed_weight: bool = False,
        fully_sharded: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
    ):
        """
        CUDA kernel path for MoE LoRA using BGMV MoE kernels.

        Optimized for minimal per-call overhead:
        - w_ptr populated once and cached until weights change
        - Pre-allocated index/ones buffers reused via slicing
        - Thin kernel wrappers with no redundant Python logic
        """
        from vllm.utils.flashinfer import (
            flashinfer_bgmv_moe_shrink,
            flashinfer_bgmv_moe_expand,
        )

        _moe_shrink = flashinfer_bgmv_moe_shrink
        _moe_expand = flashinfer_bgmv_moe_expand

        y_orig = y
        num_slices = len(lora_a_stacked)
        rank = max_lora_rank
        is_w2 = mul_routed_weight

        if is_w2:
            num_tokens_orig = y.size(0)
            num_tokens = x.size(0)
        else:
            num_tokens = x.size(0)
            num_tokens_orig = num_tokens

        if _MOE_DEBUG:
            tag = "W2" if is_w2 else "W13"
            import sys
            print(f"[MOE_CUDA {tag}] num_tokens={num_tokens} "
                  f"num_tokens_orig={num_tokens_orig if is_w2 else num_tokens} "
                  f"x.shape={list(x.shape)} y.shape={list(y.shape)} "
                  f"num_slices={num_slices} rank={rank} "
                  f"top_k={top_k_num} "
                  f"expert_ids.shape={list(expert_ids.shape)} "
                  f"lora_a[0].shape={list(lora_a_stacked[0].shape)} "
                  f"lora_b[0].shape={list(lora_b_stacked[0].shape)}",
                  file=sys.stderr, flush=True)

        # ----- lora_indices -----
        (
            token_lora_mapping_meta, _, _, _, _, _, _,
        ) = self.token_mapping_meta.meta_args(
            num_tokens_orig, self.lora_config.specialize_active_lora
        )
        if token_lora_mapping is None:
            token_lora_mapping = token_lora_mapping_meta

        total_pairs = num_tokens_orig * top_k_num

        if is_w2:
            num_pairs = num_tokens
            self._moe_lora_indices_i64[:num_tokens_orig].copy_(
                token_lora_mapping)
            max_lora_id = self.lora_config.max_loras - 1
            lora_buf = self._moe_lora_indices_i64[:num_tokens_orig]
            lora_buf[(lora_buf < -1) | (lora_buf > max_lora_id)] = -1
            src = lora_buf
            torch.div(self._moe_seq_indices[:num_pairs], top_k_num,
                       rounding_mode='trunc',
                       out=self._moe_sorted_token_ids_div[:num_pairs])
            idx = self._moe_sorted_token_ids_div[:num_pairs]
            torch.index_select(src, 0, idx,
                               out=self._moe_lora_indices_expanded[:num_pairs])
            lora_indices = self._moe_lora_indices_expanded[:num_pairs]
        else:
            num_pairs = total_pairs
            self._moe_lora_indices_i64[:num_tokens_orig].copy_(
                token_lora_mapping)
            max_lora_id = self.lora_config.max_loras - 1
            lora_buf = self._moe_lora_indices_i64[:num_tokens_orig]
            lora_buf[(lora_buf < -1) | (lora_buf > max_lora_id)] = -1
            lora_indices = lora_buf

        # ----- sorted_token_ids -----
        if is_w2:
            sorted_token_ids_i64 = self._moe_seq_indices[:num_pairs]
        else:
            torch.div(self._moe_seq_indices[:num_pairs], top_k_num,
                       rounding_mode='trunc',
                       out=self._moe_sorted_token_ids_div[:num_pairs])
            sorted_token_ids_i64 = self._moe_sorted_token_ids_div[:num_pairs]

        # ----- expert_ids -----
        n_eid = min(expert_ids.view(-1).size(0), num_pairs)
        self._moe_expert_ids_i64[:n_eid].copy_(expert_ids.view(-1)[:n_eid])
        expert_ids_i64 = self._moe_expert_ids_i64[:num_pairs]

        # ----- topk_weights -----
        if mul_routed_weight:
            n_tw = min(topk_weights.view(-1).size(0), num_pairs)
            self._moe_topk_weights_flat[:n_tw].copy_(
                topk_weights.view(-1)[:n_tw])
            topk_weights_flat = self._moe_topk_weights_flat[:num_pairs]
        else:
            topk_weights_flat = self._moe_ones[:num_pairs]

        # ----- w_ptr -----
        w_ptr_a, lora_stride_a = self._ensure_moe_w_ptr(
            lora_a_stacked, self._moe_w_ptr_buffer_shrink, "shrink")
        w_ptr_b, lora_stride_b = self._ensure_moe_w_ptr(
            lora_b_stacked, self._moe_w_ptr_buffer_expand, "expand")

        # ----- Shrink -----
        if x.dtype == torch.bfloat16:
            shrink_buf = self._moe_shrink_out_bf16
        else:
            shrink_buf = self._moe_shrink_out_fp16
        flat_elems = num_slices * num_pairs * rank
        shrink_out = shrink_buf.view(-1)[:flat_elems].view(
            num_slices, num_pairs, rank)
        shrink_out.zero_()

        if _MOE_DEBUG:
            import sys
            num_experts_w = w_ptr_a.size(1)
            eid_max = expert_ids_i64[:num_pairs].max().item()
            eid_min = expert_ids_i64[:num_pairs].min().item()
            stid_max = sorted_token_ids_i64[:num_pairs].max().item()
            stid_min = sorted_token_ids_i64[:num_pairs].min().item()
            lid_max = lora_indices.max().item()
            lid_min = lora_indices.min().item()
            print(f"[MOE_CUDA PRE-SHRINK] "
                  f"shrink_out={list(shrink_out.shape)} "
                  f"x={list(x.shape)} "
                  f"w_ptr_a={list(w_ptr_a.shape)} "
                  f"num_experts_w={num_experts_w} "
                  f"expert_ids range=[{eid_min}, {eid_max}] "
                  f"sorted_token_ids range=[{stid_min}, {stid_max}] "
                  f"lora_indices range=[{lid_min}, {lid_max}] "
                  f"num_pairs={num_pairs} num_tokens={num_tokens}",
                  file=sys.stderr, flush=True)

        _moe_shrink(
            shrink_out, x, w_ptr_a,
            sorted_token_ids_i64, expert_ids_i64, lora_indices,
            lora_stride_a)

        if _MOE_DEBUG:
            torch.cuda.synchronize()
            import sys
            print(f"[MOE_CUDA] shrink completed OK", file=sys.stderr,
                  flush=True)

        # ----- Expand -----
        feat_out_per_slice = [
            lora_b_stacked[s].shape[2] for s in range(num_slices)
        ]
        total_feat_out = sum(feat_out_per_slice)

        loc = offset
        for s in range(num_slices):
            self._moe_slice_start_loc_cpu[s] = loc
            loc += feat_out_per_slice[s]
        self._moe_slice_start_loc_buffer[:num_slices].copy_(
            self._moe_slice_start_loc_cpu[:num_slices], non_blocking=True)

        y_accum = self._moe_y_accum[:num_tokens * total_feat_out].view(
            num_tokens, total_feat_out)
        y_accum.zero_()

        _moe_expand(
            y_accum, shrink_out, w_ptr_b,
            sorted_token_ids_i64, expert_ids_i64, topk_weights_flat,
            lora_indices,
            self._moe_slice_start_loc_buffer[:num_slices],
            feat_out_per_slice,
            lora_stride_b)

        # FlashInfer BGMV MoE kernels may launch on an internal stream.
        # Ensure the expand kernel completes before the caller accumulates
        # the result or the next layer zeros the shared buffer.
        torch.cuda.current_stream().synchronize()

        if _MOE_DEBUG:
            torch.cuda.synchronize()
            import sys
            print(f"[MOE_CUDA] expand completed OK", file=sys.stderr,
                  flush=True)

        # ----- Accumulate into caller's y -----
        if is_w2:
            y_orig.add_(y_accum.view(num_tokens_orig, top_k_num,
                                     total_feat_out))
        elif y_orig.dim() == 3 and y_accum.dim() == 2:
            y_orig.add_(y_accum.unsqueeze(1))
        else:
            y_orig.add_(y_accum)

    def _use_cuda_for_moe_lora(self, num_tokens: int) -> bool:
        """Decide whether to use CUDA BGMV or Triton for this call."""
        if _MOE_DECODE_THRESHOLD > 0:
            if num_tokens <= _MOE_DECODE_THRESHOLD:
                return _MOE_DECODE_USE_CUDA
            else:
                return _MOE_PREFILL_USE_CUDA
        return _MOE_PREFILL_USE_CUDA

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor | None,
        max_lora_rank: int,
        top_k_num: int,
        shrink_config,
        expand_config,
        adapter_enabled: torch.Tensor,
        mul_routed_weight=False,
        fully_sharded: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
    ):
        """
        Performs a fused forward computation for LoRA of MoE layer.

        Routes between CUDA BGMV and Triton kernels based on:
          - VLLM_MOE_LORA_PREFILL_BACKEND (default: cuda)
          - VLLM_MOE_LORA_DECODE_BACKEND  (default: cuda)
          - VLLM_MOE_LORA_DECODE_THRESHOLD (default: 0 = always)
        """
        # Skip entirely during cudagraph capture -- neither CUDA nor
        # Triton MoE LoRA ops are capture-safe (they allocate GPU
        # memory or use data-dependent tensor sizes).  During capture
        # the model runs with dummy data and no real LoRA adapters,
        # so the LoRA contribution is zero.
        if torch.cuda.is_current_stream_capturing():
            return

        # MoE LoRA kernels (both CUDA BGMV and Triton) support bf16/fp16 only.
        # FP8 quantized models may pass raw FP8 activations here; cast to bf16.
        if x.dtype not in (torch.bfloat16, torch.float16):
            x = x.to(torch.bfloat16)

        num_tokens = x.size(0)

        if self._use_cuda_for_moe_lora(num_tokens):
            return self.add_lora_fused_moe_cuda(
                y, x, lora_a_stacked, lora_b_stacked,
                topk_weights, sorted_token_ids, expert_ids,
                num_tokens_post_padded, max_lora_rank, top_k_num,
                adapter_enabled,
                mul_routed_weight=mul_routed_weight,
                fully_sharded=fully_sharded,
                offset=offset,
                token_lora_mapping=token_lora_mapping,
            )

        # Triton path (fallback)
        (
            token_lora_mapping_meta, _, _, _,
            lora_ids, _, num_active_loras,
        ) = self.token_mapping_meta.meta_args(
            x.size(0), self.lora_config.specialize_active_lora
        )
        if token_lora_mapping is None:
            token_lora_mapping = token_lora_mapping_meta
        fused_moe_lora(
            y, x, lora_a_stacked, lora_b_stacked,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, token_lora_mapping,
            max_lora_rank, top_k_num, lora_ids, num_active_loras,
            adapter_enabled,
            shrink_config.get("BLOCK_SIZE_M", 64),
            shrink_config.get("BLOCK_SIZE_N", 64),
            shrink_config.get("BLOCK_SIZE_K", 32),
            shrink_config.get("GROUP_SIZE_M", 8),
            shrink_config.get("NUM_WARPS", 4),
            shrink_config.get("NUM_STAGES", 3),
            shrink_config.get("SPLIT_K", 1),
            expand_config.get("BLOCK_SIZE_M", 64),
            expand_config.get("BLOCK_SIZE_N", 64),
            expand_config.get("BLOCK_SIZE_K", 32),
            expand_config.get("GROUP_SIZE_M", 8),
            expand_config.get("NUM_WARPS", 4),
            expand_config.get("NUM_STAGES", 3),
            expand_config.get("SPLIT_K", 1),
            mul_routed_weight, fully_sharded, offset,
        )

    def add_lora_w13(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        num_tokens: int,
        top_k_num: int,
        max_loras: int,
        adapter_enabled: torch.Tensor,
        local_num_experts: int,
        top_k: int,
        num_slices: int,
        fully_sharded: bool,
        use_tuned_config: bool,
        token_lora_mapping: torch.Tensor | None = None,
        add_inputs: bool = True,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        import functools

        from vllm.lora.layers.utils import try_get_optimal_moe_lora_config
        from vllm.lora.ops.triton_ops.utils import (
            _normalize_lora_config_keys,
            get_lora_op_configs,
        )
        from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str

        config_dtype = _get_config_dtype_str(
            dtype=x.dtype,
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
        )
        max_lora_rank = lora_a_stacked[0].shape[-2]

        if use_tuned_config:
            shrink_config = get_lora_op_configs(
                op_type="fused_moe_lora_w13_shrink",
                max_loras=max_loras,
                batch=num_tokens,
                hidden_size=x.shape[-1],
                rank=max_lora_rank,
                num_slices=num_slices,
                moe_intermediate_size=lora_b_stacked[0].shape[-2],
            )
            expand_config = get_lora_op_configs(
                op_type="fused_moe_lora_w13_expand",
                max_loras=max_loras,
                batch=num_tokens,
                hidden_size=x.shape[-1],
                rank=max_lora_rank,
                num_slices=num_slices,
                moe_intermediate_size=lora_b_stacked[0].shape[-2],
            )
        else:
            get_config = functools.partial(
                try_get_optimal_moe_lora_config,
                w1_shape=w1.shape,
                w2_shape=w2.shape,
                rank=max_lora_rank,
                top_k=top_k,
                dtype=config_dtype,
                M=num_tokens,
            )
            shrink_config = get_config(op_type="fused_moe_lora_w13_shrink")
            expand_config = get_config(op_type="fused_moe_lora_w13_expand")

        shrink_config = _normalize_lora_config_keys(shrink_config)
        expand_config = _normalize_lora_config_keys(expand_config)

        SPARSITY_FACTOR = 8
        naive_block_assignment = (
            expert_map is None
            and num_tokens * top_k * SPARSITY_FACTOR <= local_num_experts * max_loras
        )

        (
            token_lora_mapping,
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
        ) = self.moe_lora_align_block_size(
            topk_ids,
            num_tokens,
            int(shrink_config.get("BLOCK_SIZE_M") or 64),
            local_num_experts,
            max_loras,
            adapter_enabled,
            expert_map,
            naive_block_assignment=naive_block_assignment,
            token_lora_mapping=token_lora_mapping,
        )

        _sorted = sorted_token_ids_lora
        _eids = expert_ids_lora
        if _sorted is not None:
            _eids = _eids.view(max_loras, -1)
            _sorted = _sorted.view(max_loras, -1)

        self.add_lora_fused_moe(
            y.view(-1, top_k_num, y.shape[-1]),
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            _sorted,
            _eids,
            num_tokens_post_padded_lora,
            max_lora_rank,
            top_k,
            shrink_config,
            expand_config,
            adapter_enabled,
            fully_sharded=fully_sharded,
            token_lora_mapping=token_lora_mapping,
        )

        return (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        )

    def add_lora_w2(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids_lora: torch.Tensor | None,
        expert_ids_lora: torch.Tensor | None,
        num_tokens_post_padded_lora: torch.Tensor | None,
        token_lora_mapping: torch.Tensor | None,
        num_tokens: int,
        w1: torch.Tensor,
        w2: torch.Tensor,
        top_k_num: int,
        max_loras: int,
        adapter_enabled: torch.Tensor,
        top_k: int,
        fully_sharded: bool,
        tp_rank: int,
        use_tuned_config: bool,
        add_inputs: bool = True,
    ) -> None:
        import functools

        from vllm.lora.layers.utils import try_get_optimal_moe_lora_config
        from vllm.lora.ops.triton_ops.utils import (
            _normalize_lora_config_keys,
            get_lora_op_configs,
        )
        from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str

        config_dtype = _get_config_dtype_str(
            dtype=x.dtype,
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
        )
        max_lora_rank = lora_a_stacked[0].shape[-2]

        if use_tuned_config:
            shrink_config = get_lora_op_configs(
                op_type="fused_moe_lora_w2_shrink",
                max_loras=max_loras,
                batch=num_tokens,
                hidden_size=y.shape[-1],
                rank=max_lora_rank,
                num_slices=1,
                moe_intermediate_size=lora_a_stacked[0].shape[-1],
            )
            expand_config = get_lora_op_configs(
                op_type="fused_moe_lora_w2_expand",
                max_loras=max_loras,
                batch=num_tokens,
                hidden_size=y.shape[-1],
                rank=max_lora_rank,
                num_slices=1,
                moe_intermediate_size=lora_a_stacked[0].shape[-1],
            )
        else:
            get_config = functools.partial(
                try_get_optimal_moe_lora_config,
                w1_shape=w1.shape,
                w2_shape=w2.shape,
                rank=max_lora_rank,
                top_k=top_k,
                dtype=config_dtype,
                M=num_tokens,
            )
            shrink_config = get_config(op_type="fused_moe_lora_w2_shrink")
            expand_config = get_config(op_type="fused_moe_lora_w2_expand")

        shrink_config = _normalize_lora_config_keys(shrink_config)
        expand_config = _normalize_lora_config_keys(expand_config)

        _sorted = sorted_token_ids_lora
        _eids = expert_ids_lora
        if _sorted is not None:
            assert _eids is not None
            _eids = _eids.view(max_loras, -1)
            _sorted = _sorted.view(max_loras, -1)

        # w2_lora_b shape[-2] is hidden_size // tp_size when fully_sharded
        shard_size = lora_b_stacked[0].shape[-2]
        offset = shard_size * tp_rank if fully_sharded else 0

        self.add_lora_fused_moe(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            _sorted,
            _eids,
            num_tokens_post_padded_lora,
            max_lora_rank,
            top_k,
            shrink_config,
            expand_config,
            adapter_enabled,
            True,  # mul_routed_weight
            fully_sharded=fully_sharded,
            offset=offset,
            token_lora_mapping=token_lora_mapping,
        )
