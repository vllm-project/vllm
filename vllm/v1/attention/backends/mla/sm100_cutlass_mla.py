# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch

import vllm._custom_ops as ops
from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonImpl,
                                                   MLACommonMetadata)

logger = init_logger(__name__)


class Sm100CutlassMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "SM100_CUTLASS_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["Sm100CutlassMLAImpl"]:
        return Sm100CutlassMLAImpl


workspace_size = 1024 * 1024 * 1024  # 1GB
g_workspace = torch.zeros(workspace_size, device="cuda", dtype=torch.uint8)


class Sm100CutlassMLAImpl(MLACommonImpl[MLACommonMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "CutlassMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "CutlassMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "CutlassMLA V1 with FP8 KV cache not yet supported")

        # TODO: Ensure the kernel does not hang.
        #       If it happens, then set _num_kv_splits to 1
        self._num_kv_splits = 8

        # This is a workaround to set a workspace for the sm100 cutlass mla.
        # We do need to execute sm100_cutlass_mla_get_workspace_size(..)
        # since it calls set_kv_split() internally, however, the workspace
        # size we use is large and shared over all executions.
        # TODO: Refactor this code to a place that can observe params
        #       more fine-grained
        bs = 128  # dummy
        PAGE_SIZE = 128  # dummy
        max_seqlen_pad = 131072  # dummy
        sm_count = 0  # dummy
        _ = ops.sm100_cutlass_mla_get_workspace_size(
            max_seqlen_pad * PAGE_SIZE,
            bs,
            sm_count,
            num_kv_splits=self._num_kv_splits)

        self._workspace = g_workspace

    def sm100_cutlass_mla_decode(
            self,
            q_nope: torch.Tensor,
            q_pe: torch.Tensor,
            kv_c_and_k_pe_cache: torch.Tensor,
            seq_lens: torch.Tensor,
            page_table: torch.Tensor,
            workspace: torch.Tensor,
            sm_scale: float,
            num_kv_splits:
        int = 1,  # Set to 1 to avoid cuda_graph issue by default.
    ) -> torch.Tensor:
        assert (q_nope.ndim == 3
                ), f"q_nope must be a 3D tensor, but got {q_nope.ndim}"
        assert (
            q_pe.ndim == 3), f"q_pe must be a 3D tensor, but got {q_pe.ndim}"
        assert (
            kv_c_and_k_pe_cache.ndim == 3
        ), "kv_c_and_k_pe_cache must be a 3D tensor, but got {}".format(
            kv_c_and_k_pe_cache.ndim)

        B_q, H, D_q_nope = q_nope.shape
        B_q_2, H_2, D_q_pe = q_pe.shape
        assert (B_q == B_q_2) and (H == H_2)

        _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

        D_latent = 512
        D_rope = 64
        assert D_q_nope == D_latent
        assert D_q_pe == D_rope
        assert D_ckv == D_latent + D_rope

        MAX_HEADS = 128
        assert H <= MAX_HEADS, f"H must be <= {MAX_HEADS}, but got {H}"
        if H < MAX_HEADS:
            q_nope_padded = q_nope.new_empty((B_q, MAX_HEADS, D_q_nope))
            q_nope_padded[:, :H] = q_nope
            q_nope = q_nope_padded

            q_pe_padded = q_pe.new_empty((B_q, MAX_HEADS, D_q_pe))
            q_pe_padded[:, :H] = q_pe
            q_pe = q_pe_padded

        assert len(page_table.shape) == 2
        B_block_table, block_num = page_table.shape
        assert B_block_table == B_q
        assert (block_num
                > 0), f"block num must be greater than 0, got {block_num}"
        assert block_num % (128 / PAGE_SIZE) == 0

        # TODO(kaixih@nvidia): support fp8
        assert q_nope.dtype in (
            torch.float16,
            torch.bfloat16,
        ), f"q_nope.dtype needs to be fp16 or bf16 but got {q_nope.dtype}."
        assert q_nope.dtype == q_pe.dtype == kv_c_and_k_pe_cache.dtype
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens.dtype needs to be int32 but got {seq_lens.dtype}."
        assert (
            page_table.dtype == torch.int32
        ), f"page_table.dtype needs to be int32 but got {page_table.dtype}."

        out = q_nope.new_empty((B_q, MAX_HEADS, D_latent))

        ops.sm100_cutlass_mla_decode(
            out,
            q_nope,
            q_pe,
            kv_c_and_k_pe_cache,
            seq_lens,
            page_table,
            workspace,
            sm_scale,
            num_kv_splits,
        )
        return out[:, :H].contiguous()

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Cutlass MLA not yet supported")

        # Run MLA
        # Clone q_nope and q_pe to make sure strides computation is correct.
        # TODO: Check if we really need it
        q_nope = q_nope.clone()
        q_pe = q_pe.clone()

        o = self.sm100_cutlass_mla_decode(q_nope, q_pe, kv_c_and_k_pe_cache,
                                          attn_metadata.decode.seq_lens,
                                          attn_metadata.decode.block_table,
                                          self._workspace, self.scale,
                                          self._num_kv_splits)

        return self._v_up_proj(o)
