# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Tuple, Union
import math

# Third Party
from flashinfer import VariableBlockSparseAttentionWrapper
from flashinfer.page import block_sparse_indices_to_vector_sparse_offsets
from flashinfer.utils import (
    TensorLayout,
    _check_pos_encoding_mode,
    check_shape_dtype_device,
    device_support_pdl,
)
from vllm.attention import Attention
from vllm.v1.attention.backends.flashinfer import FlashInferImpl
import flashinfer
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashInferSparseMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.compute.attention.metadata import LMCAttnMetadata


# NOTE(Jiayi): This flashinfer version is 0.3.1.
class HackBSAWrapper(VariableBlockSparseAttentionWrapper):
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        This is adapted from https://github.com/flashinfer-ai/flashinfer/blob/cc5ab77370dd9a489357a47e34315d9a8f3ad5fb/flashinfer/sparse.py#L1073

        Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(qo_len, num_qo_heads, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(kv_len, num_kv_heads, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(kv_len, num_kv_heads, head_dim)``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be
            allocated internally.
        return_lse : bool
            Whether to return the log-sum-exp of attention logits
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape:
            ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """

        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)

        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        # 1 denotes page size
        k = k.reshape(-1, 1, *k.shape[-2:])
        v = v.reshape(-1, 1, *v.shape[-2:])

        stride_block = k.stride(0)
        stride_n = k.stride(1)

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q, dtype=self._o_dtype)
        else:
            check_shape_dtype_device(out, q.shape, self._o_dtype, q.device, "out")

        if self._backend == "fa3":
            if (
                self._vector_sparse_indices_buffer.numel()
                <= self._paged_kv_indices_buf.numel()
            ):
                raise ValueError(
                    "_vector_sparse_indices_buffer is not large enough. "
                    "Please increase the buffer size."
                )

            sparse_indices = block_sparse_indices_to_vector_sparse_offsets(
                self._paged_kv_indices_buf,
                self._paged_kv_indptr_buf,
                self._vector_sparse_indices_buffer,  # output
                self._vector_sparse_indptr_buffer,
                self._kv_lens_buffer,
                stride_block // stride_n,
                1,  # stride_n // stride_n
                1,  # block_size
            )
            sparse_indptr = self._vector_sparse_indptr_buffer
        else:
            sparse_indices = self._paged_kv_indices_buf
            sparse_indptr = self._paged_kv_indptr_buf

        self._cached_module.paged_run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k,
            v,
            self._qo_indptr,
            sparse_indptr,
            sparse_indices,
            self._paged_kv_last_page_len,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            -1,  # window_left
            enable_pdl,
            # ADDITIONAL_FUNC_PARAMS
            # Not supported yet
            None,  # packed_mask_buf
            None,  # mask_indptr_buf
            None,  # alibi_slopes_buf
            None,
            None,
            None,
            logits_soft_cap,
            sm_scale,
            None,  # scale_q
            None,  # scale_k
            None,  # scale_v
            rope_scale,
            rope_theta,
            0,  # token_pos_in_items_len
        )

        return (out, lse) if return_lse else out


class LMCFlashInferSparseBackend(AttentionInterface):
    """
    FlashAttention backend for LMCache.
    This backend uses the FlashAttention implementation
    for efficient attention computation.
    """

    # Workspace buffer size in bytes (128 MiB)
    _WORKSPACE_BUFFER_SIZE_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        vllm_attn: Attention,
    ):
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl: FlashInferImpl = vllm_attn.impl

        idx = torch_dev.current_device()
        self.device = torch.device(f"{torch_device_type}:{idx}")

        self.workspace_buffer = torch.empty(
            self._WORKSPACE_BUFFER_SIZE_BYTES, dtype=torch.uint8, device=self.device
        )

        self.num_qo_heads = self.vllm_attn_impl.num_heads
        self.num_kv_heads = self.vllm_attn_impl.num_kv_heads
        self.head_dim = self.vllm_attn_impl.head_size

        self.sm_scale = self.vllm_attn_impl.scale
        self.window_left = self.vllm_attn_impl.window_left
        self.logits_soft_cap = self.vllm_attn_impl.logits_soft_cap

        self.k_scale = vllm_attn._k_scale_float
        self.v_scale = vllm_attn._v_scale_float

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, LMCFlashInferSparseMetadata)
        is_causal = attn_metadata.is_causal
        if is_causal:
            output = flashinfer.prefill.single_prefill_with_kv_cache(
                q=query,
                k=key,
                v=value,
                scale_q=None,
                scale_k=self.k_scale,
                scale_v=self.v_scale,
                o_dtype=query.dtype,
                custom_mask=None,
                packed_custom_mask=None,
                causal=True,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                use_fp16_qk_reduction=False,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                rope_scale=None,
                rope_theta=None,
                backend="auto",
                return_lse=False,
            )
        else:
            output = attn_metadata.wrapper.run(query, key, value, output)
        return output

    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> LMCFlashInferSparseMetadata:
        """
        Initialize non-sparse attention metadata first.
        """

        wrapper = HackBSAWrapper(self.workspace_buffer)
        seq_len = len(input_ids)

        # TODO(Jiayi): remove this hardcode
        sparse_blk_row_size = 32
        sparse_blk_col_size = 32

        num_block_col = seq_len // sparse_blk_col_size
        last_col_len = seq_len % sparse_blk_col_size
        block_col_sizes = torch.tensor(
            [sparse_blk_col_size] * num_block_col, device=self.device
        )
        block_col_sizes[-1] += last_col_len
        return LMCFlashInferSparseMetadata(
            wrapper,
            seq_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            block_col_sizes,
            sparse_blk_row_size,
            sparse_blk_col_size,
            is_causal=True,
        )
