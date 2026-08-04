# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING

try:
    # Third Party
    from vllm.attention import Attention
except ImportError:
    # vLLM >= 0.17 moved the Attention class.
    from vllm.model_executor.layers.attention.attention import Attention

# Third Party
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata
from lmcache.v1.compute.attention.utils import infer_attn_func_from_vllm

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.compute.attention.metadata import LMCAttnMetadata


class LMCFlashAttnBackend(AttentionInterface):
    """
    FlashAttention backend for LMCache.
    This backend uses the FlashAttention implementation
    for efficient attention computation.
    """

    def __init__(
        self,
        vllm_attn: Attention,
    ):
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl: FlashAttentionImpl = vllm_attn.impl

        # TODO(Jiayi): remove this hardcode
        self.aot_schedule = False

        idx = torch_dev.current_device()
        self.device = torch.device(f"{torch_device_type}:{idx}")
        self.flash_attn_varlen_func, self.get_scheduler_metadata = (
            infer_attn_func_from_vllm(torch_device_type)
        )

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, LMCFlashAttnMetadata)

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        # TODO(Jiayi): Figure out how to use aot_schedule.
        scheduler_metadata = self._schedule(
            batch_size=1,  # NOTE(Jiayi): Assuming batch size is 1,
            # since we are processing request by request.
            cu_query_lens=cu_seqlens_q,
            max_query_len=max_seqlen_q,
            seqlens=seqused_k,
            max_seq_len=max_seqlen_k,
            causal=True,  # Assuming causal attention
        )

        self.flash_attn_varlen_func(
            q=query,  # contiguous
            k=key,  # contiguous
            v=value,  # contiguous
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            # seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.vllm_attn_impl.scale,
            causal=True,
            alibi_slopes=self.vllm_attn_impl.alibi_slopes,
            window_size=self.vllm_attn_impl.sliding_window,
            block_table=None,
            softcap=self.vllm_attn_impl.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=self.vllm_attn_impl.vllm_flash_attn_version,
            q_descale=self.vllm_attn._q_scale.expand(descale_shape),
            k_descale=self.vllm_attn._k_scale.expand(descale_shape),
            v_descale=self.vllm_attn._v_scale.expand(descale_shape),
        )

        return output

    def _schedule(
        self, batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
    ):
        if self.aot_schedule:
            return self.get_scheduler_metadata(
                batch_size=batch_size,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                cache_seqlens=seqlens,
                num_heads_q=self.vllm_attn_impl.num_heads_q,
                num_heads_kv=self.vllm_attn_impl.num_heads_kv,
                headdim=self.vllm_attn_impl.headdim,
                page_size=self.vllm_attn_impl.block_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
                window_size=self.vllm_attn_impl.aot_sliding_window,
            )
        return None

    def init_attn_metadata(
        self,
        input_ids: torch.tensor,
        **kwargs,
    ) -> LMCFlashAttnMetadata:
        seq_len = input_ids.shape[0]
        device = input_ids.device
        return LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor([seq_len], device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_query_len=seq_len,
            max_seq_len=seq_len,
        )
