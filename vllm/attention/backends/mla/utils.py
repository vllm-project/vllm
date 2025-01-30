from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import (AttentionLayer,
                                              AttentionMetadata,
                                              MLAAttentionImpl, T)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.vllm_flash_attn import flash_attn_varlen_func


@dataclass
class MLACommonMetadata(AttentionMetadata):
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor


class MLACommonImpl(MLAAttentionImpl[T], Generic[T]):
    """
    Common class for implementing repeated parts 
    
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).
    
    Deepseek's MLA attention works the following way:
    * Use a single latent vector to represent the entire KV cache.
    * The attention "simulates" a multi-head attention, while the compute is
      similar to multi-query attention.
    * The dataflow is as follows,

        * B: batch/sequence length
        * H: hidden size
        * N: number of attention heads
        * Lq: latent dimension for Q
        * Lkv: latent dimension for K/V
        * P: nope dimension, P+R is the actual head_dim in common attention.
        * R: rope dimension, this slide of the head_dim goes through rope.
        * V: V head dim.
        * kv_c: latent/compressed KV
        * q_c: latent/compressed Q
        
        #
        # Outside the MLA attention backend
        #

        1. The hidden states (B, H) are projected down into cq (B, Lq) and
           kv_c_k_pe (B, Lkv+R).
        2. The kv_c_k_pe is split into kv_c (B, Lkv) and k_pe (B, R). cq
           and kv_c are normalized.
        
        #
        # Inside the MLA attention backend
        #

        * if prefill:
        
        3. The q_c is then projected up into the multi-head version. 
           * q_c goes from (B, Lq) to (B, N, (P+R)), which is split into q_nope 
             (B, N, P) and q_pe (B, N, R). 
        4. q_pe, k_pe are then passed through rotary embeddings.
        5. kv_c and k_pe are concatenated and inserted into the cache
        6. The kv_c is then projected up into the multi-head version. 
           * kv_c goes from (B, Lkv) to (B, N, (P+V)) which has the nope 
             dimensions for K and V, which is split into k_nope (B, N, P) 
             and v (B, N, V).
        7. q (B, N, (P+R)) and k (B, N, (P+R)) matrices are assembled from
           q_nope, q_pe, k_nope, k_pe.
        8. Attention is computued with q, k, v.
        9. The attention computation returns (B, N, V), which is projected back
           to (B, H) using out projection.

        * if decode:

        3. Here's the change, we do not perform up the full up projection for
           q_c, and there is no up projection at all for kv_c. This is
           achieved by the technique of "weight absorption". The paper says
           "Fortunately, due to the associative law of matrix multiplication,
           we can absorb WUK into WUQ, and WUV into WO"
           * The q up projection turns (B, Lq) into (B, N, (P+R)), we split it
             into W_UQ (Lq, N, P) and W_QR (Lq, N, R).
           * The kv_c up projection turns (B, Lkv) into (B, N, (P+V)), we split
             it into W_UK (Lkv, N, P) and W_UV (Lkv, N, V).
           * The out projection shape W_O (N*V, H) turns (B, N, V) into (B, H).
           * We can precompute the product of W_UQ and W_UK into
             W_UQ_UK (Lq, N, Lkv), which is possible due to QK^T operation in
             attention.
           * We can precompute the product of W_UV and W_O into
             W_UV_O (N, Lkv, H), which is possible due to V@O as the
             "epilogue" of attention
        4. We still need to compute q_pe (B, N, R) by applying W_QR to q_latent.
        5. q_pe, k_pe are then passed through rotary embeddings.
        6. kv_c and k_pe are concatenated and inserted into the cache
        7. By applying W_UQ_UK to q_latent, we have the new q_nope of shape
           (B, N, Lkv).
        8. q (B, N, (Lkv+R)), k (B, (Lkv+R)) are assembled from q_nope, q_pe,
           kv_a, k_pe. v (B, Lkv) is exactly the same vector as kv_a.
        9. The attention is computed with q, k, v. Note that we just performed
           a MQA attention with (LKv+R) as our head dim.
        10. The KV cache is updated using the new entries k (B, N, (Lkv+R)),
           which included the v and rope values.
        11. The attention computation returns (B, N, Lkv), which is projected
           back to (B, H) using W_UV_O.

    From @tsu-bin's calculation, we only want to use the absorption technique
    for decode. The prefill algorithm should still use the up-projected MHA
    for less flops and memory usage.
    
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]],
        logits_soft_cap: Optional[float],
        attn_type: str,
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        rotary_emb: RotaryEmbedding,
        # q_proj should be q_b_proj if q_lora_rank is not None, but from an
        # attention backend perspective we rely on the layer to pass in the
        # correct matrix
        q_proj: ColumnParallelLinear,
        kv_b_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        self.rotary_emb = rotary_emb
        self.q_proj = q_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj

    def _v_up_proj_and_o_proj(self, x):
        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            return self.o_proj_absorbed(
                x.reshape(-1, self.num_heads * self.kv_lora_rank))[0]
        else:
            x = torch.einsum("bnl,lnv->bnv", x, self.W_UV)
            return self.o_proj(x.reshape(-1,
                                         self.num_heads * self.v_head_dim))[0]

    def _q_proj_and_k_up_proj(self, x):
        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            return torch.matmul(x, self.W_Q_UK)\
                .view(-1, self.num_heads, self.kv_lora_rank)
        else:
            x = torch.matmul(x, self.W_Q)\
                .view(-1, self.num_heads, self.qk_nope_head_dim)
            return torch.einsum("bnp,lnp->bnl", x, self.W_UK)\
                .view(-1, self.num_heads, self.kv_lora_rank)

    def process_weights_after_loading(self):
        kv_b_proj_weight = self.kv_b_proj.weight.T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_proj = self.q_proj.weight.T\
                .view(-1, self.num_heads, self.qk_head_dim)

        # can be W_Q or W_UQ depending q_lora_rank, the former if
        # q_lora_rank is None, the latter otherwise. From the Attention backend
        # perspective though we call these both W_Q and rely on the layer
        # to pass in the correct matrix
        W_Q = q_proj[..., :self.qk_nope_head_dim]
        self.W_QR = q_proj[..., self.qk_nope_head_dim:]\
            .flatten(start_dim=1).contiguous()

        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            #
            # Perform matrix-absorption following
            #     https://github.com/flashinfer-ai/flashinfer/pull/551
            # for decode, as a result we end up with absorbed weights for decode
            # and another copy of raw weights for prefill.
            #
            self.W_UK, self.W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # We absorb `W_UK` into `W_Q` resulting in either W_Q_UK or W_UQ_UK
            # depending q_lora_rank, the former if q_lora_rank is None, the
            # latter otherwise
            # basically if q_lora_rank is none we are absorbing into q_proj
            # instead of UQ
            self.W_Q_UK = torch.einsum("qnd,lnd -> qnl", W_Q, W_UK)\
                .flatten(start_dim=1).contiguous()

            W_O = self.o_proj.weight\
                .view(-1, self.num_heads, self.v_head_dim)
            self.W_UV_O = torch.einsum("lnd,hnd -> nlh", W_UV, W_O)\
                .flatten(start_dim=0, end_dim=1).contiguous()

            tp_size = get_tensor_model_parallel_world_size()
            self.o_proj_absorbed = RowParallelLinear(
                self.W_UV_O.shape[0] * tp_size,
                self.W_UV_O.shape[1],
                bias=False,
                # TODO(lucas) figure out how to properly forward quant_method
                #quant_config=self.o_proj.quant_method,
            )

            self.o_proj_absorbed.weight = torch.nn.Parameter(self.W_UV_O.T)
        else:
            self.W_UV = W_UV
            self.W_UK = W_UK
            self.W_Q = W_Q.flatten(start_dim=1)

    @abstractmethod
    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError(
                "output is not yet supported for MLAImplBase")

        is_decode = attn_metadata.decode_metadata is not None
        is_prefill = attn_metadata.prefill_metadata is not None

        if (is_decode and is_prefill):
            raise NotImplementedError(
                "chunked prefill is not supported for MLAImplBase")

        # Restore head dim (for rotary embedding)
        k_pe = k_pe.unsqueeze(1)
        assert hasattr(attn_metadata, "input_positions")

        if is_decode:
            q_nope = self._q_proj_and_k_up_proj(hidden_states_or_q_c)
            q_pe = torch.matmul(hidden_states_or_q_c, self.W_QR)\
                .view(-1, self.num_heads, self.qk_rope_head_dim)
            q_pe, k_pe = \
                self.rotary_emb(attn_metadata.input_positions, q_pe, k_pe)
        else:
            assert is_prefill
            q = self.q_proj(hidden_states_or_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)

            # TODO(lucas): there must be a nicer way to write this line
            q[..., self.qk_nope_head_dim:], k_pe = \
                self.rotary_emb(
                    attn_metadata.input_positions,
                    q[..., self.qk_nope_head_dim:], k_pe)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if attn_metadata.prefill_metadata is not None:
            return self._forward_prefill(q, k_c_normed, k_pe, attn_metadata)

        if attn_metadata.decode_metadata is not None:
            return self._forward_decode(q_nope, q_pe, kv_cache, attn_metadata)

    # Optional common flash-attn based prefill
    def _forward_prefill_flash(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        seq_start_loc: torch.Tensor,
        max_prefill_seq_len: int,
    ) -> torch.Tensor:

        kv_nope = self.kv_b_proj(k_c_normed)[0]\
            .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)

        attn_output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v_padded,
            cu_seqlens_q=seq_start_loc,
            cu_seqlens_k=seq_start_loc,
            max_seqlen_q=max_prefill_seq_len,
            max_seqlen_k=max_prefill_seq_len,
            softmax_scale=self.scale,
            causal=True,
        )
        attn_output = attn_output\
            .view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(-1, self.num_heads * v.shape[-1])

        return self.o_proj(attn_output)[0]
