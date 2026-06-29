# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oscar attention backend for vLLM (pure PyTorch).

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K/V and store into combined cache slot.
Decode:  Dequantize from compressed cache, compute standard attention,
         un-rotate output.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_packed_size
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.oscar_ops import (
    build_rotation_matrix,
    dequant_and_unrotate,
    oscar_decode_attention,
    store_kv_to_cache,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


class OscarAttentionBackend(AttentionBackend):
    """Attention backend using Oscar KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "oscar_int2",
    ]

    @staticmethod
    def get_name() -> str:
        return "OSCAR"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["OscarAttentionImpl"]:
        return OscarAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["OscarMetadataBuilder"]:
        return OscarMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "oscar_int2",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension."""
        oscar_config = OscarConfig.from_cache_dtype(
            cache_dtype_str, head_size
        )
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            oscar_config.slot_size_aligned,
        )

    @classmethod
    def supports_kv_cache_dtype(
        cls, kv_cache_dtype: CacheDType | None
    ) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("oscar_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Pure PyTorch — any head_dim divisible by 4 works.
        return head_size > 0 and head_size % 4 == 0


@dataclass
class OscarMetadata(AttentionMetadata):
    """Metadata for Oscar attention."""

    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None


class OscarMetadataBuilder(AttentionMetadataBuilder[OscarMetadata]):
    """Builds OscarMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device
        )
        self._init_reorder_batch_threshold(
            1, supports_spec_as_decode=False
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> OscarMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self, common_prefix_len, common_attn_metadata, fast_build=False
    ):
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, _, num_decode_tokens, _ = (
            split_decodes_and_prefills(
                cam, decode_threshold=self.reorder_batch_threshold
            )
        )
        return OscarMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
        )


class OscarAttentionImpl(AttentionImpl["OscarMetadata"]):
    """Oscar attention implementation (pure PyTorch)."""

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else num_heads
        )
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.oscar_config = OscarConfig.from_cache_dtype(
            kv_cache_dtype, head_size
        )
        self.fa_version = get_flash_attn_version(head_size=head_size)

    # -------------------------------------------------------------- #
    #  One-time setup of rotation matrix on layer                      #
    # -------------------------------------------------------------- #

    def _ensure_on_device(self, layer: Any, device: torch.device) -> None:
        """Lazily build and cache the rotation matrix on the layer."""
        if not hasattr(layer, "_oscar_cached"):
            D = self.head_size
            R = build_rotation_matrix(D, device)
            layer._oscar_R = R
            layer._oscar_RT = R.T.contiguous()
            layer._oscar_cached = True

    # -------------------------------------------------------------- #
    #  KV cache update (called before forward)                         #
    # -------------------------------------------------------------- #

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)
        R = layer._oscar_R

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)

        # Rotate both K and V before quantization
        R_dtype = R.to(k.dtype)
        k_rot = (k.float() @ R_dtype.float()).to(k.dtype)
        v_rot = (v.float() @ R_dtype.float()).to(v.dtype)

        store_kv_to_cache(
            k_rot,
            v_rot,
            kv_cache,
            slot_mapping,
            clip_ratio_k=OscarConfig.get_k_clip_ratio(),
            clip_ratio_v=OscarConfig.get_v_clip_ratio(),
        )

    # -------------------------------------------------------------- #
    #  Flash attention helper                                          #
    # -------------------------------------------------------------- #

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q, k=k, v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        return flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.fa_version,
        )

    # -------------------------------------------------------------- #
    #  Forward: route to decode / prefill / mixed                      #
    # -------------------------------------------------------------- #

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "OscarMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)
        oscar_layer: Any = layer
        device = q.device
        self._ensure_on_device(oscar_layer, device)
        R = oscar_layer._oscar_R

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, R
            )
        elif num_decodes == 0:
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q, k, v, kv_cache, attn_metadata, R, layer=layer
            )
        else:
            attn_out = self._mixed_batch(
                q, key, value, N, kv_cache, attn_metadata, R, layer
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # -------------------------------------------------------------- #
    #  Decode attention                                                #
    # -------------------------------------------------------------- #

    @torch.compiler.disable
    def _decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: OscarMetadata,
        R: torch.Tensor,
    ) -> torch.Tensor:
        return oscar_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table[: query.shape[0]],
            seq_lens=attn_metadata.seq_lens[: query.shape[0]],
            R=R,
            scale=self.scale,
        )

    # -------------------------------------------------------------- #
    #  Prefill attention                                                #
    # -------------------------------------------------------------- #

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: OscarMetadata,
        R: torch.Tensor,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape
        Hk = key.shape[1]
        use_gqa = Hk < Hq

        # Fast path: flash_attn for first-chunk prefills
        if (
            _HAS_FLASH_ATTN
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
        ):
            return self._flash_attn_varlen(
                q=query, k=key, v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        # Per-request attention (continuation prefills)
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1
        output = torch.zeros(
            N, Hq, D, device=query.device, dtype=query.dtype
        )

        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(
                2, device=query.device, dtype=torch.int32
            )

        for i in range(num_reqs):
            q_start, q_end = qsl[i], qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]

            if q_len == seq_len:
                out = self._first_chunk_prefill(
                    q_seq, k_seq, v_seq, q_len, use_gqa
                )
            else:
                out = self._continuation_prefill(
                    layer, q_seq, k_seq, v_seq,
                    kv_cache, attn_metadata.block_table[i: i + 1],
                    seq_len - q_len, seq_len, R,
                )
            output[q_start:q_end] = out.to(query.dtype)

        return output

    def _first_chunk_prefill(
        self,
        q_seq: torch.Tensor,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        q_len: int,
        use_gqa: bool,
    ) -> torch.Tensor:
        if _HAS_FLASH_ATTN:
            self._cu_2[1:2] = q_len
            return self._flash_attn_varlen(
                q=q_seq, k=k_seq, v=v_seq,
                cu_seqlens_q=self._cu_2, cu_seqlens_k=self._cu_2,
                max_seqlen_q=q_len, max_seqlen_k=q_len,
            )
        q_t = q_seq.transpose(0, 1).contiguous()
        k_t = k_seq.transpose(0, 1).contiguous()
        v_t = v_seq.transpose(0, 1).contiguous()
        return F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            is_causal=True, scale=self.scale, enable_gqa=use_gqa,
        ).transpose(0, 1)

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,
        key_chunk: torch.Tensor,
        val_chunk: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int,
        seq_len: int,
        R: torch.Tensor,
    ) -> torch.Tensor:
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]

        # Build slot mapping for cached tokens
        seq_indices = torch.arange(cached_len, device=device)
        block_indices = seq_indices // block_size
        slot_offsets = seq_indices % block_size
        blocks = block_table[0, block_indices]
        slot_mapping = blocks * block_size + slot_offsets

        # Dequant + un-rotate cached KV
        k_cached, v_cached = dequant_and_unrotate(
            kv_cache, slot_mapping, R, D
        )

        # Concatenate cached + current chunk
        qdtype = query.dtype
        k_full = torch.empty(
            seq_len, Hk, D, dtype=qdtype, device=device
        )
        v_full = torch.empty(
            seq_len, Hk, D, dtype=qdtype, device=device
        )
        k_full[:cached_len] = k_cached.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached.to(qdtype)
        v_full[cached_len:] = val_chunk

        if _HAS_FLASH_ATTN:
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(
                    2, device=device, dtype=torch.int32
                )
                self._cu_2_k = torch.zeros(
                    2, device=device, dtype=torch.int32
                )
            self._cu_2_q[1:2] = q_len
            self._cu_2_k[1:2] = seq_len
            return self._flash_attn_varlen(
                q=query, k=k_full, v=v_full,
                cu_seqlens_q=self._cu_2_q,
                cu_seqlens_k=self._cu_2_k,
                max_seqlen_q=q_len, max_seqlen_k=seq_len,
            )

        # SDPA fallback
        q_t = query.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        q_pos = (
            torch.arange(q_len, device=device).unsqueeze(1) + cached_len
        )
        k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = k_pos <= q_pos
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=mask, scale=self.scale,
            enable_gqa=(Hk < Hq),
        )
        return out[0].transpose(0, 1)

    # -------------------------------------------------------------- #
    #  Mixed batch (decode + prefill)                                  #
    # -------------------------------------------------------------- #

    def _mixed_batch(
        self,
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        N: int,
        kv_cache: torch.Tensor,
        attn_metadata: OscarMetadata,
        R: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        device = q.device
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        attn_out = torch.empty(
            N, self.num_heads, self.head_size,
            device=device, dtype=q.dtype,
        )

        # Decode portion
        decode_meta = OscarMetadata(
            seq_lens=attn_metadata.seq_lens[:num_decodes],
            slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
            block_table=attn_metadata.block_table[:num_decodes],
            query_start_loc=(
                attn_metadata.query_start_loc[:num_decodes + 1]
            ),
            num_actual_tokens=num_decode_tokens,
            max_query_len=1,
            max_seq_len=attn_metadata.max_seq_len,
            is_prefill=False,
        )
        attn_out[:num_decode_tokens] = self._decode_attention(
            q[:num_decode_tokens], kv_cache, decode_meta, R
        )

        # Prefill portion
        prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
        if attn_metadata.seq_lens_cpu is not None:
            prefill_max_seq = int(
                attn_metadata.seq_lens_cpu[num_decodes:].max()
            )
        else:
            prefill_max_seq = attn_metadata.max_seq_len

        prefill_qsl = (
            attn_metadata.query_start_loc[num_decodes:]
            - num_decode_tokens
        )
        prefill_qsl_cpu = None
        if attn_metadata.query_start_loc_cpu is not None:
            prefill_qsl_cpu = (
                attn_metadata.query_start_loc_cpu[num_decodes:]
                - num_decode_tokens
            )

        prefill_meta = OscarMetadata(
            seq_lens=prefill_seq_lens,
            slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
            block_table=attn_metadata.block_table[num_decodes:],
            query_start_loc=prefill_qsl,
            num_actual_tokens=N - num_decode_tokens,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=prefill_max_seq,
            is_prefill=True,
            query_start_loc_cpu=prefill_qsl_cpu,
            seq_lens_cpu=(
                attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None
            ),
        )

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        attn_out[num_decode_tokens:] = self._prefill_attention(
            q[num_decode_tokens:],
            k[num_decode_tokens:],
            v[num_decode_tokens:],
            kv_cache, prefill_meta, R, layer=layer,
        )

        return attn_out
