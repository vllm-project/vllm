# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR INT2 KV-cache attention backend for vLLM.

Prefill: standard attention on uncompressed (BF16) K/V, then rotate, clip
         and store K/V as INT2 into a combined cache slot.
Decode:  rotate the query by ``R_k``, score against the INT2 keys, dequant
         the INT2 values, weighted-sum, then map the output back with
         ``R_v^T``.

Cache layout (no leading 2 dimension), mirroring TurboQuant:
    (num_blocks, block_size, num_kv_heads, slot_size_aligned)
with each slot ``[key_packed | value_packed]`` and each region
``[int2_data | fp16 scale | fp16 zero]``.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.model_executor.layers.quantization.oscar.rotation import get_layer_rotation
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
from vllm.v1.attention.ops.triton_oscar_decode import (
    oscar_decode_attention,
    oscar_full_dequant_kv,
)
from vllm.v1.attention.ops.triton_oscar_store import oscar_store

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


class OscarAttentionBackend(AttentionBackend):
    """Attention backend using OSCAR INT2 KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["oscar_int2"]

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
        cfg = OscarConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, cfg.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is not None and kv_cache_dtype.startswith("oscar_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0


@dataclass
class OscarMetadata(AttentionMetadata):
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
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> OscarMetadata:
        m = self.build(0, common_attn_metadata)
        m.seq_lens.fill_(1)
        return m

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
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
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.cfg = OscarConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self.fa_version = get_flash_attn_version(head_size=head_size)
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

    # ---- rotation setup (one-time per layer) ------------------------------
    def _ensure_rotations(self, layer: Any, device: torch.device):
        if getattr(layer, "_oscar_ready", False):
            return
        D = self.head_size
        rk = get_layer_rotation(
            self.cfg.k_rotation_path, layer.layer_name, D, device, torch.float32
        )
        rv = get_layer_rotation(
            self.cfg.v_rotation_path, layer.layer_name, D, device, torch.float32
        )
        layer._oscar_Rk = rk
        layer._oscar_RkT = rk.t().contiguous()
        layer._oscar_Rv = rv
        layer._oscar_RvT = rv.t().contiguous()
        layer._oscar_ready = True

    def _rotate_clip(
        self, x: torch.Tensor, R: torch.Tensor, clip_ratio: float
    ) -> torch.Tensor:
        """Rotate ``x`` (``[N, H, D]``) by ``R`` and optionally percentile-clip."""
        x_rot = torch.matmul(x.float(), R)
        if clip_ratio > 0.0:
            thr = torch.quantile(x_rot.abs(), clip_ratio, dim=-1, keepdim=True)
            x_rot = torch.clamp(x_rot, -thr, thr)
        return x_rot

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
        self._ensure_rotations(layer, key.device)
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        k_rot = self._rotate_clip(k, layer._oscar_Rk, self.cfg.k_clip_ratio)
        v_rot = self._rotate_clip(v, layer._oscar_Rv, self.cfg.v_clip_ratio)
        oscar_store(
            k_rot,
            v_rot,
            kv_cache,
            slot_mapping,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_packed_size=self.cfg.key_packed_size,
            data_bytes=self.cfg.key_data_bytes,
        )

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

        self._ensure_rotations(layer, query.device)
        q = query[:N].view(N, self.num_heads, self.head_size)

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            attn_out = self._decode_attention(q, kv_cache, attn_metadata, layer)
        elif num_decodes == 0:
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(q, k, v, kv_cache, attn_metadata, layer)
        else:
            attn_out = torch.empty(
                N, self.num_heads, self.head_size, device=q.device, dtype=q.dtype
            )
            decode_meta = OscarMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, layer
            )
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
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
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                layer,
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _flash_attn_varlen(self, q, k, v, cu_q, cu_k, max_q, max_k):
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.scale,
            causal=True,
        )
        if self.fa_version is not None:
            kwargs["fa_version"] = self.fa_version
        return flash_attn_varlen_func(**kwargs)

    def _prefill_attention(self, query, key, value, kv_cache, attn_metadata, layer):
        N, Hq, D = query.shape
        Hk = key.shape[1]
        use_gqa = Hk < Hq

        # First-chunk prefill: all K/V are in the current batch — attend on the
        # raw (uncompressed) K/V exactly like the standard backend.
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                query,
                key,
                value,
                attn_metadata.query_start_loc,
                attn_metadata.query_start_loc,
                attn_metadata.max_query_len,
                attn_metadata.max_query_len,
            )

        # Continuation: dequant cached (rotated) INT2 K/V, inverse-rotate back to
        # the original space, concat with the current chunk, run flash attention.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = attn_metadata.query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        num_reqs = len(qsl) - 1
        for i in range(num_reqs):
            q_start, q_end = qsl[i], qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]
            cached_len = seq_len - q_len
            if cached_len <= 0:
                out = self._sdpa(q_seq, k_seq, v_seq, cached_len, seq_len, use_gqa)
            else:
                k_cached, v_cached = oscar_full_dequant_kv(
                    kv_cache,
                    attn_metadata.block_table[i : i + 1],
                    cached_len,
                    Hk,
                    D,
                    self.cfg.key_levels,
                    self.cfg.value_levels,
                    self.cfg.key_data_bytes,
                    self.cfg.key_packed_size,
                    self.cfg.value_data_bytes,
                )
                # inverse rotation back to original space
                k_cached = torch.matmul(k_cached.float(), layer._oscar_RkT).to(
                    query.dtype
                )
                v_cached = torch.matmul(v_cached.float(), layer._oscar_RvT).to(
                    query.dtype
                )
                k_full = torch.cat([k_cached, k_seq], dim=0)
                v_full = torch.cat([v_cached, v_seq], dim=0)
                out = self._sdpa(q_seq, k_full, v_full, cached_len, seq_len, use_gqa)
            output[q_start:q_end] = out.to(query.dtype)
        return output

    def _sdpa(self, q_seq, k_full, v_full, cached_len, seq_len, use_gqa):
        q_len = q_seq.shape[0]
        q_t = q_seq.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        device = q_seq.device
        if cached_len <= 0:
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scale, enable_gqa=use_gqa
            )
        else:
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=mask, scale=self.scale, enable_gqa=use_gqa
            )
        return out[0].transpose(0, 1)

    def _decode_attention(self, query, kv_cache, attn_metadata, layer):
        # Rotate the query into the same space as the rotated stored keys.
        q_rot = torch.matmul(query.float(), layer._oscar_Rk)
        out_rot = oscar_decode_attention(
            q_rot,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self.scale,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            max_num_kv_splits=self.max_num_kv_splits,
        )
        # Map attention output (rotated-V space) back: out_true = out_rot @ R_v^T.
        out = torch.matmul(out_rot, layer._oscar_RvT)
        return out.to(query.dtype)
