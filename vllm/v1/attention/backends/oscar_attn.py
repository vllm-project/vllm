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
    # True when any request in the batch attends context beyond its own new
    # tokens (prefix-cache hit or chunked-prefill continuation). Guards the
    # flash-attn first-chunk fast path.
    has_context: bool = False
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None


class OscarMetadataBuilder(AttentionMetadataBuilder[OscarMetadata]):
    # The BF16 sink/recent staging path uses data-dependent block allocation
    # and has only been validated in eager mode.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

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
        has_context = False
        if cam.query_start_loc_cpu is not None and cam.seq_lens_cpu is not None:
            q_lens_cpu = cam.query_start_loc_cpu[1:] - cam.query_start_loc_cpu[:-1]
            has_context = bool((cam.seq_lens_cpu > q_lens_cpu).any())
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
            has_context=has_context,
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
        if sliding_window is not None:
            raise NotImplementedError(
                "OSCAR INT2 KV cache does not support sliding-window attention "
                "layers; add 'sliding_window' to kv_cache_dtype_skip_layers to "
                "keep them in the native dtype."
            )
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
        # BF16 sink/recent windows (see OscarConfig). Resolved per-layer at
        # first forward when the cache block size is known.
        self.window_enabled = (
            self.cfg.sink_tokens > 0 or self.cfg.recent_tokens > 0
        ) and self.cfg.staging_tokens > 0

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

    # ---- BF16 sink/recent staging ------------------------------------------
    # The reference OSCAR serving stack never attends the newest (and first,
    # "sink") tokens in INT2: they stay BF16 and are only demoted once they
    # age out of the recent window. Mirroring that here: sink/recent tokens
    # are dual-written into a per-layer BF16 staging arena keyed by cache
    # block id (owner-tagged per slot, so evictions/collisions degrade
    # gracefully to the INT2 copy), and attention reads them in BF16.

    def _ensure_staging(self, layer: Any, kv_cache: torch.Tensor):
        if getattr(layer, "_oscar_stage_ready", False):
            return
        bs = kv_cache.shape[1]
        cfg = self.cfg
        self.stage_block = bs
        self.sink_eff = (cfg.sink_tokens // bs) * bs
        self.sink_pages = self.sink_eff // bs
        # recent window can straddle one extra page
        self.tail_pages = (cfg.recent_tokens + bs - 1) // bs + 1
        rows = max(
            (cfg.staging_tokens + bs - 1) // bs,
            self.sink_pages + self.tail_pages + 2,
        )
        layer._oscar_stage_k = torch.zeros(
            rows,
            bs,
            self.num_kv_heads,
            self.head_size,
            dtype=torch.bfloat16,
            device=kv_cache.device,
        )
        layer._oscar_stage_v = torch.zeros_like(layer._oscar_stage_k)
        layer._oscar_slot_owner = torch.full(
            (rows, bs), -1, dtype=torch.int64, device=kv_cache.device
        )
        layer._oscar_stage_rows = rows
        layer._oscar_stage_ready = True

    def _staging_write(self, layer: Any, key, value, attn_metadata):
        """Write this step's sink/recent tokens (raw BF16 K/V) into staging.

        The INT2 copy is still written by ``do_kv_cache_update``, so a token
        evicted from staging (row reuse or hash collision) simply falls back
        to its quantized copy.
        """
        N = attn_metadata.num_actual_tokens
        slot = attn_metadata.slot_mapping[:N].to(torch.int64)
        seq = attn_metadata.seq_lens.to(torch.int64)
        qsl = attn_metadata.query_start_loc.to(torch.int64)
        q_lens = qsl[1:] - qsl[:-1]
        req = torch.repeat_interleave(
            torch.arange(seq.shape[0], device=slot.device), q_lens
        )
        # position of token j (global index) in its sequence
        pos = seq[req] - qsl[req + 1] + torch.arange(N, device=slot.device)
        keep = (slot >= 0) & (
            (pos >= seq[req] - self.cfg.recent_tokens) | (pos < self.sink_eff)
        )
        bs = self.stage_block
        rows = (slot // bs) % layer._oscar_stage_rows
        kb, kr, ko = (slot // bs)[keep], rows[keep], (slot % bs)[keep]
        if kb.numel() == 0:
            return
        # Owner tag first; only the winning writer of each (row, off) slot
        # stores its data, so a stale tag can never point at mixed contents.
        layer._oscar_slot_owner.index_put_((kr, ko), kb)
        win = layer._oscar_slot_owner[kr, ko] == kb
        sel = keep.nonzero(as_tuple=True)[0][win]
        layer._oscar_stage_k.index_put_((kr[win], ko[win]), key[sel].to(torch.bfloat16))
        layer._oscar_stage_v.index_put_(
            (kr[win], ko[win]), value[sel].to(torch.bfloat16)
        )

    def _stage_splice(self, layer: Any, bt_row, cached_len: int, k_cached, v_cached):
        """Replace INT2-reconstructed rows of a cached prefix with their exact
        BF16 staged values wherever the staging tag still matches."""
        bs = self.stage_block
        rows_total = layer._oscar_stage_rows
        dev = k_cached.device
        npg = (cached_len + bs - 1) // bs
        blk = bt_row[:npg].to(torch.int64)
        rows = blk % rows_total
        staged = layer._oscar_slot_owner[rows] == blk.unsqueeze(-1)  # [npg, bs]
        pos = (torch.arange(npg, device=dev) * bs).unsqueeze(-1) + torch.arange(
            bs, device=dev
        )
        staged = (staged & (pos < cached_len)).reshape(-1)[:cached_len]
        ks = layer._oscar_stage_k[rows].reshape(npg * bs, self.num_kv_heads, -1)
        vs = layer._oscar_stage_v[rows].reshape(npg * bs, self.num_kv_heads, -1)
        m = staged.view(-1, 1, 1)
        k_out = torch.where(m, ks[:cached_len].to(k_cached.dtype), k_cached)
        v_out = torch.where(m, vs[:cached_len].to(v_cached.dtype), v_cached)
        return k_out, v_out

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

        if self.window_enabled and kv_cache.numel() > 0:
            self._ensure_staging(layer, kv_cache)
            self._staging_write(
                layer,
                key[:N].view(N, self.num_kv_heads, self.head_size),
                value[:N].view(N, self.num_kv_heads, self.head_size),
                attn_metadata,
            )

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
        # raw (uncompressed) K/V exactly like the standard backend. Note that
        # max_query_len == max_seq_len alone does NOT imply this: a fresh
        # request can share the batch with a shorter continuation (prefix-cache
        # hit / chunked prefill), so also require that no request has prior
        # context.
        if (
            _HAS_FLASH_ATTN
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and not attn_metadata.has_context
        ):
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
                if self.window_enabled and getattr(layer, "_oscar_stage_ready", False):
                    # exact BF16 values for any still-staged prefix positions
                    k_cached, v_cached = self._stage_splice(
                        layer,
                        attn_metadata.block_table[i],
                        cached_len,
                        k_cached,
                        v_cached,
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
        if self.window_enabled and getattr(layer, "_oscar_stage_ready", False):
            return self._decode_attention_windowed(
                query, kv_cache, attn_metadata, layer
            )
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

    def _decode_attention_windowed(self, query, kv_cache, attn_metadata, layer):
        """Decode attention with BF16 sink/recent windows.

        Splits every sequence into three ranges: ``[0, sink)`` and
        ``[cut, seq)`` are attended in BF16 from the staging arena, the
        middle ``[sink, cut)`` runs through the fused INT2 kernel (with the
        block table shifted past the sink pages so the kernel needs no
        changes), and the two partial results are LSE-merged. ``cut`` is the
        earliest position from which staging coverage is contiguous to the
        end of the sequence, so evicted staging entries transparently fall
        back to INT2.
        """
        B = query.shape[0]
        Hq, Hk, D = self.num_heads, self.num_kv_heads, self.head_size
        g = Hq // Hk
        bs = self.stage_block
        R = layer._oscar_stage_rows
        dev = query.device
        owner = layer._oscar_slot_owner
        bt = attn_metadata.block_table
        seq = attn_metadata.seq_lens.to(torch.int64)
        maxpg = bt.shape[1]
        S_nb, TP, S_eff = self.sink_pages, self.tail_pages, self.sink_eff
        W = self.cfg.recent_tokens
        offs = torch.arange(bs, device=dev)

        # ---- sink coverage: BF16 only when every sink position is staged ----
        if S_nb > 0:
            sblk = bt[:, :S_nb].to(torch.int64)  # [B, S_nb]
            sown = owner[(sblk % R).unsqueeze(-1), offs.view(1, 1, bs)]
            s_staged = sown == sblk.unsqueeze(-1)  # [B, S_nb, bs]
            sink_active = (seq > S_eff) & s_staged.reshape(B, -1).all(dim=1)
            spos = (torch.arange(S_nb, device=dev) * bs).view(1, S_nb, 1) + offs.view(
                1, 1, bs
            )
            s_valid = sink_active.view(B, 1, 1) & (spos < S_eff)
            s_valid = s_valid.expand(B, S_nb, bs)
        else:
            sblk = torch.zeros(B, 0, dtype=torch.int64, device=dev)
            s_valid = torch.zeros(B, 0, bs, dtype=torch.bool, device=dev)
            sink_active = torch.zeros(B, dtype=torch.bool, device=dev)
        si = torch.where(sink_active, torch.full_like(seq, S_nb), torch.zeros_like(seq))

        # ---- recent-tail coverage: find the contiguous staged suffix ----
        last_page = (seq - 1) // bs
        pg = (last_page - (TP - 1)).unsqueeze(1) + torch.arange(
            TP, device=dev
        ).unsqueeze(0)  # [B, TP], ascending, may be < 0
        pg_ok = pg >= 0
        tblk = torch.gather(bt.to(torch.int64), 1, pg.clamp(0, maxpg - 1))
        town = owner[(tblk % R).unsqueeze(-1), offs.view(1, 1, bs)]
        t_staged = (town == tblk.unsqueeze(-1)) & pg_ok.unsqueeze(-1)
        pos = (pg * bs).unsqueeze(-1) + offs.view(1, 1, bs)  # [B, TP, bs]
        t0 = torch.maximum(seq - W, si * bs)
        inrange = (pos >= t0.view(B, 1, 1)) & (pos < seq.view(B, 1, 1))
        ok = torch.where(inrange, t_staged, torch.ones_like(t_staged))
        sv = (
            torch.flip(torch.cumprod(torch.flip(ok.reshape(B, -1).long(), [1]), 1), [1])
            > 0
        )
        posf = pos.reshape(B, -1)
        cand = sv & inrange.reshape(B, -1)
        big = torch.iinfo(torch.int64).max
        cut = torch.where(cand, posf, torch.full_like(posf, big)).amin(1)
        cut = torch.minimum(cut, seq)  # no coverage -> empty BF16 tail
        t_valid = t_staged & inrange & (pos >= cut.view(B, 1, 1))

        # ---- INT2 kernel over [sink, cut): shift the block table by the
        # sink pages so positions stay page-aligned and the kernel is unchanged
        seq_eff = (cut - si * bs).to(torch.int32)
        gidx = (torch.arange(maxpg, device=dev).unsqueeze(0) + si.unsqueeze(1)).clamp(
            max=maxpg - 1
        )
        bt_eff = torch.gather(bt, 1, gidx)
        q_rot = torch.matmul(query.float(), layer._oscar_Rk)
        lse1 = torch.full((B, Hq), float("-inf"), dtype=torch.float32, device=dev)
        out1 = oscar_decode_attention(
            q_rot,
            kv_cache,
            bt_eff,
            seq_eff,
            self.scale,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            max_num_kv_splits=self.max_num_kv_splits,
            lse_buf=lse1,
        )
        o1 = torch.matmul(out1, layer._oscar_RvT)  # true space, normalized
        empty1 = (seq_eff <= 0).view(B, 1)
        lse1 = torch.where(
            empty1 | ~torch.isfinite(lse1),
            torch.full_like(lse1, float("-inf")),
            lse1,
        )
        o1 = torch.nan_to_num(o1)

        # ---- BF16 attention over the staged sink + tail segments ----
        all_blk = torch.cat([sblk, tblk], dim=1)  # [B, P]
        valid = torch.cat([s_valid, t_valid], dim=1)  # [B, P, bs]
        P = all_blk.shape[1]
        L = P * bs
        rowsP = all_blk % R
        kseg = layer._oscar_stage_k[rowsP].reshape(B, L, Hk, D).float()
        vseg = layer._oscar_stage_v[rowsP].reshape(B, L, Hk, D).float()
        vmask = valid.reshape(B, L)
        qh = query.float().view(B, Hk, g, D)
        sc = torch.einsum("bkgd,blkd->bkgl", qh, kseg) * self.scale
        sc = sc.masked_fill(~vmask.view(B, 1, 1, L), float("-inf"))
        m2 = sc.amax(dim=-1)
        m2s = torch.where(torch.isfinite(m2), m2, torch.zeros_like(m2))
        p2 = torch.exp(sc - m2s.unsqueeze(-1))
        p2 = torch.where(vmask.view(B, 1, 1, L), p2, torch.zeros_like(p2))
        s2 = p2.sum(-1)
        o2 = torch.einsum("bkgl,blkd->bkgd", p2, vseg) / s2.clamp_min(1e-38).unsqueeze(
            -1
        )
        lse2 = torch.where(
            s2 > 0,
            m2s + torch.log(s2.clamp_min(1e-38)),
            torch.full_like(s2, float("-inf")),
        )
        o2 = o2.reshape(B, Hq, D)
        lse2 = lse2.reshape(B, Hq)

        # ---- merge the two normalized partial attentions via LSE ----
        new_lse = torch.logaddexp(lse1, lse2)
        w1 = torch.exp(lse1 - new_lse).unsqueeze(-1)
        w2 = torch.exp(lse2 - new_lse).unsqueeze(-1)
        out = o1 * w1 + torch.nan_to_num(o2) * w2
        return out.to(query.dtype)
