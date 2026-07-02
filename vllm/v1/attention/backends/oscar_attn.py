# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR INT2 KV-cache attention backend for vLLM V1 (pure PyTorch).

Prefill uses flash-attn on uncompressed KV; decode dequantizes from the
compressed cache, computes attention in the rotated space, then un-rotates.

Cache layout per slot: [k_packed | k_scale_zero | pad | v_packed | v_scale_zero]
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

_HAS_FA = is_flash_attn_varlen_func_available()
if _HAS_FA:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

# ------------------------------------------------------------------ #
#  Rotation matrix (Hadamard / random orthogonal)                      #
# ------------------------------------------------------------------ #


@functools.cache
def _hadamard(d: int, dev: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1),
                        torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(dev))


def build_rotation_matrix(d: int, device: torch.device) -> torch.Tensor:
    if d > 0 and (d & (d - 1)) == 0:
        return _hadamard(d, str(device))
    gen = torch.Generator(device="cpu").manual_seed(42)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
    return Q.to(device)


# ------------------------------------------------------------------ #
#  INT2 pack / unpack / quantize / dequantize                          #
# ------------------------------------------------------------------ #


def _pack2(v: torch.Tensor) -> torch.Tensor:
    v = v.to(torch.uint8).view(*v.shape[:-1], v.shape[-1] // 4, 4)
    return v[..., 0] | (v[..., 1] << 2) | (v[..., 2] << 4) | (v[..., 3] << 6)


def _unpack2(p: torch.Tensor, D: int) -> torch.Tensor:
    p = p.to(torch.int32).unsqueeze(-1)
    return torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3,
                        (p >> 6) & 3], -1).view(*p.shape[:-2], D)


def _quant2(row: torch.Tensor, clip: float = 0.0):
    r = row.float()
    if clip > 0:
        idx = min(int(clip * r.shape[-1]), r.shape[-1] - 1)
        t = r.abs().sort(-1).values[..., idx:idx + 1]
        r = r.clamp(-t, t)
    rmin = r.amin(-1, keepdim=True)
    scale = (r.amax(-1, keepdim=True) - rmin).clamp(min=1e-8) / 3.0
    zero = -rmin / scale
    q = _pack2(((r - rmin) / scale + 0.5).clamp(0, 3))
    return q, scale.half().squeeze(-1), zero.half().squeeze(-1)


def _dequant2(packed, scale, zero, D):
    return (_unpack2(packed, D).float()
            - zero.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)


# ------------------------------------------------------------------ #
#  Cache layout helpers                                                #
# ------------------------------------------------------------------ #


def _layout(D: int):
    """Returns (k_start, ksz_off, v_start, vsz_off, slot_size)."""
    kq = D // 4
    ka = (kq + 4 + 15) // 16 * 16
    return 0, kq, ka, ka + kq, ka * 2


def _store_kv(k_rot, v_rot, cache, slots, clip_k, clip_v):
    Hk, D = k_rot.shape[1:]
    ks, ksz, vs, vsz, _ = _layout(D)
    flat = cache.view(-1, Hk, cache.shape[-1])
    valid = slots >= 0
    if not valid.any():
        return
    idx = valid.nonzero(as_tuple=True)[0]
    s, k, v = slots[idx], k_rot[idx], v_rot[idx]
    M = k.shape[0]

    kp, ksc, kz = _quant2(k, clip_k)
    vp, vsc, vz = _quant2(v, clip_v)

    flat[s, :, ks:ks + D // 4] = kp
    flat[s, :, ksz:ksz + 4] = torch.stack(
        [ksc, kz], -1).view(torch.uint8).view(M, Hk, 4)
    flat[s, :, vs:vs + D // 4] = vp
    flat[s, :, vsz:vsz + 4] = torch.stack(
        [vsc, vz], -1).view(torch.uint8).view(M, Hk, 4)


def _load_kv(cache, slots, D):
    Hk, seq = cache.shape[2], slots.shape[0]
    ks, ksz, vs, vsz, _ = _layout(D)
    flat = cache.view(-1, Hk, cache.shape[-1])
    s = slots.long()

    def _read(start, sz_off):
        p = flat[s, :, start:start + D // 4]
        sz = flat[s, :, sz_off:sz_off + 4].view(torch.float16).view(seq, Hk, 2)
        return _dequant2(p, sz[..., 0], sz[..., 1], D)

    return _read(ks, ksz), _read(vs, vsz)


# ------------------------------------------------------------------ #
#  Backend / Metadata / Builder                                        #
# ------------------------------------------------------------------ #


class OscarAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16]
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
    def get_impl_cls():
        return OscarAttentionImpl

    @staticmethod
    def get_builder_cls():
        return OscarMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                           head_size, cache_dtype_str="oscar_int2"):
        cfg = OscarConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, cfg.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, dt):
        return dt is not None and dt.startswith("oscar_")

    @classmethod
    def supports_head_size(cls, hd):
        return hd > 0 and hd % 4 == 0


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
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.NEVER)

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        m = self.build(0, common_attn_metadata)
        m.seq_lens.fill_(1)
        return m

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        nd, _, ndt, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold)
        return OscarMetadata(
            seq_lens=cam.seq_lens, slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len, max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1), num_decodes=nd,
            num_decode_tokens=ndt,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound)


# ------------------------------------------------------------------ #
#  Attention implementation                                            #
# ------------------------------------------------------------------ #


class OscarAttentionImpl(AttentionImpl["OscarMetadata"]):
    supports_quant_query_input: bool = False

    def __init__(self, num_heads, head_size, scale,
                 num_kv_heads=None, *args, **kwargs):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_group = num_heads // self.num_kv_heads
        self.fa_ver = get_flash_attn_version(head_size=head_size)

    def _ensure_R(self, layer: Any, device: torch.device):
        if not hasattr(layer, "_oscar_R"):
            layer._oscar_R = build_rotation_matrix(self.head_size, device)

    # -- KV cache update ------------------------------------------------- #

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        self._ensure_R(layer, key.device)
        R = layer._oscar_R.to(key.dtype).float()
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        _store_kv((k.float() @ R).to(k.dtype), (v.float() @ R).to(v.dtype),
                  kv_cache, slot_mapping,
                  OscarConfig.get_k_clip_ratio(),
                  OscarConfig.get_v_clip_ratio())

    # -- Forward --------------------------------------------------------- #

    def forward(self, layer, query, key, value, kv_cache,
                attn_metadata, output=None, **kwargs):
        N = attn_metadata.num_actual_tokens if attn_metadata else 0
        if output is None:
            output = torch.zeros(query.shape[0],
                                 self.num_heads * self.head_size,
                                 dtype=query.dtype, device=query.device)
        if N <= 0:
            return output.fill_(0)

        self._ensure_R(layer, query.device)
        q = query[:N].view(N, self.num_heads, self.head_size)

        if not attn_metadata.is_prefill:
            res = self._decode(q, kv_cache, attn_metadata, layer._oscar_R)
        elif attn_metadata.num_decodes == 0:
            res = self._prefill(
                q, key[:N].view(N, self.num_kv_heads, self.head_size),
                value[:N].view(N, self.num_kv_heads, self.head_size),
                attn_metadata)
        else:
            # Mixed decode+prefill: split and handle each part
            res = self._mixed(q, key, value, N, kv_cache, attn_metadata,
                              layer._oscar_R)

        if output.ndim == 3:
            output[:N] = res.to(output.dtype)
        else:
            output[:N] = res.reshape(N, -1).to(output.dtype)
        return output

    # -- Prefill (flash-attn on raw KV) ---------------------------------- #

    def _prefill(self, q, k, v, meta):
        if _HAS_FA and meta.max_query_len == meta.max_seq_len:
            kw = dict(q=q, k=k, v=v, cu_seqlens_q=meta.query_start_loc,
                      cu_seqlens_k=meta.query_start_loc,
                      max_seqlen_q=meta.max_query_len,
                      max_seqlen_k=meta.max_query_len,
                      softmax_scale=self.scale, causal=True)
            if self.fa_ver is not None:
                kw["fa_version"] = self.fa_ver
            return flash_attn_varlen_func(**kw)
        # Per-request SDPA fallback for continuation prefills
        return self._prefill_sdpa(q, k, v, meta)

    def _prefill_sdpa(self, q, k, v, meta):
        qsl = (meta.query_start_loc if meta.query_start_loc_cpu is None else meta.query_start_loc_cpu).tolist()
        out = torch.zeros_like(q)
        for i in range(len(qsl) - 1):
            s, e = qsl[i], qsl[i + 1]
            if e <= s:
                continue
            qt = q[s:e].transpose(0, 1).unsqueeze(0)
            kt = k[s:e].transpose(0, 1).unsqueeze(0)
            vt = v[s:e].transpose(0, 1).unsqueeze(0)
            out[s:e] = F.scaled_dot_product_attention(
                qt, kt, vt, is_causal=True, scale=self.scale,
                enable_gqa=(self.num_kv_heads < self.num_heads),
            )[0].transpose(0, 1)
        return out

    # -- Decode (dequant from cache) ------------------------------------- #

    @torch.compiler.disable
    def _decode(self, q, kv_cache, meta, R):
        B, D = q.shape[0], self.head_size
        bs = kv_cache.shape[1]
        q_rot = q.float() @ R.float()
        out = torch.empty_like(q)

        for b in range(B):
            slen = int(meta.seq_lens[b])
            if slen <= 0:
                out[b] = 0
                continue
            pos = torch.arange(slen, device=q.device)
            slots = meta.block_table[b, pos // bs].long() * bs + pos % bs
            k_rot, v_rot = _load_kv(kv_cache, slots, D)

            ke = k_rot.transpose(0, 1).repeat_interleave(self.kv_group, 0)
            ve = v_rot.transpose(0, 1).repeat_interleave(self.kv_group, 0)
            w = F.softmax(torch.bmm(
                q_rot[b].unsqueeze(1),
                ke.transpose(1, 2)).squeeze(1) * self.scale, -1)
            out_rot = torch.bmm(w.unsqueeze(1), ve).squeeze(1)
            out[b] = (out_rot @ R.T.float()).to(q.dtype)
        return out

    # -- Mixed batch ----------------------------------------------------- #

    def _mixed(self, q, key, value, N, kv_cache, meta, R):
        nd, ndt = meta.num_decodes, meta.num_decode_tokens
        out = torch.empty(N, self.num_heads, self.head_size,
                          device=q.device, dtype=q.dtype)
        # Decode portion
        dm = OscarMetadata(
            seq_lens=meta.seq_lens[:nd], slot_mapping=meta.slot_mapping[:ndt],
            block_table=meta.block_table[:nd],
            query_start_loc=meta.query_start_loc[:nd + 1],
            num_actual_tokens=ndt, max_query_len=1,
            max_seq_len=meta.max_seq_len, is_prefill=False)
        out[:ndt] = self._decode(q[:ndt], kv_cache, dm, R)
        # Prefill portion
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        pqsl = meta.query_start_loc[nd:] - ndt
        pm = OscarMetadata(
            seq_lens=meta.seq_lens[nd:],
            slot_mapping=meta.slot_mapping[ndt:N],
            block_table=meta.block_table[nd:],
            query_start_loc=pqsl,
            num_actual_tokens=N - ndt,
            max_query_len=meta.max_query_len,
            max_seq_len=meta.max_seq_len, is_prefill=True,
            query_start_loc_cpu=(meta.query_start_loc_cpu[nd:] - ndt
                                 if meta.query_start_loc_cpu is not None
                                 else None),
            seq_lens_cpu=(meta.seq_lens_cpu[nd:]
                          if meta.seq_lens_cpu is not None else None))
        out[ndt:] = self._prefill(q[ndt:], k[ndt:], v[ndt:], pm)
        return out
