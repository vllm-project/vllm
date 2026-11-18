# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV cache formats the QKV front writes and attn2 reads, head_dim 64 and
P = 16 or 32 slots a page (the hybrid KV cache manager gives an FP8 group
32-slot pages when other layers keep BF16, e.g. with
--kv-cache-dtype-skip-layers):

- bf16: [pages, H, P, 128] BF16, K | V per slot.
- fp8: the same in E4M3 bytes (uint8 or float8_e4m3fn storage), entries
  stored divided by the per-tensor k_scale / v_scale.

The kernels work in 16-slot sub-pages: sub-page i of a request is sub-page
i % (P / 16) of its block-table entry i // (P / 16).
"""

from typing import NamedTuple

import torch

HEAD_DIM = 64
PAGE = 16  # slots of a sub-page
PAGE_SLOTS = (16, 32)
KINDS = ("bf16", "fp8")
FP8_STORAGE = (torch.uint8, torch.float8_e4m3fn)


class KvQuant(NamedTuple):
    """A quantized cache: `kind` "fp8" and the per-tensor scales."""

    kind: str
    k_scale: float
    v_scale: float


class KvLayout(NamedTuple):
    """The compile-time cache layout: format and slots a page."""

    kind: str
    slots: int = PAGE

    @property
    def shift(self) -> int:
        """log2 of the sub-pages a page."""
        return (self.slots // PAGE).bit_length() - 1


BF16 = KvLayout("bf16")


def elem_bytes(kind: str) -> int:
    """Bytes per cache element of a BF16 or FP8 cache."""
    return 1 if kind == "fp8" else 2


def kind_of(quant: KvQuant | None) -> str:
    return "bf16" if quant is None else quant.kind


def layout_of(kv_cache: torch.Tensor, kind: str) -> KvLayout:
    if kv_cache.dim() != 4:
        return KvLayout(kind, 0)
    return KvLayout(kind, kv_cache.shape[2])


def layout_ok(kv_cache: torch.Tensor, num_kv_heads: int, kind: str) -> bool:
    """Whether a cache has the layout of `kind` for these KV heads: pages of
    16 or 32 slots of dense heads (HND), 16-byte aligned. Pages may be
    padded (the hybrid KV cache manager pads a group's pages to the largest
    group's bytes)."""
    if kind == "bf16":
        dtype_ok = kv_cache.dtype == torch.bfloat16
    elif kind == "fp8":
        dtype_ok = kv_cache.dtype in FP8_STORAGE
    else:
        return False
    heads, row = num_kv_heads, 2 * HEAD_DIM
    if not dtype_ok or kv_cache.dim() != 4:
        return False
    page = kv_cache.shape[2]
    if page not in PAGE_SLOTS or tuple(kv_cache.shape[1:]) != (heads, page, row):
        return False
    stride = kv_cache.stride()
    return (
        stride[0] >= heads * page * row
        and stride[0] * kv_cache.element_size() % 16 == 0
        and (heads == 1 or stride[1] == page * row)
        and stride[2] == row
        and stride[3] == 1
        and kv_cache.data_ptr() % 16 == 0
    )
