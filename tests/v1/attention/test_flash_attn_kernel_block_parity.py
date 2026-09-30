# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for FlashAttentionKernelBlock parity between the zero-arg and
spec-fed forms of ``FlashAttentionBackend.get_supported_kernel_block_sizes``.

(issue #58858): ``select_common_block_size`` now forwards the cache
group's ``kv_cache_spec`` to ``get_supported_kernel_block_sizes`` the same way
``vllm/model_executor/layers/attention/attention.py`` already does at layer
construction. The two gated branches (FA4-SM90-FP8-KV, FA4-hd256) read their
inputs from either the spec or ambient config, so ``FlashAttentionBackend``
can report different values for the two forms.

Both helpers read the same facts from two sources (spec.head_size vs
``model_config.get_head_size()``, spec.kv_quant_mode vs ``cache_config.cache_dtype``),
so both forms must agree for a homogeneous cache group. These tests force
each gated branch and assert that agreement.
"""

from types import SimpleNamespace

import torch

import vllm.v1.attention.backends.flash_attn as fa_mod
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.fa_utils import FA4_HD256_PAGE_SIZE
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import KVQuantMode, MLAAttentionSpec


def _bases(sizes: list) -> list:
    return [s.base if isinstance(s, MultipleOf) else s for s in sizes]


def _ambient(monkeypatch, head_size: int, cache_dtype: str) -> None:
    monkeypatch.setattr(
        fa_mod,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(get_head_size=lambda: head_size),
            cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        ),
    )


def test_spec_matches_ambient_default_branch(monkeypatch):
    """Without any gate applied, both input forms return the default list."""
    _ambient(monkeypatch, head_size=128, cache_dtype="auto")
    spec = MLAAttentionSpec(
        block_size=640, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    off = FlashAttentionBackend.get_supported_kernel_block_sizes()
    on = FlashAttentionBackend.get_supported_kernel_block_sizes(spec)
    assert _bases(off) == _bases(on) == [16]


def test_spec_matches_ambient_sm90_fp8_kv(monkeypatch):
    """FA4-SM90 FP8-KV: mirror inputs (spec.kv_quant_mode vs cache_dtype,
    spec.head_size vs model head) must resolve to the same 64-token tile."""
    monkeypatch.setattr(
        fa_mod.current_platform, "is_device_capability_family", lambda fam: True
    )
    monkeypatch.setattr(fa_mod, "get_flash_attn_version", lambda **kwargs: 4)
    _ambient(monkeypatch, head_size=512, cache_dtype="fp8")
    spec = MLAAttentionSpec(
        block_size=640,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    off = FlashAttentionBackend.get_supported_kernel_block_sizes()
    on = FlashAttentionBackend.get_supported_kernel_block_sizes(spec)
    assert _bases(off) == _bases(on) == [64]


def test_spec_matches_ambient_fa4_hd256(monkeypatch):
    """FA4-hd256: forced gate, both input forms land on the same page size."""
    monkeypatch.setattr(fa_mod, "uses_fa4_hd256_kernel", lambda *args, **kwargs: True)
    monkeypatch.setattr(fa_mod, "get_flash_attn_version", lambda **kwargs: 4)
    _ambient(monkeypatch, head_size=256, cache_dtype="auto")
    spec = MLAAttentionSpec(
        block_size=640, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    off = FlashAttentionBackend.get_supported_kernel_block_sizes()
    on = FlashAttentionBackend.get_supported_kernel_block_sizes(spec)
    assert _bases(off) == _bases(on) == [FA4_HD256_PAGE_SIZE]
