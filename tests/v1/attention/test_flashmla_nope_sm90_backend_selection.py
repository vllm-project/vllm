# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the FlashMLA sparse backend NoPE-512 (GLM5Next) gates.

GLM5Next is rope-free MLA: qk_nope_head_dim=256, qk_rope_head_dim=0,
v_head_dim=256, kv_lora_rank=512 -> head_size 512. These tests pin the
selection contract only (no GPU required):

* ``get_supported_head_sizes`` advertises 512 so the backend is a candidate;
* ``supports_combination`` admits exactly two shapes for 512 -- a quantized
  DS-MLA cache format (zero-padded 576/656B envelope) and a rope-free bf16
  cache on SM90 -- and rejects everything else;
* adding 512 must NOT reorder the SM100 priority list for bf16, which is why
  the bf16 arm is restricted to SM90.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    QUANTIZED_DS_MLA_CACHE_FORMATS,
    FlashMLASparseBackend,
)

SM90 = DeviceCapability(major=9, minor=0)
SM100 = DeviceCapability(major=10, minor=0)

BF16_CACHE_DTYPES = (None, "auto", "bfloat16", "float16")
QUANTIZED_CACHE_DTYPES = ("fp8_ds_mla", "nvfp4_ds_mla")


def _supports_combination(head_size, kv_cache_dtype, capability):
    return FlashMLASparseBackend.supports_combination(
        head_size,
        torch.bfloat16,
        kv_cache_dtype,
        64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=capability,
    )


@pytest.fixture
def rope_free_model(monkeypatch):
    """Patch get_current_vllm_config so the model looks like GLM5Next NoPE."""
    import vllm.config as cfg

    monkeypatch.setattr(
        cfg,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(qk_rope_head_dim=0)
            )
        ),
    )


@pytest.fixture
def rope_carrying_model(monkeypatch):
    """Patch get_current_vllm_config with a rope-carrying (576-style) model."""
    import vllm.config as cfg

    monkeypatch.setattr(
        cfg,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(qk_rope_head_dim=64)
            )
        ),
    )


def test_supported_head_sizes_include_512():
    assert FlashMLASparseBackend.get_supported_head_sizes() == [576, 512]


def test_quantized_ds_mla_formats_are_the_envelope_set():
    assert frozenset({"fp8_ds_mla", "nvfp4_ds_mla"}) == QUANTIZED_DS_MLA_CACHE_FORMATS


@pytest.mark.parametrize("kv_cache_dtype", QUANTIZED_CACHE_DTYPES)
def test_nope_512_quantized_rejected_here(kv_cache_dtype):
    # Quantized DS-MLA NoPE-512 is served by the zero-padded 576/656B
    # envelope, wired up in a separate change; this backend must reject it.
    reason = _supports_combination(512, kv_cache_dtype, SM90)
    assert reason is not None


@pytest.mark.parametrize("kv_cache_dtype", BF16_CACHE_DTYPES)
def test_nope_512_bf16_sm90_rope_free_accepted(kv_cache_dtype, rope_free_model):
    assert _supports_combination(512, kv_cache_dtype, SM90) is None


@pytest.mark.parametrize("kv_cache_dtype", BF16_CACHE_DTYPES)
def test_nope_512_bf16_sm90_rope_carrying_rejected(kv_cache_dtype, rope_carrying_model):
    reason = _supports_combination(512, kv_cache_dtype, SM90)
    assert reason is not None and "rope-free" in reason


@pytest.mark.parametrize("kv_cache_dtype", BF16_CACHE_DTYPES)
def test_nope_512_bf16_sm100_rejected(kv_cache_dtype, rope_free_model):
    # The SM90 restriction is what keeps the SM100 priority order unchanged.
    reason = _supports_combination(512, kv_cache_dtype, SM100)
    assert reason is not None


@pytest.mark.parametrize("kv_cache_dtype", ("fp8", "fp8_e4m3"))
def test_nope_512_plain_fp8_rejected(kv_cache_dtype, rope_free_model):
    reason = _supports_combination(512, kv_cache_dtype, SM90)
    assert reason is not None


@pytest.mark.parametrize("capability", [SM90, SM100], ids=["sm90", "sm100"])
@pytest.mark.parametrize(
    "kv_cache_dtype",
    ("auto", "bfloat16", "fp8", "fp8_ds_mla", "nvfp4_ds_mla"),
)
def test_576_unchanged_for_all_dtypes(kv_cache_dtype, capability):
    # DeepSeek 576 must be untouched by the NoPE-512 gates, except for the
    # pre-existing SM100-only nvfp4 rule.
    reason = _supports_combination(576, kv_cache_dtype, capability)
    if kv_cache_dtype == "nvfp4_ds_mla" and capability.major != 10:
        assert reason is not None and "SM100" in reason
    else:
        assert reason is None, reason


def _sparse_order(kv_cache_dtype, head_size, num_heads=32, capability=SM100):
    try:
        from vllm.platforms.cuda import _get_backend_priorities
    except ImportError as e:  # pragma: no cover - depends on host
        pytest.skip(f"CUDA platform unavailable on this host: {e}")

    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=capability,
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
        head_size=head_size,
    )
    sparse = {
        "FLASH_ATTN_MLA_SPARSE",
        "FLASHMLA_SPARSE",
        "FLASHINFER_MLA_SPARSE",
        "FLASHINFER_MLA_SPARSE_SM90",
    }
    return [b.name for b in priorities if b.name in sparse]


@pytest.mark.parametrize("num_heads", [32, 64], ids=["tp2", "tp1"])
def test_sm100_bf16_512_priority_unchanged(num_heads):
    """Adding head_size 512 must not reorder the SM100 bf16 sparse list.

    On SM100 with >16 heads the hardcoded bf16 order is
    [FLASHMLA_SPARSE, FLASHINFER_MLA_SPARSE] (FlashMLA first). head_size 512
    must leave this untouched (identical to the 576 order); the guard that
    keeps bf16-512 from *serving* through FlashMLA on SM100 lives in
    supports_combination (gated to SM90), tested separately.
    """
    order512 = _sparse_order("auto", 512, num_heads=num_heads)
    order576 = _sparse_order("auto", 576, num_heads=num_heads)
    assert order512 == order576, order512
    assert order512[0] == "FLASHMLA_SPARSE", order512


def test_sm100_576_priority_unchanged():
    order = _sparse_order("auto", 576, num_heads=32)
    assert order[0] == "FLASHMLA_SPARSE", order
