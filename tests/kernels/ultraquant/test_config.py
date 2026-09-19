# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only integration tests for the UltraQuant cache dtype."""

import torch

from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionBackend,
)
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    ultraquant_flydsl_decode_eligible,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
)


def test_ultraquant_quant_mode():
    mode = get_kv_quant_mode("ultraquant_4bit")
    assert mode == KVQuantMode.ULTRAQUANT_4BIT
    assert mode.is_ultraquant
    assert not mode.is_turboquant


def test_ultraquant_backend_customizes_d256_slot():
    spec = FullAttentionSpec(
        block_size=32,
        num_kv_heads=4,
        head_size=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.ULTRAQUANT_4BIT,
    )

    packed_spec = TurboQuantAttentionBackend.customize_spec(spec)

    assert TurboQuantAttentionBackend.supports_kv_cache_dtype("ultraquant_4bit")
    assert packed_spec.state_content_bytes == 272
    assert packed_spec.page_size_bytes == 32 * 4 * 272


def test_ultraquant_prefers_block_size_64():
    from unittest import mock

    fake = mock.Mock()
    fake.cache_config.cache_dtype = "ultraquant_4bit"
    with mock.patch(
        "vllm.v1.attention.backends.turboquant_attn.get_current_vllm_config_or_none",
        return_value=fake,
    ):
        assert TurboQuantAttentionBackend.get_preferred_block_size(16) == 64

    assert TurboQuantAttentionBackend.get_preferred_block_size(16) == 16


def test_turboquant_kernel_block_sizes_exclude_256():
    assert TurboQuantAttentionBackend.get_supported_kernel_block_sizes() == [
        16,
        32,
        64,
        128,
    ]


def test_ultraquant_flydsl_decode_routing():
    """CPU check of FlyDSL vs unified-Triton decode eligibility."""
    eligible = dict(
        head_size=256,
        num_kv_groups=8,
        has_sinks=False,
        sliding_window=None,
        flydsl_loaded=True,
    )
    assert ultraquant_flydsl_decode_eligible(**eligible)
    for gqa in (6, 8, 16):
        assert ultraquant_flydsl_decode_eligible(**{**eligible, "num_kv_groups": gqa})

    assert not ultraquant_flydsl_decode_eligible(**{**eligible, "flydsl_loaded": False})
    assert not ultraquant_flydsl_decode_eligible(**{**eligible, "head_size": 128})
    assert not ultraquant_flydsl_decode_eligible(**{**eligible, "num_kv_groups": 4})
    assert not ultraquant_flydsl_decode_eligible(**{**eligible, "has_sinks": True})
    assert not ultraquant_flydsl_decode_eligible(**{**eligible, "sliding_window": 4096})
