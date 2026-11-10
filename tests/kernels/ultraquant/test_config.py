# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only integration tests for the UltraQuant cache dtype."""

from vllm.v1.attention.backends.ultraquant_attn import (
    UltraQuantAttentionBackend,
)
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    ultraquant_flydsl_decode_eligible,
)


def test_ultraquant_prefers_block_size_64():
    # FlyDSL D=256 decode geometry requires a 64-token KV block.
    assert UltraQuantAttentionBackend.get_preferred_block_size(16) == 64
    assert UltraQuantAttentionBackend.get_preferred_block_size(128) == 64


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
