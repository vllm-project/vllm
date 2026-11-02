# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-V2.6 has head_dim 192 but v_head_dim 128, so only backends that keep
V's width out of the K path can serve it. An explicit ``--attention-backend``
from that set has to be honored; any other request has to fall back to the
DiffKV auto-pick with a warning rather than mount a backend that cannot serve
the shape.
"""

import pytest
import torch

from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

HEAD_DIM = 192
V_HEAD_DIM = 128

DIFFKV_BACKENDS = (
    AttentionBackendEnum.FLASH_ATTN_DIFFKV,
    AttentionBackendEnum.TRITON_ATTN_DIFFKV,
)


@pytest.fixture(autouse=True)
def bf16_default_dtype():
    # Backend selection is dtype sensitive; use the dtype the model ships in.
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(default_dtype)


@pytest.fixture(autouse=True)
def restore_diffkv_head_size_v():
    """``set_head_size_v`` writes backend class state.

    Leaving it modified would make these tests order dependent, so snapshot and
    restore it on every backend that can be selected here.
    """
    from vllm.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVBackend,
    )
    from vllm.v1.attention.backends.triton_attn_diffkv import (
        TritonAttentionDiffKVBackend,
    )

    backends = (FlashAttentionDiffKVBackend, TritonAttentionDiffKVBackend)
    saved = [backend.head_size_v for backend in backends]
    yield
    for backend, head_size_v in zip(backends, saved):
        backend.head_size_v = head_size_v


def _build_attention(
    backend: AttentionBackendEnum | None, v_head_dim: int = V_HEAD_DIM
):
    from vllm.model_executor.models.mimo_v2 import MiMoV2Attention

    config = VllmConfig(attention_config=AttentionConfig(backend=backend))
    with set_current_vllm_config(config):
        layer = MiMoV2Attention(
            hidden_size=512,
            num_heads=4,
            num_kv_heads=2,
            head_dim=HEAD_DIM,
            v_head_dim=v_head_dim,
            prefix="model.layers.0.self_attn",
        )
    return layer


def test_default_backend_is_diffkv(dist_init) -> None:
    """No explicit backend: the DiffKV auto-pick is unchanged."""
    layer = _build_attention(None)
    assert layer.attn.backend in DIFFKV_BACKENDS
    # V's width has to reach the layer: Attention defaults head_size_v to
    # head_size (192), which would corrupt the output reshape and KV spec.
    assert layer.attn.head_size_v == V_HEAD_DIM


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCM_AITER_UNIFIED_ATTN is ROCm only"
)
def test_explicit_unified_attention_is_honored(dist_init) -> None:
    from vllm._aiter_ops import IS_AITER_FOUND

    if not IS_AITER_FOUND:
        pytest.skip("aiter is not installed")

    layer = _build_attention(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
    assert layer.attn.backend is AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN
    # The enum alone would not catch the bug this guards: the impl has to be
    # unified attention's, not the DiffKV one the auto-pick installs.
    assert type(layer.attn.impl).__name__ == "RocmAiterUnifiedAttentionImpl"


def test_explicit_diffkv_backend_is_honored(dist_init) -> None:
    """The accept path also works for the DiffKV family, on any platform."""
    layer = _build_attention(AttentionBackendEnum.TRITON_ATTN_DIFFKV)
    assert layer.attn.backend is AttentionBackendEnum.TRITON_ATTN_DIFFKV
    assert layer.attn.head_size_v == V_HEAD_DIM


def test_explicit_symmetric_backend_falls_back_to_diffkv(dist_init) -> None:
    """A backend that cannot take head_size_v != head_size must not be mounted."""
    layer = _build_attention(AttentionBackendEnum.TRITON_ATTN)
    assert layer.attn.backend in DIFFKV_BACKENDS


def test_layer_v_head_dim_reaches_the_backend(dist_init) -> None:
    """The backend must be told V's width before its impl is built.

    The DiffKV backends default to ``head_size_v = 128``, which is this model's
    own V width, so a missing ``set_head_size_v`` call is invisible at the
    model's geometry. That attribute is what the metadata builder and forward
    read, so assert it carries the layer's width instead of the default.
    """
    from vllm.v1.attention.backends.triton_attn_diffkv import (
        TritonAttentionDiffKVBackend,
    )

    _build_attention(AttentionBackendEnum.TRITON_ATTN_DIFFKV, v_head_dim=64)
    assert TritonAttentionDiffKVBackend.head_size_v == 64


def test_only_backends_that_can_serve_v_head_size_are_admitted() -> None:
    """`supports_head_size_v` is what the model's gate consults."""
    from vllm.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVBackend,
    )
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )
    from vllm.v1.attention.backends.triton_attn_diffkv import (
        TritonAttentionDiffKVBackend,
    )

    assert FlashAttentionDiffKVBackend.supports_head_size_v(HEAD_DIM, V_HEAD_DIM)
    assert TritonAttentionDiffKVBackend.supports_head_size_v(HEAD_DIM, V_HEAD_DIM)
    # The packed cache is split with `split(head_size)`, so V cannot be wider.
    assert RocmAiterUnifiedAttentionBackend.supports_head_size_v(HEAD_DIM, V_HEAD_DIM)
    assert not RocmAiterUnifiedAttentionBackend.supports_head_size_v(
        V_HEAD_DIM, HEAD_DIM
    )


def test_cache_write_matches_pytorch_reference(dist_init) -> None:
    """Fill the packed cache through the impl, then compare with torch.

    The layer hands the kernel two views of one layer buffer, and V has its own
    width, so the check is what lands in that buffer: K in the first
    ``head_dim`` slots of a row, V in the remaining ``v_head_dim``. Same shape
    as the aiter FA check in tests/kernels/attention/test_rocm_aiter_fa.py, for
    the asymmetric case those backends cannot serve.
    """
    from vllm._aiter_ops import IS_AITER_FOUND

    if not IS_AITER_FOUND:
        pytest.skip("aiter is not installed")

    from tests.kernels.allclose_default import get_default_atol, get_default_rtol

    layer = _build_attention(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
    impl = layer.attn.impl
    num_kv_heads = layer.num_kv_heads
    num_tokens, block_size, num_blocks = 6, 16, 2

    # One packed layer buffer per token row: [K head_dim | V v_head_dim]. The
    # spare trailing block absorbs an over-wide write from a regressed kernel,
    # so that fails the comparison instead of writing out of bounds.
    packed = torch.zeros(
        num_blocks + 1,
        num_kv_heads,
        block_size,
        HEAD_DIM + V_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.randn(
        num_tokens, num_kv_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    value = torch.randn(
        num_tokens, num_kv_heads, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    # Slot 31 is the last slot of the cache.
    slot_mapping = torch.tensor([0, 5, 16, 31, 7, 17], dtype=torch.long, device="cuda")

    # The impl reads only the scales off `layer`; they are not the subject here.
    scales = torch.nn.Module()
    scales._k_scale = torch.ones((), dtype=torch.float32, device="cuda")
    scales._v_scale = torch.ones((), dtype=torch.float32, device="cuda")
    impl.do_kv_cache_update(scales, key, value, packed, slot_mapping)

    expected = torch.zeros_like(packed)
    for token, slot in enumerate(slot_mapping.tolist()):
        block, offset = divmod(slot, block_size)
        # packed buffer is [blocks, kv_heads, block_tokens, content]
        expected[block, :, offset, :HEAD_DIM] = key[token]
        expected[block, :, offset, HEAD_DIM:] = value[token]

    torch.testing.assert_close(
        packed, expected, atol=get_default_atol(packed), rtol=get_default_rtol(packed)
    )
