# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for attention backend selectors."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.attention.selector import AttentionSelectorConfig

# ROCm-specific attention backend selection tests
pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    config = MagicMock()
    config.model_config.dtype = torch.float16
    config.model_config.hf_config.architectures = ["LlamaForCausalLM"]
    config.cache_config.block_size = 16
    return config


@pytest.fixture
def mock_on_gfx9():
    """Mock gfx9 arch detection to return True."""
    with patch("vllm.platforms.rocm.on_gfx9", return_value=True):
        yield


@pytest.fixture
def mock_on_mi3xx():
    """Mock mi3xx arch detection to return True."""
    with patch("vllm.platforms.rocm.on_mi3xx", return_value=True):
        yield


@pytest.mark.parametrize(
    "env_vars, selected_backend, expected_backend_path",
    [
        # Test Case: Explicit FLEX_ATTENTION backend
        (
            {},
            "FLEX_ATTENTION",
            AttentionBackendEnum.FLEX_ATTENTION.get_path(),
        ),
        # Test Case 1: Default (no env vars, no explicit backend)
        (
            {},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 2: Explicit TRITON_ATTN backend
        (
            {},
            "TRITON_ATTN",
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Test Case 3: Explicit ROCM_ATTN backend
        (
            {},
            "ROCM_ATTN",
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 4: Explicit ROCM_AITER_FA backend
        (
            {},
            "ROCM_AITER_FA",
            AttentionBackendEnum.ROCM_AITER_FA.get_path(),
        ),
        # Test Case 5: Explicit ROCM_AITER_UNIFIED_ATTN backend
        (
            {},
            "ROCM_AITER_UNIFIED_ATTN",
            AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path(),
        ),
        # Test Case 6: VLLM_ROCM_USE_AITER=1
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 7: VLLM_ROCM_USE_AITER=1 + explicit TRITON_ATTN
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "TRITON_ATTN",
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Test Case 8: VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_USE_AITER_MHA=0
        (
            {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MHA": "0"},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Test Case 9: VLLM_ROCM_USE_AITER=1 + explicit ROCM_ATTN
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "ROCM_ATTN",
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
    ],
)
def test_standard_attention_backend_selection(
    env_vars,
    selected_backend,
    expected_backend_path,
    mock_vllm_config,
    mock_on_gfx9,
    mock_on_mi3xx,
    monkeypatch,
):
    """Test standard attention backend selection with various configurations."""
    # Set environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Import after setting env vars to ensure they're picked up
    # Reload envs to pick up new environment variables
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)

    # Convert string backend to enum if provided
    backend_enum = None
    if selected_backend:
        backend_enum = getattr(AttentionBackendEnum, selected_backend)

    # Get the backend class path
    from vllm.platforms.rocm import RocmPlatform

    # The AITER unified attention kernel only supports BF16/FP8 KV caches
    # (its 3D kernel asserts on fp16), so it must be selected with bf16.
    dtype = (
        torch.bfloat16
        if selected_backend == "ROCM_AITER_UNIFIED_ATTN"
        else torch.float16
    )

    attn_selector_config = AttentionSelectorConfig(
        head_size=128,
        dtype=dtype,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
    )

    backend_path = RocmPlatform.get_attn_backend_cls(
        selected_backend=backend_enum, attn_selector_config=attn_selector_config
    )

    assert backend_path == expected_backend_path


@pytest.mark.parametrize(
    "env_vars, selected_backend, block_size, expected_backend_path, should_raise",
    [
        # Test Case 1: TRITON_MLA with block_size != 1
        (
            {},
            "TRITON_MLA",
            16,
            AttentionBackendEnum.TRITON_MLA.get_path(),
            False,
        ),
        # Test Case 2: TRITON_MLA with block_size == 1 (should raise)
        (
            {},
            "TRITON_MLA",
            1,
            None,
            True,
        ),
        # Test Case 3: ROCM_AITER_MLA with block_size == 1
        (
            {},
            "ROCM_AITER_MLA",
            1,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 4: ROCM_AITER_MLA with block_size != 1 (should raise)
        (
            {},
            "ROCM_AITER_MLA",
            16,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 5: VLLM_ROCM_USE_AITER=1 with block_size == 1
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            1,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 6: VLLM_ROCM_USE_AITER=1 with block_size == 16
        # (should use ROCM_AITER_MLA now, as it supports block_size 16)
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            16,
            AttentionBackendEnum.ROCM_AITER_MLA.get_path(),
            False,
        ),
        # Test Case 7: VLLM_ROCM_USE_AITER=1 + explicit TRITON_MLA
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "TRITON_MLA",
            16,
            AttentionBackendEnum.TRITON_MLA.get_path(),
            False,
        ),
        # Test Case 8: Explicit ROCM_AITER_TRITON_MLA
        (
            {},
            "ROCM_AITER_TRITON_MLA",
            16,
            AttentionBackendEnum.ROCM_AITER_TRITON_MLA.get_path(),
            False,
        ),
    ],
)
def test_mla_backend_selection(
    env_vars,
    selected_backend,
    block_size,
    expected_backend_path,
    should_raise,
    mock_vllm_config,
    monkeypatch,
):
    """Test MLA backend selection with various configurations."""
    # Set environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    # Import after setting env vars
    # Reload envs
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)

    # Mock is_aiter_mla_enabled based on env vars and block_size
    aiter_enabled = env_vars.get("VLLM_ROCM_USE_AITER") == "1"

    mock_rocm_ops = MagicMock()
    mock_rocm_ops.is_mla_enabled.return_value = aiter_enabled
    mock_aiter_module = MagicMock()
    mock_aiter_module.rocm_aiter_ops = mock_rocm_ops

    with patch.dict("sys.modules", {"vllm._aiter_ops": mock_aiter_module}):
        # Convert string backend to enum if provided
        backend_enum = None
        if selected_backend:
            backend_enum = getattr(AttentionBackendEnum, selected_backend)

        from vllm.platforms.rocm import RocmPlatform

        if should_raise:
            with pytest.raises(ValueError):
                attn_selector_config = AttentionSelectorConfig(
                    head_size=128,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=block_size,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=False,
                )
                attn_selector_config = AttentionSelectorConfig(
                    head_size=128,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=block_size,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=False,
                )
                backend_path = RocmPlatform.get_attn_backend_cls(
                    selected_backend=backend_enum,
                    attn_selector_config=attn_selector_config,
                )

        else:
            attn_selector_config = AttentionSelectorConfig(
                head_size=128,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )

            backend_path = RocmPlatform.get_attn_backend_cls(
                selected_backend=backend_enum, attn_selector_config=attn_selector_config
            )

            assert backend_path == expected_backend_path


def test_aiter_fa_requires_mi3xx(mock_vllm_config):
    """Test that ROCM_AITER_FA requires mi3xx architecture."""
    from vllm.platforms.rocm import RocmPlatform

    # Mock on_mi3xx to return False (used by supports_compute_capability)
    with (
        patch("vllm.platforms.rocm.on_mi3xx", return_value=False),
        pytest.raises(
            ValueError,
            match="compute capability not supported",
        ),
    ):
        attn_selector_config = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )

        RocmPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.ROCM_AITER_FA,
            attn_selector_config=attn_selector_config,
        )


def test_sparse_not_supported(mock_vllm_config):
    """Test that sparse MLA without use_mla flag raises an error."""
    from vllm.platforms.rocm import RocmPlatform

    with pytest.raises(
        ValueError,
        match="No valid attention backend found",
    ):
        attn_selector_config = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=True,
        )

        RocmPlatform.get_attn_backend_cls(
            selected_backend=None, attn_selector_config=attn_selector_config
        )


def test_rocm_attn_content_packed_kv_cache_contract():
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend

    assert RocmAttentionBackend.supports_sink()
    assert RocmAttentionBackend.supports_kv_connector()
    assert RocmAttentionBackend.supports_head_size(512)
    assert RocmAttentionBackend.get_kv_cache_shape(3, 16, 2, 8) == (
        3,
        2,
        16,
        16,
    )

    try:
        set_kv_cache_layout("HND")
        assert RocmAttentionBackend.indexes_kv_by_block_stride()
        assert RocmAttentionBackend.get_kv_cache_stride_order() == (0, 1, 2, 3)
        assert RocmAttentionBackend.get_kv_cache_stride_order(True) == (
            1,
            2,
            0,
            3,
            4,
        )

        set_kv_cache_layout("NHD")
        assert RocmAttentionBackend.indexes_kv_by_block_stride()
        assert RocmAttentionBackend.get_kv_cache_stride_order() == (0, 2, 1, 3)
        assert RocmAttentionBackend.get_kv_cache_stride_order(True) == (
            1,
            0,
            3,
            2,
            4,
        )
    finally:
        set_kv_cache_layout(None)


def test_rocm_attn_decode_segment_policy():
    # The 3D split-KV segment count derives from a single launch-grid target and
    # the per-rank KV-head count -- no per-model tuning. It must reach the target
    # occupancy even for low-KV-head shards (e.g. Llama-70B TP8 -> 1 KV head),
    # which is exactly the case the earlier per-model heuristic missed.
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadataBuilder

    segments = RocmAttentionMetadataBuilder._decode_split_kv_segments

    assert segments(1) == 128  # Llama-70B TP8 (1 KV head) -> full split
    assert segments(4) == 32  # Qwen3.6-27B TP2
    assert segments(8) == 16  # gemma-4-31B TP2
    # More KV heads than the target still splits into at least one segment.
    assert segments(256) == 1


def test_rocm_aiter_unified_stride_order_matches_shape():
    # RocmAiterUnifiedAttention subclasses RocmAttentionBackend but keeps the
    # legacy 5D contiguous KV layout, so it must NOT inherit the parent's
    # content-packed 4D stride order: the runner asserts
    # len(stride_order) == len(shape) and does not catch AssertionError, so a
    # mismatch crashes KV-cache init for every model on that backend.
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    shape = RocmAiterUnifiedAttentionBackend.get_kv_cache_shape(100, 16, 8, 128)
    order = RocmAiterUnifiedAttentionBackend.get_kv_cache_stride_order()
    order_l = RocmAiterUnifiedAttentionBackend.get_kv_cache_stride_order(True)
    assert len(order) == len(shape)
    assert sorted(order) == list(range(len(shape)))
    assert sorted(order_l) == list(range(len(shape) + 1))


def test_rocm_attn_content_packed_split_views():
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        has_native_kv_cache_layout,
    )
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl

    impl = RocmAttentionImpl(
        num_heads=2,
        head_size=16,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        attn_type=AttentionType.DECODER,
    )
    kv_cache = torch.arange(3 * 2 * 4 * 32, dtype=torch.float16).reshape(3, 2, 4, 32)

    key_update, value_update = impl._split_kv_cache_for_update(kv_cache)
    assert key_update.shape == (3, 4, 2, 16)
    assert value_update.shape == (3, 4, 2, 16)
    assert key_update[1, 3, 0, 5] == kv_cache[1, 0, 3, 5]
    assert value_update[1, 3, 0, 5] == kv_cache[1, 0, 3, 21]

    key_decode, value_decode = impl._split_kv_cache_for_unified_attention(kv_cache)
    assert key_decode.shape == (3, 4, 2, 16)
    assert value_decode.shape == (3, 4, 2, 16)
    assert key_decode[1, 3, 0, 5] == kv_cache[1, 0, 3, 5]
    assert value_decode[1, 3, 0, 5] == kv_cache[1, 0, 3, 21]
    assert not has_native_kv_cache_layout(key_decode, value_decode)

    native_key = torch.empty(3, 2, 2, 4, 8)
    native_value = torch.empty(3, 2, 16, 4)
    assert has_native_kv_cache_layout(native_key, native_value)
