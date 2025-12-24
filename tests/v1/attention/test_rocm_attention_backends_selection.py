# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for attention backend selectors."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.platforms import current_platform

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
    """Mock the on_gfx9 function to return True."""
    with patch("vllm.platforms.rocm.on_gfx9", return_value=True):
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
            AttentionBackendEnum.TRITON_ATTN.get_path(),
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
        # (defaults to AITER FA when MHA not explicitly disabled)
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            AttentionBackendEnum.ROCM_AITER_FA.get_path(),
        ),
        # Test Case 7: VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_USE_AITER_MHA=1
        (
            {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MHA": "1"},
            None,
            AttentionBackendEnum.ROCM_AITER_FA.get_path(),
        ),
        # Test Case 8: VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
        (
            {
                "VLLM_ROCM_USE_AITER": "1",
                "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": "1",
            },
            None,
            AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN.get_path(),
        ),
        # Test Case 9: VLLM_ROCM_USE_AITER=1 + explicit TRITON_ATTN
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            "TRITON_ATTN",
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Test Case 10: VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_USE_AITER_MHA=0
        # (explicitly disabled)
        (
            {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MHA": "0"},
            None,
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Test Case 11: VLLM_ROCM_USE_AITER=1 + explicit ROCM_ATTN
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

    backend_path = RocmPlatform.get_attn_backend_cls(
        selected_backend=backend_enum,
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
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
                RocmPlatform.get_attn_backend_cls(
                    selected_backend=backend_enum,
                    head_size=128,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=block_size,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=False,
                )
        else:
            backend_path = RocmPlatform.get_attn_backend_cls(
                selected_backend=backend_enum,
                head_size=128,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )
            assert backend_path == expected_backend_path


def test_aiter_fa_requires_gfx9(mock_vllm_config):
    """Test that ROCM_AITER_FA requires gfx9 architecture."""
    from vllm.platforms.rocm import RocmPlatform

    # Mock on_gfx9 to return False
    with (
        patch("vllm.platforms.rocm.on_gfx9", return_value=False),
        pytest.raises(
            ValueError,
            match="only supported on gfx9",
        ),
    ):
        RocmPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.ROCM_AITER_FA,
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )


def test_sparse_not_supported(mock_vllm_config):
    """Test that sparse attention is not supported on ROCm."""
    from vllm.platforms.rocm import RocmPlatform

    with pytest.raises(
        AssertionError, match="Sparse MLA backend on ROCm only supports block size 1"
    ):
        RocmPlatform.get_attn_backend_cls(
            selected_backend=None,
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=True,
        )
