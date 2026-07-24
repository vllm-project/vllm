# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for attention backend selectors."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
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

    attn_selector_config = AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
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


# ---------------------------------------------------------------------------
# V1: Hybrid model block-size regression tests (issue #36994 / PR #36274)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block_size, expected_backend",
    [
        # Standard block sizes: must still work
        (16, AttentionBackendEnum.ROCM_ATTN),
        (32, AttentionBackendEnum.ROCM_ATTN),
        # Qwen3.5 hybrid model block sizes (multiples of 16, non-standard)
        # block_size=784: computed for Qwen3.5-14B on certain configs
        (784, AttentionBackendEnum.ROCM_ATTN),
        # block_size=1056: computed for Qwen3.5 when mamba page aligns to 1056 tokens
        (1056, AttentionBackendEnum.ROCM_ATTN),
        # block_size=544: Qwen3-Next style hybrid block size
        (544, AttentionBackendEnum.ROCM_ATTN),
        # block_size=512: another common hybrid-computed value
        (512, AttentionBackendEnum.ROCM_ATTN),
    ],
)
def test_hybrid_model_block_sizes_accepted(
    block_size,
    expected_backend,
    mock_vllm_config,
    mock_on_gfx9,
    mock_on_mi3xx,
):
    """Test that non-standard block sizes from hybrid models (e.g. Qwen3.5)
    are accepted by the ROCm attention backend selector.

    Regression test for: supports_block_size wrongly rejected dynamically
    computed block sizes (fixed in PR #36274, issue #36994).
    Non-standard multiples of 16 must pass, not just {1,8,16,32,64,128,256}.
    """
    import importlib

    import vllm.envs as envs

    importlib.reload(envs)

    from vllm.platforms.rocm import RocmPlatform

    attn_selector_config = AttentionSelectorConfig(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=block_size,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
    )

    # Must not raise — regression guard for issue #36994
    backend_path = RocmPlatform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=attn_selector_config,
    )

    assert backend_path == expected_backend.get_path()


def test_supports_block_size_rocm_attn():
    """RocmAttentionBackend accepts any multiple of 16, rejects non-multiples.

    Unit test for the MultipleOf(16) kernel block-size contract introduced by
    PR #36274 to fix issue #36994 (Qwen3.5 hybrid models on ROCm).
    """
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend

    # Standard sizes must be accepted
    assert RocmAttentionBackend.supports_block_size(16)
    assert RocmAttentionBackend.supports_block_size(32)
    # Qwen3.5 / Qwen3-Next hybrid model block sizes must also be accepted
    assert RocmAttentionBackend.supports_block_size(784)
    assert RocmAttentionBackend.supports_block_size(1056)
    assert RocmAttentionBackend.supports_block_size(544)
    assert RocmAttentionBackend.supports_block_size(512)
    # Non-multiples of 16 must be rejected
    assert not RocmAttentionBackend.supports_block_size(15)
    assert not RocmAttentionBackend.supports_block_size(1)


# ---------------------------------------------------------------------------
# V2: RDNA3/RDNA4 (gfx11xx/gfx12xx) backend selection tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_on_gfx1x():
    """Mock gfx1x arch detection (RDNA3/RDNA4) to return True."""
    with patch("vllm.platforms.rocm.on_gfx1x", return_value=True):
        yield


@pytest.mark.parametrize(
    "env_vars, selected_backend, expected_backend_path",
    [
        # Case 1: gfx1151 default (no env vars, no explicit backend)
        # ROCM_ATTN has no compute cap restriction, should be selected
        (
            {},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Case 2: gfx1151 with explicit TRITON_ATTN
        # TritonAttentionBackend.supports_compute_capability always returns True
        (
            {},
            "TRITON_ATTN",
            AttentionBackendEnum.TRITON_ATTN.get_path(),
        ),
        # Case 3: gfx1151 + VLLM_ROCM_USE_AITER=1 (aiter not available on RDNA)
        # Falls back to ROCM_ATTN since is_mha_enabled() returns False
        (
            {"VLLM_ROCM_USE_AITER": "1"},
            None,
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
        # Case 4: explicit ROCM_ATTN on gfx1151
        (
            {},
            "ROCM_ATTN",
            AttentionBackendEnum.ROCM_ATTN.get_path(),
        ),
    ],
)
def test_gfx1x_rdna_attention_backend_selection(
    env_vars,
    selected_backend,
    expected_backend_path,
    mock_vllm_config,
    mock_on_gfx1x,
    monkeypatch,
):
    """Test attention backend selection on RDNA3/RDNA4 (gfx11xx/gfx12xx) GPUs.

    RDNA devices are NOT mi3xx and NOT gfx9. Tests ensure:
    - Default backend selection does not require MI3XX-only backends
    - ROCM_AITER_FA is correctly rejected (mi3xx required)
    - TRITON_ATTN is always valid as fallback
    - VLLM_ROCM_USE_AITER=1 still falls back gracefully when aiter unavailable
    """
    # on_gfx9 and on_mi3xx must return False for a pure RDNA simulation
    with (
        patch("vllm.platforms.rocm.on_gfx9", return_value=False),
        patch("vllm.platforms.rocm.on_mi3xx", return_value=False),
    ):
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        import importlib

        import vllm.envs as envs

        importlib.reload(envs)

        backend_enum = None
        if selected_backend:
            backend_enum = getattr(AttentionBackendEnum, selected_backend)

        from vllm.platforms.rocm import RocmPlatform

        attn_selector_config = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )

        backend_path = RocmPlatform.get_attn_backend_cls(
            selected_backend=backend_enum,
            attn_selector_config=attn_selector_config,
        )

        assert backend_path == expected_backend_path


def test_aiter_fa_rejected_on_gfx1x(mock_vllm_config):
    """ROCM_AITER_FA must be rejected on RDNA hardware (requires mi3xx).

    Validates that explicitly selecting ROCM_AITER_FA on a gfx1151 (RDNA4)
    device raises ValueError with a clear "compute capability not supported"
    message, because ROCM_AITER_FA.supports_compute_capability checks
    on_mi3xx() which returns False on RDNA.
    """
    from vllm.platforms.rocm import RocmPlatform

    with (
        patch("vllm.platforms.rocm.on_gfx9", return_value=False),
        patch("vllm.platforms.rocm.on_mi3xx", return_value=False),
        patch("vllm.platforms.rocm.on_gfx1x", return_value=True),
        pytest.raises(ValueError, match="compute capability not supported"),
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


def test_gfx1x_with_triton_flash_attn_enabled(mock_vllm_config):
    """On gfx1x with flash_attn_triton_available mocked True, the main
    attention path still selects ROCM_ATTN (not FLASH_ATTN).

    flash_attn_triton_available() only affects the ViT attention path
    (get_vit_attn_backend), not the main LLM decode/prefill path handled by
    get_attn_backend_cls. The main path on RDNA selects ROCM_ATTN as the
    highest-priority backend since it has no compute capability restriction.
    """
    with (
        patch("vllm.platforms.rocm.on_gfx9", return_value=False),
        patch("vllm.platforms.rocm.on_mi3xx", return_value=False),
        patch("vllm.platforms.rocm.on_gfx1x", return_value=True),
        patch("vllm.platforms.rocm.flash_attn_triton_available", return_value=True),
    ):
        from vllm.platforms.rocm import RocmPlatform

        attn_selector_config = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )

        # Main attention path: ROCM_ATTN (flash_attn_triton only affects ViT)
        backend_path = RocmPlatform.get_attn_backend_cls(
            selected_backend=None,
            attn_selector_config=attn_selector_config,
        )
        assert backend_path == AttentionBackendEnum.ROCM_ATTN.get_path()
