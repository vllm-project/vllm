# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Platform._align_hybrid_block_size's --block-size warning."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf


def _make_backend_cls(
    supported_sizes: list[int | MultipleOf],
    preferred_block_size: int | None = None,
) -> MagicMock:
    """A backend double. `preferred_block_size` defaults to the backend's own
    minimum, i.e. a backend with no preference beyond what it supports."""
    backend_cls = MagicMock()
    backend_cls.get_name.return_value = "MockAttentionBackend"
    backend_cls.get_supported_kernel_block_sizes.return_value = supported_sizes
    if preferred_block_size is None:
        preferred_block_size = min(
            s.base if isinstance(s, MultipleOf) else s for s in supported_sizes
        )
    backend_cls.get_preferred_block_size.return_value = preferred_block_size
    return backend_cls


def _make_vllm_config(block_size: int | None, use_mla: bool) -> MagicMock:
    cache_config = CacheConfig(block_size=block_size, mamba_cache_mode="none")
    model_config = MagicMock()
    model_config.use_mla = use_mla
    model_config.is_hybrid = True
    model_config.dtype = torch.float16
    model_config.architecture = "FakeHybridForCausalLM"
    model_config.get_num_kv_heads.return_value = 1
    model_config.get_head_size.return_value = 64
    model_config.get_mamba_chunk_size.return_value = 256

    vllm_config = MagicMock()
    vllm_config.cache_config = cache_config
    vllm_config.model_config = model_config
    return vllm_config


def _resolve_fake_hybrid_model_cls():
    model_cls = MagicMock()
    model_cls.get_mamba_state_shape_from_config.return_value = ((1, 8, 16),)
    model_cls.get_mamba_state_dtype_from_config.return_value = (torch.float16,)
    return patch(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        return_value=(model_cls, "FakeHybridForCausalLM"),
    )


def _run_align_hybrid_block_size(
    block_size: int | None,
    use_mla: bool,
    supported_sizes: list[int | MultipleOf],
    preferred_block_size: int | None = None,
) -> MagicMock:
    vllm_config = _make_vllm_config(block_size, use_mla)
    backend_cls = _make_backend_cls(supported_sizes, preferred_block_size)
    with _resolve_fake_hybrid_model_cls():
        Platform._align_hybrid_block_size(vllm_config, backend_cls)
    return backend_cls


@pytest.mark.parametrize(
    "block_size,use_mla,supported_sizes,should_warn",
    [
        # Not explicitly set: never warn, whatever the backend/MLA floor is.
        (None, True, [MultipleOf(64)], False),
        (None, False, [MultipleOf(64)], False),
        # Below the backend minimum: discarded, must warn.
        (8, False, [MultipleOf(64)], True),
        # At the backend minimum: still discarded (max() is a no-op), warn.
        (64, False, [MultipleOf(64)], True),
        # Above the backend minimum, non-MLA: takes effect, no warning.
        (65, False, [MultipleOf(64)], False),
        # Above the backend minimum but at/below the MLA 128 floor: this is
        # the case a comparison against only the backend minimum would miss.
        (100, True, [MultipleOf(64)], True),
        (128, True, [MultipleOf(64)], True),
        # Above the MLA floor: takes effect, no warning.
        (256, True, [MultipleOf(64)], False),
    ],
)
def test_block_size_warning_fires_only_when_value_is_discarded(
    block_size, use_mla, supported_sizes, should_warn
):
    with patch("vllm.platforms.interface.logger") as mock_logger:
        _run_align_hybrid_block_size(block_size, use_mla, supported_sizes)

    assert mock_logger.warning_once.called == should_warn


@pytest.mark.parametrize(
    "block_size,should_warn",
    [
        # Below the floor but also below the backend's own preference: the
        # backend would have picked 64 with the flag absent (Phase 1), so
        # this explicit value still changes the outcome and must not warn.
        # This is the FlashAttention-on-XPU case from the review: floor 16,
        # preferred 64.
        (8, False),
        (16, False),
        # Exactly the backend's preferred value: identical outcome to
        # omitting the flag, so it does warn despite being "high".
        (64, True),
        # Above the backend's preference: changes the outcome, no warning.
        (65, False),
    ],
)
def test_block_size_warning_accounts_for_backend_preference_above_floor(
    block_size, should_warn
):
    with patch("vllm.platforms.interface.logger") as mock_logger:
        _run_align_hybrid_block_size(
            block_size,
            use_mla=False,
            supported_sizes=[MultipleOf(16)],
            preferred_block_size=64,
        )

    assert mock_logger.warning_once.called == should_warn


def test_block_size_warning_message_reports_backend_name_and_alignment():
    with patch("vllm.platforms.interface.logger") as mock_logger:
        _run_align_hybrid_block_size(8, use_mla=False, supported_sizes=[MultipleOf(64)])

    assert mock_logger.warning_once.called
    args = mock_logger.warning_once.call_args.args
    assert args[1] == 8
    assert args[2] == "MockAttentionBackend"
    assert args[3] == 64
    assert args[4] == 64


def test_update_block_size_for_backend_warns_only_through_phase_one():
    """`_align_hybrid_block_size` is only Phase 2 of
    `update_block_size_for_backend`; going through the public entry point
    exercises Phase 1's `get_preferred_block_size` skip-on-explicit-flag
    behavior, which is what makes the floor-only comparison unsound."""
    backend_cls = _make_backend_cls([MultipleOf(16)], preferred_block_size=64)

    def _make_config(block_size):
        return _make_vllm_config(block_size, use_mla=False)

    with (
        patch.object(Platform, "_find_non_ssm_backend", return_value=backend_cls),
        _resolve_fake_hybrid_model_cls(),
    ):
        # No flag: Phase 1 picks the backend's preferred 64.
        no_flag_config = _make_config(None)
        Platform.update_block_size_for_backend(no_flag_config)
        assert no_flag_config.cache_config.block_size == 64

        # Explicit value at/below the floor but below the backend's
        # preference: Phase 1 is skipped, so this does change the outcome
        # relative to the no-flag run above and must not warn.
        with patch("vllm.platforms.interface.logger") as mock_logger:
            low_config = _make_config(16)
            Platform.update_block_size_for_backend(low_config)
        assert low_config.cache_config.block_size == 16
        assert not mock_logger.warning_once.called

        # Explicit value equal to what Phase 1 would have chosen anyway:
        # identical outcome to omitting the flag, so it does warn.
        with patch("vllm.platforms.interface.logger") as mock_logger:
            same_as_preferred_config = _make_config(64)
            Platform.update_block_size_for_backend(same_as_preferred_config)
        assert same_as_preferred_config.cache_config.block_size == 64
        assert mock_logger.warning_once.called


@pytest.mark.parametrize(
    "block_size,should_warn",
    [
        # A platform that resolves 128 with the flag absent. Passing 16 lands
        # at 16, so the flag does change the outcome and must not warn -- the
        # backend's own preference of 16 is not what an omitted flag gives on
        # such a platform. This is the CPU case from the review.
        (16, False),
        (64, False),
        # Exactly what the platform resolves anyway: identical outcome.
        (128, True),
        (256, False),
    ],
)
def test_block_size_warning_uses_the_platform_no_flag_value(block_size, should_warn):
    class PlatformWithOwnDefault(Platform):
        @classmethod
        def block_size_without_user_flag(cls, backend_cls) -> int:
            return 128

    vllm_config = _make_vllm_config(block_size, use_mla=False)
    backend_cls = _make_backend_cls([MultipleOf(16)])
    with (
        _resolve_fake_hybrid_model_cls(),
        patch("vllm.platforms.interface.logger") as mock_logger,
    ):
        PlatformWithOwnDefault._align_hybrid_block_size(vllm_config, backend_cls)

    assert mock_logger.warning_once.called == should_warn


def test_cpu_platform_no_flag_block_size_matches_what_it_configures():
    """CpuPlatform skips Phase 1, so its no-flag value has to come from
    `check_and_update_config`, which sets an unspecified block size to 128."""
    from vllm.platforms.cpu import CpuPlatform

    backend_cls = _make_backend_cls([MultipleOf(16)], preferred_block_size=16)

    assert CpuPlatform.block_size_without_user_flag(backend_cls) == 128
