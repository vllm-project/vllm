# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Helion rotary embedding kernels.

Structure mirrors tests/kernels/helion/test_silu_mul_fp8.py.

Two levels of correctness testing:
  1. Against ``RotaryEmbedding.forward_static`` (the vLLM canonical reference).
  2. Against a pure-PyTorch inline implementation that is independent of
     vLLM C-extension availability.

NOTE: TestRotaryEmbeddingConfigPicker does NOT require Helion and can run
on any machine. Correctness and registration tests require Helion + CUDA.
"""

import regex as re

import pytest
import torch

from vllm.utils.import_utils import has_helion

_HELION_AVAILABLE = has_helion()

# Config picker is re-implemented here (mirroring the kernel file) so that
# TestRotaryEmbeddingConfigPicker can run without Helion being installed.
def _pick_rope_config_for_test(
    args, config_keys: list[str]
) -> str | None:
    if not config_keys:
        return None
    _pos, query, _key, _head_size, rotary_dim, _cache = args
    num_tokens = query.shape[0]
    parsed: dict[int, list[int]] = {}
    for k in config_keys:
        if k == "default":
            continue
        m = re.fullmatch(r"rotarydim_(\d+)_numtokens_(\d+)", k)
        if not m:
            raise ValueError(
                f"Malformed config key '{k}', "
                f"expected format 'rotarydim_{{int}}_numtokens_{{int}}'"
            )
        parsed.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    if not parsed:
        return "default" if "default" in config_keys else None
    best_rdim = min(parsed, key=lambda d: abs(d - rotary_dim))
    avail = sorted(parsed[best_rdim])
    best_n = next((n for n in avail if n >= num_tokens), avail[-1])
    return f"rotarydim_{best_rdim}_numtokens_{best_n}"


# These are always importable (stubs exist when Helion is unavailable).
from vllm.kernels.helion.ops.rotary_embedding import (
    rotary_embedding_baseline,
    rotary_embedding_gptj,
    rotary_embedding_neox,
)

if _HELION_AVAILABLE:
    from vllm.kernels.helion.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# Helper: skip when no pre-tuned config exists for this GPU
# ---------------------------------------------------------------------------

def _skip_if_platform_unsupported(kernel_name: str) -> None:
    """Skip the test if no pre-tuned configs are present for this GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        platform = get_canonical_gpu_name()
        try:
            config_manager = ConfigManager.get_instance()
        except RuntimeError:
            config_manager = ConfigManager()

        configs = config_manager.get_platform_configs(kernel_name, platform)
        if not configs:
            pytest.skip(
                f"Current GPU platform '{platform}' has no pre-tuned configs "
                f"for kernel '{kernel_name}'"
            )
    except (ImportError, RuntimeError, KeyError):
        pytest.skip(
            f"Error detecting platform support for kernel '{kernel_name}'"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_config_manager():
    if _HELION_AVAILABLE:
        from vllm.kernels.helion.config_manager import ConfigManager
        ConfigManager.reset_instance()
        ConfigManager()
    yield
    if _HELION_AVAILABLE:
        from vllm.kernels.helion.config_manager import ConfigManager
        ConfigManager.reset_instance()


# ---------------------------------------------------------------------------
# Pure-PyTorch reference (independent of vLLM C extension)
# ---------------------------------------------------------------------------

def _rope_pytorch(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch RoPE matching ``RotaryEmbedding.forward_static``."""
    num_tokens = positions.shape[0]
    rot_half = rotary_dim // 2

    cos_sin = cos_sin_cache.index_select(0, positions)  # [num_tokens, rotary_dim]
    cos = cos_sin[:, :rot_half]   # [num_tokens, rot_half]
    sin = cos_sin[:, rot_half:]   # [num_tokens, rot_half]

    def _apply(x_flat: torch.Tensor, n_heads: int) -> torch.Tensor:
        # x_flat: [num_tokens, n_heads * head_size]
        x = x_flat.reshape(num_tokens, n_heads, head_size)
        x_rot = x[:, :, :rotary_dim]   # [num_tokens, n_heads, rotary_dim]
        x_pass = x[:, :, rotary_dim:]  # unrotated tail

        cos_h = cos.unsqueeze(1).to(x.dtype)  # [num_tokens, 1, rot_half]
        sin_h = sin.unsqueeze(1).to(x.dtype)

        if is_neox_style:
            x1 = x_rot[:, :, :rot_half]
            x2 = x_rot[:, :, rot_half:]
            x_rot_out = torch.cat(
                (x1 * cos_h - x2 * sin_h, x2 * cos_h + x1 * sin_h), dim=-1
            )
        else:
            x1 = x_rot[:, :, ::2]
            x2 = x_rot[:, :, 1::2]
            x_rot_out = torch.stack(
                (x1 * cos_h - x2 * sin_h, x2 * cos_h + x1 * sin_h), dim=-1
            ).flatten(-2)

        x_out = torch.cat((x_rot_out, x_pass), dim=-1)
        return x_out.reshape_as(x_flat)

    num_q_heads = query.shape[1] // head_size
    num_kv_heads = key.shape[1] // head_size

    q_out = _apply(query, num_q_heads)
    k_out = _apply(key, num_kv_heads)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Config picker tests
# ---------------------------------------------------------------------------

class TestRotaryEmbeddingConfigPicker:
    """Tests for pick_rotary_embedding_config() selection logic."""

    def _make_args(
        self,
        num_tokens: int,
        rotary_dim: int,
        head_size: int = 128,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,
    ) -> tuple:
        positions = torch.zeros(num_tokens, dtype=torch.int64, device="cpu")
        query = torch.zeros(num_tokens, num_q_heads * head_size, device="cpu")
        key = torch.zeros(num_tokens, num_kv_heads * head_size, device="cpu")
        cos_sin_cache = torch.zeros(2048, rotary_dim, device="cpu")
        return positions, query, key, head_size, rotary_dim, cos_sin_cache

    def test_exact_match(self):
        keys = ["rotarydim_128_numtokens_32", "rotarydim_128_numtokens_64"]
        args = self._make_args(num_tokens=32, rotary_dim=128)
        assert _pick_rope_config_for_test(args, keys) == "rotarydim_128_numtokens_32"

    def test_ceiling_selection(self):
        """Smallest num_tokens >= input should be selected."""
        keys = [
            "rotarydim_128_numtokens_8",
            "rotarydim_128_numtokens_32",
            "rotarydim_128_numtokens_128",
        ]
        args = self._make_args(num_tokens=20, rotary_dim=128)
        assert _pick_rope_config_for_test(args, keys) == "rotarydim_128_numtokens_32"

    def test_fallback_to_largest(self):
        """When input exceeds all available num_tokens, pick the largest."""
        keys = [
            "rotarydim_128_numtokens_8",
            "rotarydim_128_numtokens_32",
            "rotarydim_128_numtokens_128",
        ]
        args = self._make_args(num_tokens=512, rotary_dim=128)
        assert _pick_rope_config_for_test(args, keys) == "rotarydim_128_numtokens_128"

    def test_closest_rotary_dim(self):
        """When exact rotary_dim not found, use closest available."""
        keys = [
            "rotarydim_64_numtokens_32",
            "rotarydim_128_numtokens_32",
        ]
        # rotary_dim=100 → closer to 128 (dist=28) than 64 (dist=36) → pick rotarydim_128_numtokens_32
        args = self._make_args(num_tokens=32, rotary_dim=100)
        assert _pick_rope_config_for_test(args, keys) == "rotarydim_128_numtokens_32"

    def test_fallback_to_default(self):
        args = self._make_args(num_tokens=32, rotary_dim=128)
        assert _pick_rope_config_for_test(args, ["default"]) == "default"

    def test_empty_config_keys(self):
        args = self._make_args(num_tokens=32, rotary_dim=128)
        assert _pick_rope_config_for_test(args, []) is None

    def test_malformed_key_raises(self):
        keys = ["rotarydim_128_badformat_32"]
        args = self._make_args(num_tokens=32, rotary_dim=128)
        with pytest.raises(ValueError, match="Malformed config key"):
            _pick_rope_config_for_test(args, keys)

    def test_default_skipped_when_valid_keys_exist(self):
        keys = [
            "default",
            "rotarydim_128_numtokens_32",
            "rotarydim_128_numtokens_64",
        ]
        args = self._make_args(num_tokens=32, rotary_dim=128)
        assert _pick_rope_config_for_test(args, keys) == "rotarydim_128_numtokens_32"


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestRotaryEmbeddingNeoxCorrectness:
    """Compare rotary_embedding_neox against RotaryEmbedding.forward_static."""

    @pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
    @pytest.mark.parametrize("rotary_dim", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_neox_vs_baseline(self, num_tokens, rotary_dim, dtype):
        _skip_if_platform_unsupported("_helion_rotary_embedding_neox")

        head_size = rotary_dim
        num_q_heads, num_kv_heads = 32, 8

        positions = torch.randint(0, 512, (num_tokens,), device="cuda")
        query = torch.randn(num_tokens, num_q_heads * head_size, device="cuda", dtype=dtype)
        key = torch.randn(num_tokens, num_kv_heads * head_size, device="cuda", dtype=dtype)
        cos_sin_cache = torch.randn(1024, rotary_dim, device="cuda", dtype=dtype)

        q_ref, k_ref = rotary_embedding_baseline(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache, is_neox_style=True,
        )
        q_helion, k_helion = rotary_embedding_neox(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache,
        )

        assert q_helion.shape == q_ref.shape
        assert k_helion.shape == k_ref.shape

        rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (2e-3, 2e-3)
        torch.testing.assert_close(
            q_helion.float(), q_ref.float(), rtol=rtol, atol=atol,
            msg=f"Query mismatch: tokens={num_tokens}, rotary_dim={rotary_dim}, dtype={dtype}",
        )
        torch.testing.assert_close(
            k_helion.float(), k_ref.float(), rtol=rtol, atol=atol,
            msg=f"Key mismatch: tokens={num_tokens}, rotary_dim={rotary_dim}, dtype={dtype}",
        )

    def test_neox_vs_pytorch(self):
        """Against pure-PyTorch reference (no C extension needed)."""
        _skip_if_platform_unsupported("_helion_rotary_embedding_neox")

        num_tokens, rotary_dim, head_size = 32, 128, 128
        num_q_heads, num_kv_heads = 16, 8
        dtype = torch.bfloat16

        positions = torch.randint(0, 512, (num_tokens,), device="cuda")
        query = torch.randn(num_tokens, num_q_heads * head_size, device="cuda", dtype=dtype)
        key = torch.randn(num_tokens, num_kv_heads * head_size, device="cuda", dtype=dtype)
        cos_sin_cache = torch.randn(1024, rotary_dim, device="cuda", dtype=dtype)

        q_ref, k_ref = _rope_pytorch(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache, is_neox_style=True,
        )
        q_helion, k_helion = rotary_embedding_neox(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache,
        )
        torch.testing.assert_close(q_helion.float(), q_ref.float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(k_helion.float(), k_ref.float(), rtol=2e-2, atol=2e-2)

    def test_neox_partial_rope(self):
        """rotary_dim < head_size (partial RoPE): un-rotated tail must be unchanged."""
        _skip_if_platform_unsupported("_helion_rotary_embedding_neox")

        num_tokens = 16
        head_size = 128
        rotary_dim = 64   # partial: rotate only first 64 of 128 dims
        num_q_heads, num_kv_heads = 8, 4
        dtype = torch.bfloat16

        positions = torch.randint(0, 512, (num_tokens,), device="cuda")
        query = torch.randn(num_tokens, num_q_heads * head_size, device="cuda", dtype=dtype)
        key = torch.randn(num_tokens, num_kv_heads * head_size, device="cuda", dtype=dtype)
        cos_sin_cache = torch.randn(1024, rotary_dim, device="cuda", dtype=dtype)

        q_ref, k_ref = rotary_embedding_baseline(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache, is_neox_style=True,
        )
        q_helion, k_helion = rotary_embedding_neox(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache,
        )
        torch.testing.assert_close(q_helion.float(), q_ref.float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(k_helion.float(), k_ref.float(), rtol=2e-2, atol=2e-2)


class TestRotaryEmbeddingGptjCorrectness:
    """Compare rotary_embedding_gptj against RotaryEmbedding.forward_static."""

    @pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
    @pytest.mark.parametrize("rotary_dim", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gptj_vs_baseline(self, num_tokens, rotary_dim, dtype):
        _skip_if_platform_unsupported("_helion_rotary_embedding_gptj")

        head_size = rotary_dim
        num_q_heads, num_kv_heads = 32, 8

        positions = torch.randint(0, 512, (num_tokens,), device="cuda")
        query = torch.randn(num_tokens, num_q_heads * head_size, device="cuda", dtype=dtype)
        key = torch.randn(num_tokens, num_kv_heads * head_size, device="cuda", dtype=dtype)
        cos_sin_cache = torch.randn(1024, rotary_dim, device="cuda", dtype=dtype)

        q_ref, k_ref = rotary_embedding_baseline(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache, is_neox_style=False,
        )
        q_helion, k_helion = rotary_embedding_gptj(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache,
        )

        assert q_helion.shape == q_ref.shape
        assert k_helion.shape == k_ref.shape

        rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (2e-3, 2e-3)
        torch.testing.assert_close(q_helion.float(), q_ref.float(), rtol=rtol, atol=atol)
        torch.testing.assert_close(k_helion.float(), k_ref.float(), rtol=rtol, atol=atol)

    def test_gptj_vs_pytorch(self):
        _skip_if_platform_unsupported("_helion_rotary_embedding_gptj")

        num_tokens, rotary_dim, head_size = 32, 128, 128
        num_q_heads, num_kv_heads = 16, 8
        dtype = torch.bfloat16

        positions = torch.randint(0, 512, (num_tokens,), device="cuda")
        query = torch.randn(num_tokens, num_q_heads * head_size, device="cuda", dtype=dtype)
        key = torch.randn(num_tokens, num_kv_heads * head_size, device="cuda", dtype=dtype)
        cos_sin_cache = torch.randn(1024, rotary_dim, device="cuda", dtype=dtype)

        q_ref, k_ref = _rope_pytorch(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache, is_neox_style=False,
        )
        q_helion, k_helion = rotary_embedding_gptj(
            positions, query.clone(), key.clone(),
            head_size, rotary_dim, cos_sin_cache,
        )
        torch.testing.assert_close(q_helion.float(), q_ref.float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(k_helion.float(), k_ref.float(), rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Integration test: kernel registration
# ---------------------------------------------------------------------------

# Internal names used by @register_kernel (the decorated function names).
_NEOX_KERNEL_NAME = "_helion_rotary_embedding_neox"
_GPTJ_KERNEL_NAME = "_helion_rotary_embedding_gptj"


class TestRotaryEmbeddingRegistration:
    def test_both_kernels_registered(self):
        if not _HELION_AVAILABLE:
            pytest.skip("Helion not installed")
        from vllm.kernels.helion.register import get_registered_kernels

        registered = get_registered_kernels()
        assert _NEOX_KERNEL_NAME in registered, (
            f"{_NEOX_KERNEL_NAME} was not auto-registered"
        )
        assert _GPTJ_KERNEL_NAME in registered, (
            f"{_GPTJ_KERNEL_NAME} was not auto-registered"
        )

    def test_kernel_wrappers_have_config_picker(self):
        if not _HELION_AVAILABLE:
            pytest.skip("Helion not installed")
        from vllm.kernels.helion.register import get_registered_kernels

        registered = get_registered_kernels()
        for name in [_NEOX_KERNEL_NAME, _GPTJ_KERNEL_NAME]:
            wrapper = registered[name]
            assert wrapper._config_picker is not None, (
                f"Kernel '{name}' has no config picker"
            )

    def test_kernel_wrappers_have_input_generator(self):
        if not _HELION_AVAILABLE:
            pytest.skip("Helion not installed")
        from vllm.kernels.helion.register import get_registered_kernels

        registered = get_registered_kernels()
        for name in [_NEOX_KERNEL_NAME, _GPTJ_KERNEL_NAME]:
            wrapper = registered[name]
            assert wrapper._input_generator is not None, (
                f"Kernel '{name}' has no input generator"
            )
            # Smoke-test: calling get_inputs() on CPU should not crash.
            inputs = wrapper.get_inputs()
            assert len(inputs) > 0

    def test_output_is_not_inplace(self):
        """The Helion kernel must return new tensors (out-of-place)."""
        _skip_if_platform_unsupported("rotary_embedding_neox")

        num_tokens, rotary_dim, head_size = 8, 128, 128
        positions = torch.randint(0, 512, (num_tokens,), device="cuda")
        query = torch.randn(num_tokens, 32 * head_size, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(num_tokens, 8 * head_size, device="cuda", dtype=torch.bfloat16)
        cos_sin_cache = torch.randn(1024, rotary_dim, device="cuda", dtype=torch.bfloat16)

        q_before = query.clone()
        k_before = key.clone()

        q_out, k_out = rotary_embedding_neox(
            positions, query, key, head_size, rotary_dim, cos_sin_cache
        )

        # Input tensors must be unchanged (out-of-place).
        torch.testing.assert_close(query, q_before)
        torch.testing.assert_close(key, k_before)

        # Output tensors must differ from the input.
        assert not torch.allclose(q_out, query), "q_out should differ from query"
        assert not torch.allclose(k_out, key), "k_out should differ from key"
