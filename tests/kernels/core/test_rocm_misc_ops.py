# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm platform and extension smoke tests.

These tests catch ROCm-specific regressions early:

1. **Extension import smoke tests**: Verify that ROCm native extensions load
   without crashing. Build/linking issues surface here before runtime.

2. **GCN arch parsing tests**: Verify _capability_from_gcn_arch correctly
   parses all known GPU architectures. Wrong parsing causes silent feature
   gate failures or crashes from selecting unsupported kernels.

3. **AITER ops availability tests**: Verify rocm_aiter_ops accessor methods
   work without crashing, regardless of hardware support.

Related coverage:
- AITER kernel numerics: ``tests/kernels/core/test_rocm_aiter_ops.py``
- Attention backend selection: ``test_rocm_attention_selector.py``
- GEMM dispatch: ``test_rocm_unquantized_gemm.py``
- Custom paged attention: tests below for predicate logic
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


# ---------------------------------------------------------------------------
# Extension import smoke tests
# ---------------------------------------------------------------------------


def test_rocm_extension_imports():
    """ROCm native extension loads without error.

    Catches build/linking issues before runtime. If this fails, the ROCm
    build is broken and nothing else will work.
    """
    import vllm._rocm_C  # noqa: F401


def test_aiter_ops_module_imports():
    """AITER ops module imports without error.

    The _aiter_ops module registers custom ops and initializes AITER state.
    Import failures here indicate missing dependencies or incompatible
    AITER versions.
    """
    import vllm._aiter_ops  # noqa: F401


def test_rocm_platform_module_imports():
    """ROCm platform module imports without error.

    The platform module queries hardware via amdsmi and sets up arch flags.
    Import failures indicate missing amdsmi or HIP runtime issues.
    """
    import vllm.platforms.rocm  # noqa: F401


# ---------------------------------------------------------------------------
# GCN arch parsing tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gcn_arch", "expected_major", "expected_minor"),
    [
        pytest.param("gfx90a", 9, 0, id="MI200-gfx90a"),
        pytest.param("gfx942", 9, 4, id="MI300X-gfx942"),
        pytest.param("gfx950", 9, 5, id="MI350-gfx950"),
        pytest.param("gfx1100", 11, 0, id="RX7900-gfx1100"),
        pytest.param("gfx1101", 11, 0, id="RX7800-gfx1101"),
        pytest.param("gfx1151", 11, 5, id="StrixHalo-gfx1151"),
        pytest.param("gfx1200", 12, 0, id="RDNA4-gfx1200"),
        pytest.param("gfx1201", 12, 0, id="RX9070-gfx1201"),
        pytest.param("gfx1250", 12, 5, id="CDNA5-gfx1250"),
    ],
)
def test_capability_from_gcn_arch_known_gpus(gcn_arch, expected_major, expected_minor):
    """_capability_from_gcn_arch parses known GPU architectures correctly.

    This is critical for feature gating - wrong parsing causes:
    - FP8 kernels selected on hardware that doesn't support them
    - MX format support incorrectly enabled/disabled
    - Custom allreduce used on unsupported hardware
    """
    from vllm.platforms.rocm import _capability_from_gcn_arch

    result = _capability_from_gcn_arch(gcn_arch)
    assert result is not None, f"Failed to parse {gcn_arch}"
    assert result == (expected_major, expected_minor), (
        f"Wrong capability for {gcn_arch}: got {result}, "
        f"expected ({expected_major}, {expected_minor})"
    )


@pytest.mark.parametrize(
    "gcn_arch",
    [
        pytest.param("not_a_gfx_string", id="non-gfx-string"),
        pytest.param("cuda_arch", id="cuda-style"),
        pytest.param("", id="empty-string"),
    ],
)
def test_capability_from_gcn_arch_returns_none_for_non_gfx(gcn_arch):
    """_capability_from_gcn_arch returns None for non-GFX strings.

    Non-GFX strings should return None so the caller can fall back to
    torch.cuda.get_device_capability.
    """
    from vllm.platforms.rocm import _capability_from_gcn_arch

    result = _capability_from_gcn_arch(gcn_arch)
    assert result is None


@pytest.mark.parametrize(
    "gcn_arch",
    [
        pytest.param("gfx8", id="too-few-digits"),
        pytest.param("gfx12345", id="too-many-digits"),
        pytest.param("gfx700", id="unsupported-major-7"),
        pytest.param("gfx1500", id="future-major-15"),
    ],
)
def test_capability_from_gcn_arch_raises_for_malformed(gcn_arch):
    """_capability_from_gcn_arch raises ValueError for malformed GFX strings.

    Malformed strings that look like GFX but can't be parsed should raise
    so the user gets a clear error message asking them to file an issue.
    """
    from vllm.platforms.rocm import _capability_from_gcn_arch

    with pytest.raises(ValueError, match="GCN arch"):
        _capability_from_gcn_arch(gcn_arch)


# ---------------------------------------------------------------------------
# AITER ops availability tests
# ---------------------------------------------------------------------------


def test_rocm_aiter_ops_accessor_methods_dont_crash():
    """rocm_aiter_ops accessor methods work without crashing.

    These methods gate kernel selection. Even if AITER isn't available,
    they should return False gracefully rather than crashing.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    # These should all return bool without crashing
    assert isinstance(rocm_aiter_ops.is_enabled(), (bool, type(None)))
    assert isinstance(rocm_aiter_ops.is_linear_enabled(), (bool, type(None)))
    assert isinstance(rocm_aiter_ops.is_fused_moe_enabled(), (bool, type(None)))
    assert isinstance(rocm_aiter_ops.is_mla_enabled(), (bool, type(None)))
    assert isinstance(rocm_aiter_ops.is_mha_enabled(), (bool, type(None)))


def test_rocm_aiter_ops_is_aiter_found():
    """is_aiter_found returns consistent results."""
    from vllm._aiter_ops import IS_AITER_FOUND, is_aiter_found

    # The cached global should match the function result
    assert is_aiter_found() == IS_AITER_FOUND


def test_rocm_aiter_ops_refresh_env_variables_doesnt_crash():
    """refresh_env_variables works without crashing.

    This is called after monkeypatching env vars in tests. It should
    never crash, even with unusual env var combinations.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    # Should not raise
    rocm_aiter_ops.refresh_env_variables()


# ---------------------------------------------------------------------------
# Env var propagation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
def test_shuffle_kv_cache_env_propagates_to_rocm_aiter_ops(enabled, monkeypatch):
    """VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT propagates to AITER state."""
    import vllm.envs as envs
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1" if enabled else "0")
    importlib.reload(envs)
    rocm_aiter_ops.refresh_env_variables()

    assert envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT is enabled
    assert rocm_aiter_ops.is_shuffle_kv_cache_enabled() is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_skinny_gemm_env_controls_rocm_fp8_scaled_mm_support(enabled, monkeypatch):
    """VLLM_ROCM_USE_SKINNY_GEMM gates ROCm FP8 scaled-mm kernel eligibility."""
    import vllm.envs as envs
    from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
        ROCmFP8ScaledMMLinearKernel,
    )

    monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "1" if enabled else "0")
    importlib.reload(envs)
    monkeypatch.setattr("vllm.platforms.rocm.on_mi3xx", lambda: True)

    supported, reason = ROCmFP8ScaledMMLinearKernel.is_supported()

    assert supported is enabled
    if enabled:
        assert reason is None
    else:
        assert reason == "requires VLLM_ROCM_USE_SKINNY_GEMM to be enabled."


# ---------------------------------------------------------------------------
# Custom paged-attention eligibility tests
# ---------------------------------------------------------------------------


def _set_rocm_arch(monkeypatch: pytest.MonkeyPatch, gcn_arch: str):
    """Patch ROCm arch detection for testing.

    Patches all on_*() accessor functions based on the provided gcn_arch string.
    """
    import vllm.platforms.rocm as rocm_platform

    monkeypatch.setattr(rocm_platform, "_GCN_ARCH", gcn_arch)

    _on_gfx11 = "gfx11" in gcn_arch
    _on_gfx1100 = "gfx1100" in gcn_arch
    _on_gfx1151 = "gfx1151" in gcn_arch
    _on_gfx12x_raw = "gfx12" in gcn_arch
    _on_gfx1250 = "gfx1250" in gcn_arch
    _on_gfx1x_raw = any(arch in gcn_arch for arch in ["gfx11", "gfx12"])

    _on_gfx90a = "gfx90a" in gcn_arch
    _on_gfx942 = "gfx942" in gcn_arch
    _on_gfx950 = "gfx950" in gcn_arch
    _on_gfx9 = any(gfx in gcn_arch for gfx in ["gfx90a", "gfx942", "gfx950"])
    _on_mi3xx = any(gfx in gcn_arch for gfx in ["gfx942", "gfx950"])

    _on_cdna = any(arch in gcn_arch for arch in ["gfx9", "gfx1250"])
    _on_rdna = _on_gfx1x_raw and not _on_cdna
    _on_rdna4 = any(arch in gcn_arch for arch in ["gfx1200", "gfx1201"])

    _on_gfx1x = _on_gfx1x_raw and not _on_cdna
    _on_gfx12x = _on_gfx12x_raw and not _on_cdna

    monkeypatch.setattr(rocm_platform, "on_gfx9", lambda: _on_gfx9)
    monkeypatch.setattr(rocm_platform, "on_gfx90a", lambda: _on_gfx90a)
    monkeypatch.setattr(rocm_platform, "on_gfx942", lambda: _on_gfx942)
    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: _on_gfx950)
    monkeypatch.setattr(rocm_platform, "on_gfx11", lambda: _on_gfx11)
    monkeypatch.setattr(rocm_platform, "on_gfx1100", lambda: _on_gfx1100)
    monkeypatch.setattr(rocm_platform, "on_gfx1151", lambda: _on_gfx1151)
    monkeypatch.setattr(rocm_platform, "on_gfx1250", lambda: _on_gfx1250)
    monkeypatch.setattr(rocm_platform, "on_gfx1x", lambda: _on_gfx1x)
    monkeypatch.setattr(rocm_platform, "on_gfx12x", lambda: _on_gfx12x)
    monkeypatch.setattr(rocm_platform, "on_mi3xx", lambda: _on_mi3xx)
    monkeypatch.setattr(rocm_platform, "on_cdna", lambda: _on_cdna)
    monkeypatch.setattr(rocm_platform, "on_rdna", lambda: _on_rdna)
    monkeypatch.setattr(rocm_platform, "on_rdna4", lambda: _on_rdna4)
    monkeypatch.setattr(rocm_platform, "_ON_GFX1X", _on_gfx1x_raw)

    rocm_platform.use_rocm_custom_paged_attention.cache_clear()

    return rocm_platform


def test_rocm_custom_paged_attention_gfx9_supported_case(monkeypatch):
    """gfx9 custom paged-attention accepts its documented fast-path."""
    rocm_platform = _set_rocm_arch(monkeypatch, "gfx942")

    assert rocm_platform.use_rocm_custom_paged_attention(
        qtype=torch.bfloat16,
        head_size=128,
        block_size=16,
        gqa_ratio=4,
        max_seq_len=4096,
        sliding_window=0,
        kv_cache_dtype="auto",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"head_size": 256}, id="reject-head-size"),
        pytest.param({"block_size": 8}, id="reject-block-size"),
        pytest.param({"gqa_ratio": 17}, id="reject-gqa"),
        pytest.param({"sliding_window": 128}, id="reject-sliding-window"),
        pytest.param({"max_seq_len": 128 * 1024 + 16}, id="reject-seq-len"),
        pytest.param({"sinks": torch.ones(1)}, id="reject-sinks"),
    ],
)
def test_rocm_custom_paged_attention_gfx9_rejects_unsupported_cases(
    kwargs, monkeypatch
):
    """gfx9 custom paged-attention rejects unsupported shapes/features."""
    rocm_platform = _set_rocm_arch(monkeypatch, "gfx950")

    params = dict(
        qtype=torch.float16,
        head_size=64,
        block_size=16,
        gqa_ratio=4,
        max_seq_len=4096,
        sliding_window=0,
        kv_cache_dtype="auto",
        alibi_slopes=None,
        sinks=None,
    )
    params.update(kwargs)

    assert not rocm_platform.use_rocm_custom_paged_attention(**params)


def test_rocm_custom_paged_attention_gfx1x_supported_case(monkeypatch):
    """gfx1x custom paged-attention accepts the narrower RDNA fast-path."""
    rocm_platform = _set_rocm_arch(monkeypatch, "gfx1201")

    assert rocm_platform.use_rocm_custom_paged_attention(
        qtype=torch.float16,
        head_size=128,
        block_size=16,
        gqa_ratio=4,
        max_seq_len=4096,
        sliding_window=0,
        kv_cache_dtype="auto",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"head_size": 64}, id="reject-head-size"),
        pytest.param({"gqa_ratio": 2}, id="reject-gqa"),
        pytest.param({"kv_cache_dtype": "fp8"}, id="reject-kv-cache-dtype"),
        pytest.param({"alibi_slopes": torch.ones(8)}, id="reject-alibi"),
        pytest.param({"sinks": torch.ones(1)}, id="reject-sinks"),
    ],
)
def test_rocm_custom_paged_attention_gfx1x_rejects_unsupported_cases(
    kwargs, monkeypatch
):
    """gfx1x custom paged-attention rejects unsupported RDNA configurations."""
    rocm_platform = _set_rocm_arch(monkeypatch, "gfx1100")

    params = dict(
        qtype=torch.bfloat16,
        head_size=128,
        block_size=16,
        gqa_ratio=4,
        max_seq_len=4096,
        sliding_window=0,
        kv_cache_dtype="auto",
        alibi_slopes=None,
        sinks=None,
    )
    params.update(kwargs)

    assert not rocm_platform.use_rocm_custom_paged_attention(**params)
