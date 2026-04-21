# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``system_fingerprint`` string construction."""

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai import fingerprint as fp


def _vllm_config(
    *,
    tp=1,
    pp=1,
    dp=1,
    ep=False,
    dtype="bfloat16",
    quant=None,
    kv="auto",
    digest="a3b21f94deadbeefcafe",
):
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
        ),
        model_config=SimpleNamespace(dtype=dtype, quantization=quant),
        cache_config=SimpleNamespace(cache_dtype=kv),
    )
    cfg.compute_hash = lambda: digest  # type: ignore[attr-defined]
    return cfg


@pytest.fixture(autouse=True)
def _reset_fingerprint_state():
    fp.set_default_fingerprint_mode("full")
    yield
    fp.set_default_fingerprint_mode("full")


@pytest.fixture
def patch_platform(monkeypatch):
    """Stub ``current_platform.get_device_name`` so tests are hardware-agnostic."""

    def _apply(name: str = "NVIDIA H100 80GB HBM3") -> None:
        import vllm.platforms as vp

        monkeypatch.setattr(
            type(vp.current_platform),
            "get_device_name",
            classmethod(lambda cls, device_id=0: name),
            raising=False,
        )

    return _apply


def test_full_mode_contains_expected_pieces(patch_platform):
    patch_platform("NVIDIA H100 80GB HBM3")
    s = fp.build_system_fingerprint(_vllm_config(tp=8), mode="full")
    assert s.startswith("vllm-")
    assert "H100" in s
    assert "tp8" in s
    assert "bf16" in s
    assert s.endswith("a3b21f94")


def test_hash_mode_is_opaque(patch_platform):
    patch_platform("NVIDIA H100")
    s = fp.build_system_fingerprint(_vllm_config(tp=8), mode="hash")
    assert s == "vllm-a3b21f94"


def test_includes_pp_dp_ep(patch_platform):
    patch_platform("NVIDIA A100-SXM4-80GB")
    s = fp.build_system_fingerprint(
        _vllm_config(tp=8, pp=2, dp=4, ep=True), mode="full"
    )
    assert "A100" in s
    assert "tp8" in s and "pp2" in s and "dp4" in s and "ep" in s


def test_omits_trivial_parallelism(patch_platform):
    patch_platform("NVIDIA H100")
    s = fp.build_system_fingerprint(_vllm_config(), mode="full")
    for tok in ("tp1", "pp1", "dp1", "ep"):
        assert f"-{tok}-" not in s, s


def test_includes_quant_and_kv_cache_dtype(patch_platform):
    patch_platform("NVIDIA H100")
    s = fp.build_system_fingerprint(
        _vllm_config(dtype="float16", quant="awq", kv="fp8_e4m3"),
        mode="full",
    )
    assert "fp16" in s
    assert "awq" in s
    assert "kvfp8_e4m3" in s


def test_device_name_normalization():
    cases = {
        "NVIDIA A100-SXM4-80GB": "A100",
        "NVIDIA H100 80GB HBM3": "H100",
        "AMD Instinct MI300X": "MI300X",
        "AMD MI250": "MI250",
        "Intel Gaudi2": "Gaudi2",
        "": "unknown",
    }
    for raw, expected in cases.items():
        assert fp._short_device_name(raw) == expected, raw


def test_cache_reuses_computation(monkeypatch, patch_platform):
    patch_platform("NVIDIA H100")
    cfg = _vllm_config(tp=8)

    calls = {"n": 0}
    real_build = fp.build_system_fingerprint

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(fp, "build_system_fingerprint", counting)
    a = fp.get_system_fingerprint(cfg)
    b = fp.get_system_fingerprint(cfg)
    assert a == b
    assert calls["n"] == 1


def test_set_default_mode_switches_and_clears_cache(patch_platform):
    patch_platform("NVIDIA H100")
    cfg = _vllm_config(tp=8)
    full = fp.get_system_fingerprint(cfg)
    fp.set_default_fingerprint_mode("hash")
    hashed = fp.get_system_fingerprint(cfg)
    assert full != hashed
    assert hashed.startswith("vllm-")
    assert "H100" not in hashed


def test_fallback_when_compute_hash_fails(patch_platform):
    patch_platform("NVIDIA H100")
    cfg = _vllm_config()

    def boom():
        raise RuntimeError("nope")

    cfg.compute_hash = boom
    s = fp.build_system_fingerprint(cfg, mode="full")
    assert s.endswith("-nohash")
