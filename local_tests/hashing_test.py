import sys
import types
import importlib.util
from pathlib import Path

import pytest


def _ensure_pkg(name: str, path: Path):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]  # mark as package
        sys.modules[name] = pkg


def _load(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Build a minimal package view to avoid importing vllm.__init__ (which needs torch)
ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "vllm"
_ensure_pkg("vllm", PKG)
_ensure_pkg("vllm.config", PKG / "config")
_ensure_pkg("vllm.compilation", PKG / "compilation")

# Provide a tiny stub for vllm.compilation.inductor_pass used by CompilationConfig
stub_inductor = types.ModuleType("vllm.compilation.inductor_pass")
class _StubPass:  # minimal shape
    def __init__(self, *args, **kwargs):
        pass
CallableInductorPass = _StubPass
InductorPass = _StubPass
stub_inductor.CallableInductorPass = CallableInductorPass
stub_inductor.InductorPass = InductorPass
sys.modules["vllm.compilation.inductor_pass"] = stub_inductor

# Load utils first so downstream imports succeed
utils = _load("vllm.config.utils", PKG / "config" / "utils.py")
envs = _load("vllm.envs", PKG / "envs.py")
cache_mod = _load("vllm.config.cache", PKG / "config" / "cache.py")
compilation_mod = _load("vllm.config.compilation", PKG / "config" / "compilation.py")

CacheConfig = cache_mod.CacheConfig
CompilationConfig = compilation_mod.CompilationConfig
CUDAGraphMode = compilation_mod.CUDAGraphMode


def test_env_hash_excludes_logging_level(monkeypatch):
    h0 = envs.compute_hash()

    # Excluded knob should not change the hash
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    h1 = envs.compute_hash()
    assert h1 == h0

    # Included knobs should change the hash
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    h2 = envs.compute_hash()
    assert h2 != h1

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "XFORMERS")
    h3 = envs.compute_hash()
    assert h3 != h2


def test_cache_config_opt_out_hashing():
    c0 = CacheConfig()
    h0 = c0.compute_hash()

    # Excluded knob should not change the hash
    c1 = CacheConfig(cpu_offload_gb=10)
    h1 = c1.compute_hash()
    assert h1 == h0

    # Included knob should change the hash
    c2 = CacheConfig(mamba_cache_dtype="float32")
    h2 = c2.compute_hash()
    assert h2 != h0



def test_compilation_config_opt_out_hashing():
    cc0 = CompilationConfig()
    h0 = cc0.compute_hash()

    # Excluded detail should not change the hash
    cc1 = CompilationConfig(debug_dump_path="/tmp/vllm-dumps")
    h1 = cc1.compute_hash()
    assert h1 == h0

    # Included details should change the hash
    cc2 = CompilationConfig(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    h2 = cc2.compute_hash()
    assert h2 != h0

    cc3 = CompilationConfig(level=3)
    h3 = cc3.compute_hash()
    assert h3 != h0


