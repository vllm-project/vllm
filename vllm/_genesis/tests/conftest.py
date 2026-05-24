# SPDX-License-Identifier: Apache-2.0
"""pytest configuration for Genesis tests.

Fixtures and helpers shared across all test modules.

Audit A-15 (2026-05-05): torch is now optionally imported. If torch is
not available in the environment, all tests using torch are skipped
automatically; pure wiring/audit tests still run. Use
`@pytest.mark.requires_torch` on tests that need torch primitives.
"""
from __future__ import annotations

import pytest

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#                          PLATFORM DETECTION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """True if CUDA is available for testing."""
    return _TORCH_AVAILABLE and torch.cuda.is_available()


@pytest.fixture(scope="session")
def rocm_available() -> bool:
    """True if running on ROCm (PyTorch built for HIP)."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
    try:
        return torch.version.hip is not None
    except AttributeError:
        return False


@pytest.fixture(scope="session")
def nvidia_cuda_available() -> bool:
    """True if NVIDIA CUDA specifically (NOT ROCm)."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
    try:
        # ROCm's torch.version.hip is a string; NVIDIA's is None
        return torch.version.hip is None
    except AttributeError:
        # Old PyTorch without torch.version.hip = NVIDIA-only build
        return torch.cuda.is_available()


# ═══════════════════════════════════════════════════════════════════════════
#                       PYTEST MARKERS
# ═══════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "cuda_required: test requires NVIDIA CUDA device",
    )
    config.addinivalue_line(
        "markers",
        "rocm_required: test requires AMD ROCm device",
    )
    config.addinivalue_line(
        "markers",
        "gpu_required: test requires any GPU (CUDA or ROCm)",
    )
    config.addinivalue_line(
        "markers",
        "slow: test takes >5 seconds",
    )
    config.addinivalue_line(
        "markers",
        "requires_torch: test imports torch — auto-skipped without torch (audit A-15)",
    )


def _file_imports_torch(file_path: str) -> bool:
    """Audit A-15 fix 2026-05-05 — auto-detect module-level `import torch`.

    Without this, tests that have `import torch` at module top fail
    pytest collection on CPU-only hosts (Mac dev rig) BEFORE any
    `requires_torch` marker can take effect. By scanning file source
    via AST at collection time we can mark ALL tests in such files
    as `requires_torch` automatically.
    """
    try:
        import ast as _ast
        with open(file_path, encoding="utf-8") as f:
            src = f.read()
        tree = _ast.parse(src)
    except (OSError, SyntaxError):
        return False
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for n in node.names:
                if n.name == "torch" or n.name.startswith("torch."):
                    return True
        elif isinstance(node, _ast.ImportFrom):
            if node.module and (node.module == "torch" or node.module.startswith("torch.")):
                return True
    return False


# Cache the scan result per file so we don't re-parse on every test item
_TORCH_FILE_CACHE: dict[str, bool] = {}


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests automatically on CPU-only hosts. Skip torch-required
    tests when torch is not importable (audit A-15)."""
    cuda = _TORCH_AVAILABLE and torch.cuda.is_available()
    for item in items:
        # Audit A-15 fix: auto-skip ALL tests in files that import torch
        # at module level when torch is not available — without this,
        # the test file fails collection (ImportError) before any explicit
        # `requires_torch` marker can take effect on CPU-only Mac dev rigs.
        if not _TORCH_AVAILABLE:
            file_path = str(item.fspath) if hasattr(item, "fspath") else item.location[0]
            if file_path not in _TORCH_FILE_CACHE:
                _TORCH_FILE_CACHE[file_path] = _file_imports_torch(file_path)
            if _TORCH_FILE_CACHE[file_path] and "requires_torch" not in item.keywords:
                item.add_marker(pytest.mark.skip(
                    reason="torch not available (auto-detected module-level import)"
                ))
                continue
            if "requires_torch" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="torch not available"))
        if "cuda_required" in item.keywords and not cuda:
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))
        if "gpu_required" in item.keywords and not cuda:
            item.add_marker(pytest.mark.skip(reason="GPU not available"))


# ═══════════════════════════════════════════════════════════════════════════
#                         FIXTURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _reset_genesis_prealloc_state() -> None:
    """Internal helper: drop all module-cached state used by Genesis preallocs."""
    from vllm._genesis.prealloc import GenesisPreallocBuffer
    GenesisPreallocBuffer.clear_for_tests()
    try:
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        TurboQuantBufferManager.clear_for_tests()
    except Exception:
        # Module not importable when torch missing — fixture is best-effort
        pass
    try:
        from vllm._genesis.kernels.gdn_core_attn_manager import GdnCoreAttnManager
        GdnCoreAttnManager.clear_for_tests()
    except Exception:
        # Module not importable when torch missing — fixture is best-effort
        pass
    # The central token-budget resolver caches its decision at module
    # scope. Tests that probe the default-fallback path need a fresh
    # cache, otherwise they see whatever an earlier test resolved.
    try:
        from vllm._genesis import prealloc_budget as _pb
        _pb._CACHED = None
    except Exception:
        # Module not importable in CPU-only minimal envs — fixture is best-effort
        pass


@pytest.fixture
def reset_genesis_prealloc():
    """Clear ALL Genesis buffer registries before/after each test.

    Covers:
      - `GenesisPreallocBuffer._REGISTRY` (universal framework)
      - `TurboQuantBufferManager._K_BUFFERS / _V_BUFFERS / _CU_* /
         _SYNTH_* / _PREFILL_OUT_BUFFERS / _DECODE_*` (P22/P26/P32/P33/P36)
      - `GdnCoreAttnManager._BUFFERS` + `_SHOULD_APPLY_CACHED` (P28)
      - `prealloc_budget._CACHED` (P73 token budget resolver)

    Test isolation is critical since these are class-level state on
    module-scoped singletons. If one test allocates and another asserts
    the registry is empty, a stale entry leaks and the assertion fails.

    Usage:
        def test_something(reset_genesis_prealloc):
            # all Genesis registries are clean
            ...
            # and cleaned again after test
    """
    _reset_genesis_prealloc_state()
    yield
    _reset_genesis_prealloc_state()


@pytest.fixture(autouse=True)
def _autoreset_token_budget_cache():
    """Always-on hygiene: drop the central P73 _CACHED before AND after
    every test in this directory. The fixture is cheap (one attribute
    write) and prevents cross-test pollution from any test that touches
    `prealloc_budget.resolve_token_budget()` directly or indirectly."""
    try:
        from vllm._genesis import prealloc_budget as _pb
        _pb._CACHED = None
    except Exception:
        # Module not importable in CPU-only minimal envs — autouse fixture is best-effort
        pass
    yield
    try:
        from vllm._genesis import prealloc_budget as _pb
        _pb._CACHED = None
    except Exception:
        # Module not importable in CPU-only minimal envs — autouse fixture is best-effort
        pass


@pytest.fixture
def deterministic_seed():
    """Set deterministic torch seed for reproducible tests."""
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield 42
