# SPDX-License-Identifier: Apache-2.0
"""Genesis defensive guards — canonical vendor/chip/model/dependency detection.

Philosophy: МЫ ЧИНИМ, НЕ ЛОМАЕМ.
Every helper is fail-safe: returns a safe default (False/None) on any exception.
If detection cannot complete, we SKIP the patch — never crash the engine.

All detection patterns mirror upstream vLLM canonical sources:
  - vllm/platforms/interface.py   (Platform predicates, DeviceCapability)
  - vllm/platforms/cuda.py        (NvmlCudaPlatform.get_device_capability)
  - vllm/platforms/rocm.py        (_GCN_ARCH parsing)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Optional

log = logging.getLogger("genesis.guards")


# ─── torch.dynamo compatibility ────────────────────────────────────────────
# Genesis guards are eager-only diagnostic helpers (vendor / SM / version
# detection). They are commonly called from kernel paths that vLLM compiles
# with torch.compile / torch.dynamo (Marlin apply_weights, FP8 scaled MM,
# CUDA-graph capture, etc.).
#
# Two issues seen empirically (2026-04-28, with GENESIS_FORCE_MARLIN_W8A16=1
# on Qwen3.6-27B-INT8-AutoRound, Marlin path):
#   1. torch.dynamo IGNORES `@functools.lru_cache` / `@functools.cache`
#      wrappers and traces the underlying function instead → crash inside
#      `current_platform.get_device_capability()` which dynamo can't trace.
#   2. `@torch._dynamo.disable` raises 'Skip calling
#      torch.compiler.disable()d function' when invoked from inside an
#      already-traced compiled region (it cannot fall back to eager mid-trace).
#
# Robust fix: snapshot the platform-derived facts ONCE at module-load time
# (which happens eagerly during plugin register, BEFORE any torch.compile /
# dynamo run), and have all public guards return those captured constants.
# The functions become pure `return _MODULE_CONSTANT` — dynamo can trace
# through them with zero risk of touching unsupported APIs.
#
# A `_refresh()` helper is exposed for tests that monkey-patch
# `current_platform`. Production has no use case for it.
def _detect_platform() -> Optional[Any]:
    try:
        from vllm.platforms import current_platform
        return current_platform
    except Exception as e:
        log.debug("[guards] vllm.platforms.current_platform unavailable: %s", e)
        return None


def _detect_is_cuda(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_cuda())
    except Exception:
        return False


def _detect_is_rocm(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_rocm())
    except Exception:
        return False


def _detect_is_xpu(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_xpu())
    except Exception:
        return False


def _detect_is_cpu(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_cpu())
    except Exception:
        return False


def _detect_is_cuda_alike(p: Optional[Any]) -> bool:
    try:
        return bool(p is not None and p.is_cuda_alike())
    except Exception:
        return False


def _detect_compute_capability(
    p: Optional[Any], is_cuda: bool
) -> Optional[tuple[int, int]]:
    if not is_cuda or p is None:
        return None
    try:
        cc = p.get_device_capability()
        if cc is None:
            return None
        return (cc.major, cc.minor)
    except Exception as e:
        log.debug("[guards] get_device_capability failed: %s", e)
        return None


# Module-level captured constants — populated once at import time (eager
# context). Public functions just return these. dynamo / torch.compile can
# trace through `return _MODULE_CONST` trivially without ever touching
# platform internals.
_PLATFORM: Optional[Any] = None
_IS_CUDA: bool = False
_IS_ROCM: bool = False
_IS_XPU: bool = False
_IS_CPU: bool = False
_IS_CUDA_ALIKE: bool = False
_COMPUTE_CAPABILITY: Optional[tuple[int, int]] = None


def _refresh() -> None:
    """Re-snapshot platform facts. Called once at module import; tests can
    re-invoke after monkey-patching `vllm.platforms.current_platform`."""
    global _PLATFORM, _IS_CUDA, _IS_ROCM, _IS_XPU, _IS_CPU, _IS_CUDA_ALIKE
    global _COMPUTE_CAPABILITY
    _PLATFORM = _detect_platform()
    _IS_CUDA = _detect_is_cuda(_PLATFORM)
    _IS_ROCM = _detect_is_rocm(_PLATFORM)
    _IS_XPU = _detect_is_xpu(_PLATFORM)
    _IS_CPU = _detect_is_cpu(_PLATFORM)
    _IS_CUDA_ALIKE = _detect_is_cuda_alike(_PLATFORM)
    _COMPUTE_CAPABILITY = _detect_compute_capability(_PLATFORM, _IS_CUDA)


_refresh()


# ═══════════════════════════════════════════════════════════════════════════
#                          VENDOR / PLATFORM IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

def _current_platform() -> Optional[Any]:
    """Return the snapshotted platform handle (set at module load).

    See module docstring for why this is a constant return now.
    """
    return _PLATFORM


def is_nvidia_cuda() -> bool:
    """True ONLY on NVIDIA CUDA (NOT ROCm).

    Returns the constant snapshot taken at module load. Trace-safe (pure
    `return _IS_CUDA`); see module docstring.
    """
    return _IS_CUDA


def is_amd_rocm() -> bool:
    """True on AMD ROCm. Trace-safe constant return."""
    return _IS_ROCM


def is_intel_xpu() -> bool:
    """True on Intel XPU. Trace-safe constant return."""
    return _IS_XPU


def is_cpu_only() -> bool:
    """True on CPU-only build. Trace-safe constant return."""
    return _IS_CPU


def is_cuda_alike() -> bool:
    """CUDA OR ROCm. Trace-safe constant return.

    ⚠️ TRAP: Do NOT use for NVIDIA-specific patches. Use is_nvidia_cuda().
    """
    return _IS_CUDA_ALIKE


# ═══════════════════════════════════════════════════════════════════════════
#                      NVIDIA COMPUTE CAPABILITY
# ═══════════════════════════════════════════════════════════════════════════

def get_compute_capability() -> Optional[tuple[int, int]]:
    """Return snapshotted (major, minor) compute capability for NVIDIA CUDA.

    Trace-safe constant return. Returns None on non-CUDA / failed detection.
    """
    return _COMPUTE_CAPABILITY


def is_sm_at_least(major: int, minor: int = 0) -> bool:
    """True if SM >= (major, minor). Trace-safe (reads module constant)."""
    cc = _COMPUTE_CAPABILITY
    if cc is None:
        return False
    return cc >= (major, minor)


def is_sm_exactly(major: int, minor: int) -> bool:
    """True if SM is exactly (major, minor). Trace-safe constant read."""
    return _COMPUTE_CAPABILITY == (major, minor)


def is_ampere_datacenter() -> bool:
    """NVIDIA A100 — SM 8.0."""
    return is_sm_exactly(8, 0)


def is_ampere_consumer() -> bool:
    """NVIDIA A5000 / A6000 / RTX 3090 — SM 8.6.

    Genesis prod baseline.
    """
    return is_sm_exactly(8, 6)


def is_ampere_any() -> bool:
    """Any Ampere (SM 8.x except Ada 8.9)."""
    cc = get_compute_capability()
    return cc is not None and cc[0] == 8 and cc[1] < 9


def is_ada_lovelace() -> bool:
    """NVIDIA RTX 4090 / L40 / RTX 6000 Ada — SM 8.9."""
    return is_sm_exactly(8, 9)


def is_hopper() -> bool:
    """NVIDIA H100 / H200 — SM 9.0."""
    return is_sm_exactly(9, 0)


def is_blackwell() -> bool:
    """NVIDIA Blackwell family — SM 10.x (datacenter/pro) OR SM 12.x (consumer).

    NVIDIA splits Blackwell across two SM major versions:
      * sm_10x (10.0/10.1/10.3): B100, B200, GB200, RTX PRO 6000 Blackwell
      * sm_120 (12.0): RTX 5090, 5080, 5070, 5060 (consumer)

    Fix for issue #20 (2026-05-04, club-3090 RTX 5090 user): original
    `cc[0] == 10` missed consumer Blackwell. Now `cc[0] in (10, 12)`.
    """
    cc = get_compute_capability()
    return cc is not None and cc[0] in (10, 12)


def is_blackwell_datacenter() -> bool:
    """NVIDIA Blackwell datacenter/pro (B100/B200/GB200/RTX PRO 6000) — SM 10.x only."""
    cc = get_compute_capability()
    return cc is not None and cc[0] == 10


def is_blackwell_consumer() -> bool:
    """NVIDIA Blackwell consumer (RTX 5090/5080/5070/5060) — SM 12.x only."""
    cc = get_compute_capability()
    return cc is not None and cc[0] == 12


def has_native_fp8() -> bool:
    """True if GPU has native FP8 tensor cores (SM >= 8.9).

    Includes Ada Lovelace (8.9), Hopper (9.0), Blackwell (10.0).
    Ampere (8.6) does NOT have native FP8 — uses emulation.
    """
    return is_sm_at_least(8, 9)


def pdl_support_expected() -> bool:
    """True if platform is expected to support Programmatic Dependent Launch.

    PDL (Programmatic Dependent Launch) is a Hopper+ / Blackwell feature. On
    older GPUs (Ampere / Ada), enabling PDL-related env vars has no effect
    at best; at worst it can trigger vLLM issue #40742 (Inductor autotune
    calls torch.cuda.synchronize() inside CUDA graph capture → illegal
    cuda operation → server crash at startup).

    Returns:
      True on SM >= 9.0 (Hopper, Blackwell, future).
      False on SM < 9.0 (Ampere consumer/datacenter, Ada Lovelace, pre-Ampere).
      False on non-NVIDIA.
    """
    return is_sm_at_least(9, 0)


def detect_pdl_env_misconfig() -> list[str]:
    """Detect user-set PDL env vars that aren't safe on this GPU.

    Reference: vLLM issue #40742 (2026-04-23) — CUDA graph capture crash when
    `TRTLLM_ENABLE_PDL=1` / `TORCHINDUCTOR_ENABLE_PDL=1` is set on GPUs where
    PDL is not fully supported, because Inductor autotune inserts a
    `torch.cuda.synchronize()` inside an active graph capture.

    Returns:
      List of env-var names that are set to truthy values but shouldn't be
      on this platform. Empty list means safe.
    """
    if pdl_support_expected():
        return []

    import os as _os
    misconfigured: list[str] = []
    for var in (
        "TRTLLM_ENABLE_PDL",
        "TORCHINDUCTOR_ENABLE_PDL",
    ):
        val = _os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            misconfigured.append(var)
    return misconfigured


# ═══════════════════════════════════════════════════════════════════════════
#                       AMD ROCm ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

@functools.cache
def _gcn_arch() -> str:
    """GCN architecture string (e.g. 'gfx942'). Empty string if not ROCm.

    Canonical source: vllm/platforms/rocm.py:187
    """
    if not is_amd_rocm():
        return ""
    try:
        from vllm.platforms import rocm as _rocm
        return getattr(_rocm, "_GCN_ARCH", "") or ""
    except Exception:
        return ""


def is_rocm_cdna2() -> bool:
    """AMD MI210 / MI250 — gfx90a (CDNA2 datacenter)."""
    return "gfx90a" in _gcn_arch()


def is_rocm_cdna3() -> bool:
    """AMD MI300X / MI325X — gfx942 / gfx950 (CDNA3 datacenter)."""
    arch = _gcn_arch()
    return "gfx942" in arch or "gfx950" in arch


def is_rocm_rdna() -> bool:
    """AMD Radeon RDNA3/4 — gfx11xx / gfx12xx (consumer)."""
    arch = _gcn_arch()
    return "gfx11" in arch or "gfx12" in arch


# ═══════════════════════════════════════════════════════════════════════════
#                   EXTERNAL DEPENDENCY VERSIONS (NEW v7.0)
# ═══════════════════════════════════════════════════════════════════════════

@functools.cache
def get_torch_version() -> Optional[tuple[int, int]]:
    """Returns (major, minor) torch version, or None on failure."""
    try:
        import torch
        parts = torch.__version__.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def is_torch_211_plus() -> bool:
    """True if torch >= 2.11 (required for vLLM v0.20.0+)."""
    v = get_torch_version()
    return v is not None and v >= (2, 11)


def is_torch_212_plus() -> bool:
    """True if torch >= 2.12 (forthcoming late 2026)."""
    v = get_torch_version()
    return v is not None and v >= (2, 12)


@functools.cache
def get_transformers_version() -> Optional[tuple[int, int, int]]:
    """Returns (major, minor, patch) transformers version, or None on failure."""
    try:
        import transformers
        parts = transformers.__version__.split(".")[:3]
        # Handle versions like "5.5.0rc1" by stripping non-digit suffix
        return tuple(int(''.join(c for c in p if c.isdigit())) for p in parts)
    except Exception:
        return None


def is_transformers_v5_plus() -> bool:
    """True if transformers >= 5.0.0 (required for vLLM v0.19.1+)."""
    v = get_transformers_version()
    return v is not None and v[0] >= 5


def is_transformers_v55_plus() -> bool:
    """True if transformers >= 5.5.0 (required for Gemma 4 support)."""
    v = get_transformers_version()
    return v is not None and v >= (5, 5, 0)


@functools.cache
def get_vllm_version_tuple() -> Optional[tuple[int, ...]]:
    """Returns (major, minor, patch) vllm version tuple, or None on failure.

    Example: vllm 0.20.0 -> (0, 20, 0)
    """
    try:
        import vllm
        parts = vllm.__version__.split(".")[:3]
        # Handle versions like "0.19.2rc1.dev8" by taking only leading digits
        result = []
        for p in parts:
            digits = ''.join(c for c in p.split('rc')[0].split('+')[0] if c.isdigit())
            result.append(int(digits) if digits else 0)
        return tuple(result)
    except Exception:
        return None


def is_vllm_020_plus() -> bool:
    """True if vllm >= 0.20.0."""
    v = get_vllm_version_tuple()
    return v is not None and v >= (0, 20, 0)


@functools.cache
def get_vllm_full_version_string() -> Optional[str]:
    """Returns the FULL vllm version string including local-pin suffix.

    Examples:
      "0.20.1rc1.dev16+g7a1eb8ac2"
      "0.20.2rc1.dev9+g01d4d1ad3"

    Returns None if vllm is not importable. This is the canonical pin
    identity used by `assert_vllm_pin_allowed` for protect-against-foot-gun
    enforcement.
    """
    try:
        import vllm
        return getattr(vllm, "__version__", None)
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────────────
# vLLM pin allowlist — known-good pins this Genesis revision validated against.
#
# When a patcher boot encounters an UNKNOWN pin, two policy modes are
# possible (controlled by `GENESIS_VLLM_PIN_POLICY`):
#   - "warn" (default) — log a loud warning, continue
#   - "strict"          — log + sys.exit(2) — refuse to apply patches that
#                         were never validated against this pin
#
# Add new entries via PR review only. Each line should be paired with a
# CHANGELOG entry documenting the validated test surface.
# ───────────────────────────────────────────────────────────────────────
KNOWN_GOOD_VLLM_PINS: tuple[str, ...] = (
    # v7.65 PROD baseline (validated 2026-04-23 → 2026-05-04, 1470 tests)
    "0.20.1rc1.dev16+g7a1eb8ac2",
    # v7.70 pin-bump target (validated 2026-05-04, Test 4 boot+smoke+tool-call clean)
    "0.20.2rc1.dev9+g01d4d1ad3",
)


def assert_vllm_pin_allowed(
    allowlist: tuple[str, ...] = KNOWN_GOOD_VLLM_PINS,
    policy: Optional[str] = None,
) -> tuple[str, str]:
    """Loud check that the running vllm pin is on the allowlist.

    Returns ("ok"|"unknown"|"missing", message). On policy="strict" and
    a non-ok status, raises SystemExit(2) — caller does NOT need to handle.

    Policy resolution order: explicit `policy` arg > env `GENESIS_VLLM_PIN_POLICY`
    > "warn" (default). Set policy="strict" in production start scripts to
    fail fast on accidental pin drift.

    Per Sander 2026-05-04 ("защита от дурака"): never silently apply
    patches against an unvalidated pin. The allowlist must be updated
    explicitly when a new pin is qualified.
    """
    import os as _os

    if policy is None:
        policy = _os.environ.get("GENESIS_VLLM_PIN_POLICY", "warn").strip().lower()
    if policy not in ("warn", "strict"):
        policy = "warn"

    pin = get_vllm_full_version_string()
    if pin is None:
        msg = "vllm not importable — cannot verify pin"
        if policy == "strict":
            print(f"[Genesis pin-gate] STRICT FAIL: {msg}", flush=True)
            import sys as _sys
            _sys.exit(2)
        return "missing", msg

    if pin in allowlist:
        return "ok", f"vllm pin {pin} is on the Genesis allowlist"

    msg = (
        f"vllm pin {pin!r} is NOT on the Genesis known-good list "
        f"({len(allowlist)} entries). Allowed pins: {list(allowlist)}. "
        f"To accept this pin, add it to KNOWN_GOOD_VLLM_PINS in "
        f"vllm/_genesis/guards.py and document the validation in CHANGELOG."
    )
    if policy == "strict":
        print(f"[Genesis pin-gate] STRICT FAIL: {msg}", flush=True)
        import sys as _sys
        _sys.exit(2)
    return "unknown", msg


@functools.cache
def get_flash_attn_major_version() -> Optional[int]:
    """Try to detect FlashAttention version (FA2 / FA3 / FA4).

    Returns major version int or None if not available or detection failed.

    Canonical source: vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version
    """
    try:
        # Try vllm's own helper first
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        v = get_flash_attn_version(head_size=128)
        return int(v) if v else None
    except Exception:
        pass
    try:
        # Fallback: check flash_attn module directly
        import flash_attn
        return int(flash_attn.__version__.split(".")[0])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#                      MODEL ARCHITECTURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def is_model_arch(model_config: Any, arch_name: str) -> bool:
    """Case-insensitive substring match against model_config.architectures.

    Canonical source: vllm/model_executor/models/registry.py resolution logic.

    Examples:
      is_model_arch(cfg, "Qwen3")       # True for Qwen3.5 / Qwen3.6 / Qwen3-Next
      is_model_arch(cfg, "DeepSeekV3")  # True for DeepSeek V3 family
      is_model_arch(cfg, "Llama")       # True for any Llama variant
    """
    if model_config is None:
        return False
    try:
        archs = getattr(model_config, "architectures", None) or []
        needle = arch_name.lower()
        return any(needle in (a or "").lower() for a in archs)
    except Exception:
        return False


def is_qwen3_family(model_config: Any) -> bool:
    """True for Qwen3.5 / Qwen3.6 / Qwen3-Next / Qwen3-Coder family."""
    return is_model_arch(model_config, "Qwen3")


def is_deepseek_v3(model_config: Any) -> bool:
    """True for DeepSeek V3 family (uses MLA attention, distinct from Qwen)."""
    return is_model_arch(model_config, "DeepseekV3") or is_model_arch(model_config, "DeepSeek-V3")


def is_llama_family(model_config: Any) -> bool:
    """True for Llama 3.x / 4.x family."""
    return is_model_arch(model_config, "Llama")


def is_gemma_family(model_config: Any) -> bool:
    """True for Gemma family (uses sliding-window hybrid)."""
    return is_model_arch(model_config, "Gemma")


def is_mixtral_family(model_config: Any) -> bool:
    """True for Mixtral MoE family (different router than Qwen3)."""
    return is_model_arch(model_config, "Mixtral")


# ═══════════════════════════════════════════════════════════════════════════
#                     BACKEND / KERNEL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def has_turboquant_support(cache_dtype: Optional[str]) -> bool:
    """True if TurboQuant path is active (cache_dtype starts with 'turboquant_').

    Canonical source: vllm/platforms/cuda.py:134 — routing key in CacheConfig.
    """
    return bool(cache_dtype and cache_dtype.startswith("turboquant_"))


def is_marlin_selected(fused_moe_layer: Any) -> bool:
    """Best-effort introspection: is Marlin kernel selected for this MoE layer?

    Returns False (safe default) if detection fails — we prefer to skip
    a Marlin-specific patch than apply it wrong.
    """
    try:
        kernel = getattr(fused_moe_layer, "kernel", None)
        if kernel is None:
            # Try alternate attribute paths
            kernel = getattr(fused_moe_layer, "quant_method", None)
        name = type(kernel).__name__ if kernel else ""
        return "marlin" in name.lower()
    except Exception:
        return False


def is_flash_attn_backend(attn_backend: Any) -> bool:
    """True if FlashAttention backend selected."""
    try:
        name = getattr(attn_backend, "name", "") or type(attn_backend).__name__
        return "flash" in name.lower() and "attn" in name.lower()
    except Exception:
        return False


def is_turboquant_backend(attn_backend: Any) -> bool:
    """True if TurboQuant backend selected."""
    try:
        name = getattr(attn_backend, "name", "") or type(attn_backend).__name__
        return "turboquant" in name.lower()
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#                      FILE PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

@functools.cache
def vllm_install_root() -> Optional[str]:
    """Returns absolute path to installed vllm package.

    CRITICAL: Replaces hardcoded /usr/local/lib/python3.12/dist-packages/vllm/
    that was in earlier patch versions — those break on:
      - macOS dev environments
      - venv installations
      - Python 3.13+ coming 2027
      - Docker slim/distroless images

    vllm.__file__ is the canonical universal way to locate the package.
    """
    try:
        import vllm
        import os
        return os.path.dirname(vllm.__file__)
    except Exception:
        return None


def resolve_vllm_file(relative_path: str) -> Optional[str]:
    """Returns absolute path to file within installed vllm, or None if missing.

    Example:
        resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
        -> "/path/to/vllm/v1/attention/backends/turboquant_attn.py" or None
    """
    import os
    root = vllm_install_root()
    if root is None:
        return None
    full = os.path.join(root, relative_path)
    return full if os.path.exists(full) else None


# ═══════════════════════════════════════════════════════════════════════════
#                         SUMMARY / DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

def platform_summary() -> dict[str, Any]:
    """Return full platform diagnostic dict for logging/debugging.

    Useful during patch application to log context:
        log.info("[Genesis] Platform: %s", json.dumps(platform_summary()))
    """
    return {
        "vendor": {
            "is_nvidia_cuda": is_nvidia_cuda(),
            "is_amd_rocm": is_amd_rocm(),
            "is_intel_xpu": is_intel_xpu(),
            "is_cpu_only": is_cpu_only(),
        },
        "nvidia": {
            "compute_capability": get_compute_capability(),
            "is_ampere_datacenter": is_ampere_datacenter(),
            "is_ampere_consumer": is_ampere_consumer(),
            "is_ada_lovelace": is_ada_lovelace(),
            "is_hopper": is_hopper(),
            "is_blackwell": is_blackwell(),
            "has_native_fp8": has_native_fp8(),
        },
        "amd": {
            "gcn_arch": _gcn_arch(),
            "is_cdna2": is_rocm_cdna2(),
            "is_cdna3": is_rocm_cdna3(),
            "is_rdna": is_rocm_rdna(),
        } if is_amd_rocm() else {},
        "versions": {
            "torch": get_torch_version(),
            "transformers": get_transformers_version(),
            "vllm": get_vllm_version_tuple(),
            "flash_attn_major": get_flash_attn_major_version(),
        },
        "paths": {
            "vllm_install_root": vllm_install_root(),
        },
    }
