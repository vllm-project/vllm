# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer for the FlyDSL W4A6 linear path (MXFP6-E2M3 activation x MXFP4
weight preshuffle GEMM).

The GEMM kernel, the MXFP6-E2M3 activation quantizer, and the weight/scale
preshuffle helpers are all owned by AMD's FlyDSL. vLLM is a thin consumer and
does NOT reimplement any of them:

  * at load time  -> preshuffle the MXFP4 weight + E8M0 scales
    (``shuffle_weight_w4`` / ``shuffle_scale_w4``);
  * at apply time -> MXFP6-E2M3 quant of the activation
    (``per_1x32_f6_quant``) + the ``a6w4`` GEMM.

Everything comes from an aiter build carrying the gfx950 a6w4 kernel and its
MXFP6 operand helpers (``aiter.ops.flydsl``); there is no other source and no
configuration. Released aiter does not ship them yet -- see the companion aiter
change -- so on a stock install the probe below simply returns ``False``.

Everything is gated behind :func:`is_flydsl_a6w4_supported`; when the kernel
cannot be resolved the probe returns ``False`` and the caller
(``QuarkOCP_MX``) falls back to its high-precision emulation path. This mirrors
``vllm/model_executor/layers/fused_moe/fused_flydsl_moe.py``.

Kernel constraints: gfx950 only, ``N % 128 == 0``, ``K % 256 == 0``. M is a
runtime argument on current FlyDSL (see ``_DENSE_GEMM_CANDIDATES``); the
activation quantizer still rounds it up to a multiple of 32.
"""

from __future__ import annotations

import functools
import inspect

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# One debug line the first time each distinct (path, shape) executes, so a run
# can be checked for whether the a6w4 kernel or the emulation fallback is live.
_w4a6_seen: set = set()


def w4a6_vlog(key, msg: str) -> None:
    if key not in _w4a6_seen:
        _w4a6_seen.add(key)
        logger.debug("[W4A6] %s", msg)


# ---------------------------------------------------------------------------
# AITER a6w4 surface probe.
#
# These imports define the contract vLLM consumes. They resolve only when an
# AITER build that has vendored FlyDSL #684 (flydsl>=0.2.2) is installed on a
# gfx950 box. The whole module degrades to "unavailable" otherwise -- import
# of this module never raises.
# ---------------------------------------------------------------------------
_A6W4_IMPORT_ERROR: str | None = None


# Dense GEMM entry point. FlyDSL has reshaped this three times, so we probe a
# list of (module, attr, kind) candidates rather than binding one name:
#
#   1. compile_preshuffle_gemm_a6w4 in kernels/preshuffle_gemm.py  (FlyDSL #684)
#   2. -> compile_mxfp6_gemm in kernels/mxfp4_preshuffle.py        (FlyDSL #754)
#   3. -> kernels/gemm/mxfp4_preshuffle.py                         (FlyDSL #807)
#   4. -> a single module-level launch_gemm(..., a_dtype=...) that also absorbed
#      fp8 A and fp4/fp8 B. The compile_* functions no longer exist upstream;
#      compile_mxfp6_gemm only ever shipped on a topic branch.
#
# "launch" kind is current FlyDSL: a thin @flyc.jit that caches per Constexpr
# config internally, so there is no separate compile step and M is a *runtime*
# argument (i32_m) rather than baked into the compiled config. "compile" kind is
# the older two-step API, kept so branches pinned to it still run -- but note it
# only builds against flydsl<=0.2.x: those kernels import `flydsl.expr.buffer_ops`,
# which 0.3.x removed, so with the flydsl aiter now pins (0.3.2) the compile
# candidates simply will not resolve.
#
# Both are normalized to one internal contract -- a callable with the signature
#     launcher(c, a, b, scale_a, scale_b, bias, m, n, stream)
# -- which is the shape FlyDSL's own tests use and the shape _gemm_args builds.
#
# NOTE: no *released* aiter ships a dense gfx950 MX preshuffle GEMM (checked
# through v0.1.21.dev0 -- its only mxfp6 is FMHA); it arrives with the companion
# aiter change. Several entries are listed because the kernel is still in flight
# upstream under more than one name. See why_unavailable().
_DENSE_GEMM_CANDIDATES = [
    ("aiter.ops.flydsl.kernels.mxfp4_preshuffle", "launch_gemm", "launch"),
    ("aiter.ops.flydsl.kernels.mxfp4_preshuffle", "compile_mxfp6_gemm",
     "compile"),
    ("aiter.ops.flydsl.kernels.preshuffle_gemm",
     "compile_preshuffle_gemm_a6w4", "compile"),
]


def _ptr(t: torch.Tensor):
    """Raw data_ptr as an ``fx.Pointer`` kernel arg for ``launch_gemm``."""
    import flydsl.expr as fx

    return flyc.from_c_void_p(fx.Uint8, t.contiguous().data_ptr())


def _make_launch_launcher(fn, *, n, k, tile_m, tile_n, tile_k, out_str):
    """Adapt current FlyDSL ``launch_gemm`` to the internal launcher contract.

    Single (non-strided) batch, contiguous bmn layout: batch=1 and every stride
    passed as -1, which is launch_gemm's "contiguous default" sentinel. ``bias``
    is accepted and dropped -- launch_gemm has no bias operand (the old API
    ignored it too).
    """

    def _launch(c, a, b, sa, sb, bias, m, n_, stream):  # noqa: ARG001
        fn(
            _ptr(c), _ptr(a), _ptr(b), _ptr(sa), _ptr(sb),
            int(m), int(n_), stream,
            n, k, tile_m, tile_n, tile_k,
            "fp6", out_str, "fp4",
            1,                              # batch
            -1, -1, -1, -1, -1, -1,         # a/sca/c row+batch strides
            0,                              # waves_per_eu
        )

    return _launch


def _make_compile_launcher(fn, *, m_pad, n, k, tile_m, tile_n, tile_k, out_str):
    """Adapt the older two-step ``compile_*`` API to the same contract.

    flyc.compile needs concrete operands, so the compile happens on first call
    and the CompiledFunction is reused after that.
    """
    try:
        accepts_m_hint = "M_hint" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # unintrospectable callable
        accepts_m_hint = False
    kwargs = dict(N=n, K=k, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                  out_dtype=out_str)
    if accepts_m_hint:
        kwargs["M_hint"] = m_pad
    state: dict = {}

    def _launch(*args):
        cf = state.get("cf")
        if cf is None:
            cf = flyc.compile(fn(**kwargs), *args)
            state["cf"] = cf
        cf(*args)

    return _launch


def _resolve_dense_gemm():
    """Return ``(fn, kind)`` for the first resolvable dense GEMM candidate."""
    import importlib

    last = None
    for mod, attr, kind in _DENSE_GEMM_CANDIDATES:
        try:
            fn = getattr(importlib.import_module(mod), attr)
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        # Symbol presence is not compatibility. _make_launch_launcher binds
        # b_dtype positionally; a launch_gemm predating that parameter would
        # silently shift every stride argument, so reject it here rather than
        # after the load-time weight preshuffle has discarded the fallback
        # layout.
        if kind == "launch":
            # launch_gemm is a flydsl JitFunction, which inspect.signature()
            # cannot read; the undecorated callable hangs off .func. If neither
            # is introspectable, accept rather than reject a working kernel.
            try:
                params = inspect.signature(getattr(fn, "func", fn)).parameters
            except (TypeError, ValueError):
                params = None
            if params is not None and "b_dtype" not in params:
                last = TypeError(
                    f"{mod}.{attr} predates the b_dtype parameter; "
                    "incompatible with this adapter"
                )
                continue
        return fn, kind
    raise ImportError(
        f"no FlyDSL dense a6w4/mxfp6 GEMM found (tried "
        f"{[(c[0], c[1]) for c in _DENSE_GEMM_CANDIDATES]}); last error: {last!r}"
    )


# Operand-prep helpers. FlyDSL renamed the module that hosts them
# (tests/kernels/utils/fp4_utils.py -> gemm_common_utils.py); aiter's
# mxfp6_utils is the third option for when the kernel is vendored there.
_HELPER_MODULES = (
    "aiter.ops.flydsl.mxfp6_utils",
)


def _resolve_helpers():
    import importlib

    last = None
    for mod in _HELPER_MODULES:
        try:
            m = importlib.import_module(mod)
            return (
                m.per_1x32_f6_quant,
                m.shuffle_scale_w4,
                m.shuffle_weight_w4,
            )
        except Exception as e:  # noqa: BLE001
            last = e
    raise ImportError(
        f"no FlyDSL a6w4 operand helpers found (tried {list(_HELPER_MODULES)}); "
        f"last error: {last!r}"
    )


def _import_a6w4_kernels():
    """Resolve flyc + the dense a6w4/mxfp6 GEMM + the w4 operand helpers.

    The kernel *source* is vendored into aiter; the flydsl *compiler* comes
    from the installed flydsl wheel, which aiter already pins.
    """
    import flydsl.compiler as flyc  # noqa: F401

    gemm, gemm_kind = _resolve_dense_gemm()
    per_1x32_f6_quant, shuffle_scale_w4, shuffle_weight_w4 = _resolve_helpers()
    return (
        flyc,
        gemm,
        gemm_kind,
        per_1x32_f6_quant,
        shuffle_scale_w4,
        shuffle_weight_w4,
    )


# flyc is imported independently of the dense-GEMM resolution so the fast fp6
# activation-quant kernel can still load even if the dense GEMM API drifted.
try:
    import flydsl.compiler as flyc
except Exception:  # noqa: BLE001
    flyc = None  # type: ignore[assignment]

try:
    (
        flyc,
        _dense_gemm_fn,
        _dense_gemm_kind,
        per_1x32_f6_quant,
        shuffle_scale_w4,
        shuffle_weight_w4,
    ) = _import_a6w4_kernels()
    _A6W4_KERNELS_IMPORTED = True
except Exception as exc:  # noqa: BLE001
    _A6W4_IMPORT_ERROR = repr(exc)
    _A6W4_KERNELS_IMPORTED = False
    # NOTE: do NOT null `flyc` here -- the standalone import above keeps it so
    # the fp6 act-quant kernel still works when only the dense GEMM is missing.
    _dense_gemm_fn = None  # type: ignore[assignment]
    _dense_gemm_kind = None  # type: ignore[assignment]
    per_1x32_f6_quant = None  # type: ignore[assignment]
    shuffle_scale_w4 = None  # type: ignore[assignment]
    shuffle_weight_w4 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Optional FlyDSL FP6-E2M3 activation-quant kernel. The default activation
# quantizer shipped with #684 (``per_1x32_f6_quant``) is a pure-PyTorch
# reference that is 46-69x slower than the fp8/mxfp4 HIP quantizers and
# dominates (~85%) the W4A6 forward. This swaps in the FlyDSL DSL quant kernel
# (bf16 -> FP6 codes + E8M0 scales) when aiter ships it, and falls back to
# ``per_1x32_f6_quant`` when it is unavailable or K % 256 != 0.
# ---------------------------------------------------------------------------
_compile_quant_act_a6_flydsl = None
_FP6_QUANT_IMPORT_ERROR: str | None = None


def _import_fp6_quant_kernel():
    """Resolve ``compile_quant_act_a6_flydsl`` (bf16 -> FP6 codes + E8M0)."""
    from aiter.ops.flydsl.quant_act_a6_flydsl import compile_quant_act_a6_flydsl

    return compile_quant_act_a6_flydsl


# Gated only on flyc (NOT on the dense-GEMM import) so the fast fp6 act-quant
# kernel loads even if the dense a6w4 GEMM API drifted / is unavailable.
if flyc is not None:
    try:
        _compile_quant_act_a6_flydsl = _import_fp6_quant_kernel()
    except Exception as exc:  # noqa: BLE001
        _FP6_QUANT_IMPORT_ERROR = repr(exc)


_A6W4_QUANT_CACHE: dict = {}


def _flydsl_fp6_quant(x: torch.Tensor, m_pad: int, k: int):
    """FP6-E2M3 activation quant via the FlyDSL kernel.

    ``x`` must already be ``m_pad``-tall. Returns ``(codes [m_pad, k] uint8,
    scale_raw [m_pad, k//32] uint8)`` matching ``per_1x32_f6_quant``'s a_pad /
    raw-scale outputs, or ``None`` when the kernel is unavailable or the kernel
    constraint ``K % 256 == 0`` is not met (caller then falls back).

    K % 256 (not 512) is required since the kernel gained a 32-thread
    half-wavefront tile for K=256 — this is exactly the DeepSeek-R1 TP=8 MoE
    stage-2 shape (inter_dim = 2048/8 = 256), which previously fell back to the
    ~48x-slower PyTorch ref on every MoE layer.

    The kernel reads its input as BF16. Quark also advertises FP16 activations
    (``QuarkConfig.get_supported_act_dtypes``), and nothing upstream converts,
    so an FP16 tensor would be reinterpreted bit-for-bit as BF16 and produce
    garbage rather than an error. Return None for anything but BF16 and let the
    caller use the dtype-agnostic reference path.
    """
    if (
        _compile_quant_act_a6_flydsl is None
        or k % 256 != 0
        or x.dtype is not torch.bfloat16
    ):
        return None
    w4a6_vlog(
        ("dense-actq-fly", k),
        f"dense activation-quant = FlyDSL fp6 (mxfp6-e2m3) kernel  K={k} "
        f"m_pad={m_pad}  [fast HIP path]",
    )
    m_align = max(m_pad, 32)
    codes = torch.zeros((m_pad, k), dtype=torch.uint8, device=x.device)
    scales = torch.zeros((m_align, k // 32), dtype=torch.uint8, device=x.device)
    stream = torch.cuda.current_stream()
    args = (x.contiguous().view(-1), codes.view(-1), scales.view(-1), m_pad, stream)
    key = (m_pad, k, x.dtype)
    launch = _A6W4_QUANT_CACHE.get(key)
    if launch is None:
        launch = flyc.compile(
            _compile_quant_act_a6_flydsl(M=m_pad, K=k, shuffled=False), *args
        )
        _A6W4_QUANT_CACHE[key] = launch
    launch(*args)
    return codes, scales[:m_pad]


@functools.cache
def is_flydsl_a6w4_supported() -> bool:
    """Whether the FlyDSL a6w4 linear kernel can be used on this system.

    Capability only (platform + arch + kernels importable); it does NOT
    consider any per-layer shape eligibility (see
    :func:`is_flydsl_a6w4_eligible`).
    """
    if not _A6W4_KERNELS_IMPORTED:
        return False
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()
    except Exception:  # noqa: BLE001
        return False


def is_flydsl_a6w4_eligible(n: int, k: int) -> bool:
    """Shape eligibility for the a6w4 GEMM: ``N % 128 == 0`` and
    ``K % 256 == 0`` (other shapes fall back to emulation)."""
    return n % 128 == 0 and k % 256 == 0


def why_unavailable() -> str:
    """Human-readable reason the a6w4 path is off, for one-time logging."""
    if not _A6W4_KERNELS_IMPORTED:
        return (
            f"FlyDSL a6w4 kernels not importable ({_A6W4_IMPORT_ERROR}); no "
            "released aiter ships a dense gfx950 MX preshuffle GEMM yet"
        )
    if not current_platform.is_rocm():
        return "not on ROCm"
    return "device is not gfx950"


# ---------------------------------------------------------------------------
# Weight-side preshuffle (load time). Thin wrappers over the AITER helpers so
# QuarkOCP_MX never imports them directly.
# ---------------------------------------------------------------------------
def preshuffle_weight(weight: torch.Tensor) -> torch.Tensor:
    """Preshuffle packed-uint8 MXFP4 weight ``[N, K//2]`` for the a6w4 GEMM."""
    assert shuffle_weight_w4 is not None
    return shuffle_weight_w4(weight.contiguous(), 16, False, False)


def preshuffle_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """Preshuffle uint8 E8M0 weight scales ``[N, K//32]`` for the a6w4 GEMM."""
    assert shuffle_scale_w4 is not None
    return shuffle_scale_w4(weight_scale.contiguous(), 1, False)


# ---------------------------------------------------------------------------
# GEMM dispatch (apply time). compile -> flyc.compile -> cache -> call,
# mirroring fused_flydsl_moe.py.
# ---------------------------------------------------------------------------
_A6W4_GEMM_CACHE: dict = {}


def _to_bytes(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.uint8, torch.int8):
        return t
    return t.view(torch.uint8)


def _gemm_args(c, a, b, sa, sb, bias, m, n, stream):
    return (
        c.contiguous().view(-1),
        _to_bytes(a).contiguous().view(-1),
        _to_bytes(b).contiguous().view(-1),
        _to_bytes(sa).contiguous().view(-1),
        _to_bytes(sb).contiguous().view(-1),
        bias,
        int(m),
        int(n),
        stream,
    )


def _flydsl_a6w4_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """MXFP6-E2M3(x) @ MXFP4(weight).T via the FlyDSL a6w4 preshuffle GEMM.

    ``weight`` / ``weight_scale`` are already preshuffled at load time
    (:func:`preshuffle_weight` / :func:`preshuffle_weight_scale`).
    """
    assert per_1x32_f6_quant is not None and _dense_gemm_fn is not None

    m_real = x.shape[0]
    k = x.shape[1]
    n = weight.shape[0]
    device = x.device
    # The kernel emits bf16 or fp16 only. Mapping "anything else" to fp16 is not
    # safe: an fp32 out_dtype (what torch.get_default_dtype() returns if vLLM has
    # not set the model dtype yet) would have the kernel write fp16 elements into
    # an fp32 tensor and return silent garbage rather than raising.
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"[W4A6] out_dtype must be bfloat16 or float16; got {out_dtype}"
        )
    out_str = "bf16" if out_dtype == torch.bfloat16 else "fp16"
    is_launch = _dense_gemm_kind == "launch"

    # F.linear accepts an empty leading dim, and vLLM does hand one down (an
    # empty batch, or a [B, 0, K] reshape). The kernel does not: grid.x is
    # ceil(M/tile_m), so M=0 launches a zero-sized grid over zero-length A/C
    # pointers. Return the empty result before quantizing or launching.
    if m_real == 0:
        return torch.empty((0, n), dtype=out_dtype, device=device)

    tile_m, tile_n, tile_k = _pick_tiles((m_real + 31) // 32 * 32, n, k)

    # The activation quantizer works on 32-row groups, so A always gets padded to
    # a multiple of 32 (and at least 32) before quantization.
    #
    # How far beyond that we pad depends on the kernel API:
    #  * "launch": M is a runtime argument (i32_m), the kernel guards the M
    #    remainder itself, and C is allocated at the true M -- this mirrors
    #    FlyDSL's own a6w4 test. No tile_m padding, and the compiled config no
    #    longer varies with M, so one compile serves every batch size.
    #  * "compile": M is baked into the config and the kernel tiles M by tile_m
    #    with NO remainder guard, so an M that is not a multiple of tile_m makes
    #    the last tile read A / write C up to tile_m rows out of bounds. That
    #    silently corrupts adjacent memory when it lands in a mapped allocation
    #    and faults when it hits an unmapped page (the R1 long-generation crash).
    m_pad = max(32, (m_real + 31) // 32 * 32)
    if not is_launch:
        m_pad = (m_real + tile_m - 1) // tile_m * tile_m
    if m_pad != m_real:
        x = torch.nn.functional.pad(x, (0, 0, 0, m_pad - m_real))

    quant = _flydsl_fp6_quant(x, m_pad, k)
    if quant is not None:
        a_pad, scale_a_raw = quant
    else:
        w4a6_vlog(
            ("dense-actq-ref", k),
            f"dense activation-quant = PyTorch ref per_1x32_f6_quant  K={k} "
            f"[slow fallback: kernel unavailable, K%256!=0, or non-bf16 act]",
        )
        a_pad, scale_a_raw, _ = per_1x32_f6_quant(x)
    # Scales stay 32-row aligned; only the codes are trimmed to the real M.
    scale_a = shuffle_scale_w4(scale_a_raw, 1, False)
    a_operand = a_pad[:m_real] if is_launch else a_pad
    m_arg = m_real if is_launch else m_pad

    w4a6_vlog(
        ("dense-gemm", n, k),
        f"dense GEMM = FlyDSL a6w4 preshuffle kernel (mxfp6 act x mxfp4 wt)  "
        f"N={n} K={k} tiles={tile_m}x{tile_n}x{tile_k} api={_dense_gemm_kind}",
    )
    stream = torch.cuda.current_stream()

    c_out = torch.zeros((m_arg, n), dtype=out_dtype, device=device)
    dummy_bias = torch.empty(0, dtype=out_dtype, device=device)

    # M only participates in the cache key on the compile API, where it is baked.
    key = (None if is_launch else m_pad, n, k, tile_m, tile_n, tile_k, out_str)
    launcher = _A6W4_GEMM_CACHE.get(key)
    if launcher is None:
        common = dict(n=n, k=k, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                      out_str=out_str)
        launcher = (
            _make_launch_launcher(_dense_gemm_fn, **common)
            if is_launch
            else _make_compile_launcher(_dense_gemm_fn, m_pad=m_pad, **common)
        )
        _A6W4_GEMM_CACHE[key] = launcher

    launcher(
        *_gemm_args(
            c_out, a_operand, weight, scale_a, weight_scale, dummy_bias,
            m_arg, n, stream,
        )
    )
    return c_out[:m_real]


def _flydsl_a6w4_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty((x.shape[0], weight.shape[0]), dtype=out_dtype, device=x.device)


@functools.cache
def _aiter_pick_mx_tiles():
    """aiter's tuned tile chooser, or None on an aiter predating it."""
    try:
        from aiter.ops.flydsl.batched_gemm_mxfp4 import pick_mx_tiles

        return pick_mx_tiles
    except Exception:  # noqa: BLE001
        return None


def _pick_tiles(m_pad: int, n: int, k: int) -> tuple[int, int, int]:
    """Pick (tile_m, tile_n, tile_k) for the a6w4 GEMM.

    Defers to aiter's ``pick_mx_tiles``, which consults the tuned table in
    ``aiter/configs/a6w4_flydsl_tuned_gemm.csv`` (exact gfx/cu_num/M/N/K/a_dtype
    matches) and falls through to its own occupancy-aware heuristic otherwise.
    Tile choice belongs next to the kernel, so vLLM does not carry a second
    table that can drift from it.

    The local fallback below is only for an aiter that has the a6w4 kernel but
    not the tuner. tile_m's floor is 32 because the kernel requires tile_m % 32
    (its A e8m0 scale is 32-row granular); callers already round m_pad up to 32.
    """
    pick = _aiter_pick_mx_tiles()
    if pick is not None:
        return pick(m_pad, n, k, "fp6")
    return (32 if m_pad <= 64 else 64), 128, 256


# Register the custom op only when the kernels are importable, so torch.compile
# tracing on systems without AITER never sees a half-defined op.
if _A6W4_KERNELS_IMPORTED:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="flydsl_a6w4_linear",
        op_func=_flydsl_a6w4_linear_impl,
        mutates_args=[],
        fake_impl=_flydsl_a6w4_linear_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def flydsl_a6w4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Public entrypoint: dispatches through the registered custom op."""
    return torch.ops.vllm.flydsl_a6w4_linear(x, weight, weight_scale, out_dtype)
