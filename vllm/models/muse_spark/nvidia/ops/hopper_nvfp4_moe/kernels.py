# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper (SM90a) NVFP4 MoE experts for Muse Spark.

Builds and dispatches two self-contained SM90a CUDA extensions with no external
CUTLASS or FlashInfer build dependency:

- `vllm_muse_spark_hopper_nvfp4_moe_decode` (`csrc/nvfp4_moe_decode.cu`):
  persistent decode/verify engine for `1 <= T <= 1536`, fusing W13+SwiGLU, W2,
  weight-first post-expert RMSNorm, and FP32 weighted top-k reduction, plus a
  CUDA weight packer for load-time E5M3 encoding.
- `vllm_muse_spark_hopper_nvfp4_moe_prefill` (`csrc/nvfp4_moe_prefill.cu`):
  prefill engine for `T >= 1537`, chunked at `MAX_PREFILL_CHUNK = 8192`.

When JIT compilation fails or a layer is ineligible, callers fall back cleanly
to vLLM's Hopper Marlin MoE path.
"""

import fcntl
import functools
import os
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Static,
)
from vllm.model_executor.utils import refill_derived, replace_parameter
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.muse_spark.nvidia.muse_moe_runner import (
        MusePostExpertFinalize,
        MuseSparkModelOptNvFp4FusedMoE,
        MuseSparkRoutedExperts,
    )

logger = init_logger(__name__)

# Fixed Muse Spark EP8 Hopper geometry supported by the Hopper NVFP4 MoE kernels.
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 4096
NUM_LOCAL_EXPERTS = 32
MAX_GLOBAL_EXPERTS = 256
TOP_K = 8
PACKED_UNIT_BYTES = 9216
DECODE_MAX_TOKENS = 1536
MAX_PREFILL_CHUNK = 8192
MAX_DECODE_ROWS = DECODE_MAX_TOKENS * TOP_K
PLAN_BYTES = MAX_GLOBAL_EXPERTS + NUM_LOCAL_EXPERTS

# Global scale factor matching the BF16 exponent bias in the kernel dequantizer.
GS_FIX = 128.0
_E5_BIAS = 16
# Largest biased E5M3 exponent the kernel decodes to a finite BF16 scale:
# (code << 4) + 0x7300 has BF16 exponent field e5 + 230, which must be <= 254.
_E5_MAX_BIASED_EXP = 24
_OFF_RANK_PLAN = 255
_CTRL_INTS = 256
_PART_COLS = 128 * 256
_DEFAULT_POST_EPS = 1e-8
_CUDA_PACK_SCALE_DTYPES = (torch.float8_e4m3fn, torch.bfloat16, torch.float32)

# Prefill tile-scheduling cost table (mode 0, mode 1, and split-K threshold).
_PREFILL_COST: list[int] = [
    653, 572, 505, 425, 697, 604, 528, 617, 1059,
    662, 582, 522, 428, 729, 583, 496, 662, 919, 1000,
]  # fmt: skip

# Bit positions within each 32-bit packed FP4 word for (pair_idx, lane_idx).
_FP4_BIT_POSITIONS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (0, 0): (6, 7, 8, 15),
    (0, 1): (22, 23, 24, 31),
    (1, 0): (28, 29, 30, 5),
    (1, 1): (12, 13, 14, 21),
    (2, 0): (25, 26, 27, 2),
    (2, 1): (9, 10, 11, 18),
    (3, 0): (3, 4, 0, 1),
    (3, 1): (19, 20, 16, 17),
}

_HERE = os.path.dirname(os.path.realpath(__file__))
_CSRC_DIR = os.path.join(_HERE, "csrc")
_COMMON_HEADER = os.path.join(_CSRC_DIR, "nvfp4_moe_common.cuh")
_DECODE_SOURCE = os.path.join(_CSRC_DIR, "nvfp4_moe_decode.cu")
_PREFILL_SOURCE = os.path.join(_CSRC_DIR, "nvfp4_moe_prefill.cu")

_DECODE_EXT_NAME = "vllm_muse_spark_hopper_nvfp4_moe_decode"
_PREFILL_EXT_NAME = "vllm_muse_spark_hopper_nvfp4_moe_prefill"

# The explicit -gencode makes torch ignore TORCH_CUDA_ARCH_LIST: sm_90a only.
_EXTRA_CUDA_CFLAGS = [
    "-O3",
    "-std=c++20",
    "-gencode=arch=compute_90a,code=sm_90a",
]

_EncodedWeights = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

_extensions: tuple[Any, Any] | None = None
_build_failed: bool = False
_build_lock = threading.Lock()


def _current_cuda_sm_count() -> int | None:
    """Return the active CUDA device SM count, or None if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    dev_idx = torch.accelerator.current_device_index()
    return int(torch.cuda.get_device_properties(dev_idx).multi_processor_count)


def is_hopper_nvfp4_moe_supported(
    moe_config: FusedMoEConfig,
    post_expert_finalize: "MusePostExpertFinalize | None",
) -> bool:
    """Return whether a Muse Spark MoE layer is eligible for Hopper NVFP4 MoE.

    Args:
        moe_config: Layer MoE configuration.
        post_expert_finalize: Attached post-expert RMSNorm finalizer, if any.

    Returns:
        True only when Hopper SM90a, SM count, shapes, parallelism, activation,
        and post-expert norm all satisfy the Hopper NVFP4 MoE contract.

    """
    experts_cls = MuseSparkHopperNvFp4Experts
    if not experts_cls._supports_current_device():
        return False
    num_sms = _current_cuda_sm_count()
    if num_sms is not None and not (20 <= num_sms < 256):
        return False
    if (
        moe_config.moe_backend != "auto"
        or envs.VLLM_BATCH_INVARIANT
        or not experts_cls._supports_parallel_config(moe_config.moe_parallel_config)
        or moe_config.num_local_experts != NUM_LOCAL_EXPERTS
        or not (NUM_LOCAL_EXPERTS <= moe_config.num_experts <= MAX_GLOBAL_EXPERTS)
        or moe_config.hidden_dim != HIDDEN_DIM
        or moe_config.hidden_dim_unpadded != HIDDEN_DIM
        or moe_config.intermediate_size_per_partition != INTERMEDIATE_DIM
        or moe_config.experts_per_token != TOP_K
        or moe_config.in_dtype != torch.bfloat16
        or not moe_config.is_act_and_mul
        or moe_config.activation != MoEActivation.SILU
        or moe_config.is_lora_enabled
        or moe_config.has_bias
        or moe_config.swiglu_limit is not None
        or moe_config.swiglu_alpha is not None
        or moe_config.swiglu_beta is not None
    ):
        return False

    if post_expert_finalize is None or post_expert_finalize.pre_norm_all_reduce:
        return False

    norm = post_expert_finalize._norm
    if not getattr(norm, "applies_weight_before_norm", False):
        return False
    if getattr(norm, "variance_size_override", None) is not None:
        return False
    return not (
        norm.has_weight
        and (norm.weight is None or tuple(norm.weight.shape) != (HIDDEN_DIM,))
    )


def _stage_source(src: str, build_dir: str) -> str:
    """Atomically copy `src` into `build_dir` if absent or modified.

    Preserving identical files keeps their mtime intact so warm-cache loads
    across ranks do not trigger spurious recompiles, while atomic rename makes
    concurrent multi-rank staging and read-only site-packages safe.
    """
    dst = os.path.join(build_dir, os.path.basename(src))
    with open(src, "rb") as f:
        data = f.read()
    try:
        with open(dst, "rb") as f:
            if f.read() == data:
                return dst
    except OSError:
        pass

    partial = f"{dst}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(partial, "wb") as f:
            f.write(data)
        os.replace(partial, dst)
    except Exception:
        with suppress(OSError):
            os.unlink(partial)
        raise
    return dst


def _compile_extension(name: str, source_path: str) -> Any:
    """Compile or load one Hopper SM90a NVFP4 MoE CUDA extension."""
    from torch.utils.cpp_extension import _get_build_directory, load

    build_dir = _get_build_directory(name, verbose=False)
    os.makedirs(build_dir, exist_ok=True)
    lock_path = os.path.join(build_dir, ".build.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _stage_source(_COMMON_HEADER, build_dir)
            staged_source = _stage_source(source_path, build_dir)

            saved_path = os.environ.get("PATH")
            exe_dir = os.path.dirname(sys.executable)
            if exe_dir and exe_dir not in (saved_path or "").split(os.pathsep):
                os.environ["PATH"] = (
                    f"{exe_dir}{os.pathsep}{saved_path}" if saved_path else exe_dir
                )
            try:
                return load(
                    name=name,
                    sources=[staged_source],
                    extra_cuda_cflags=list(_EXTRA_CUDA_CFLAGS),
                    extra_include_paths=[build_dir],
                    build_directory=build_dir,
                    verbose=False,
                )
            finally:
                if saved_path is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = saved_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build_hopper_nvfp4_moe() -> tuple[Any, Any] | None:
    """Build or load the Hopper NVFP4 MoE decode and prefill CUDA extensions.

    Returns:
        `(decode_ext, prefill_ext)` on success, or `None` if building failed
        (caching the failure and logging a warning so callers fall back cleanly
        to Marlin).

    """
    global _extensions, _build_failed
    if _extensions is not None:
        return _extensions
    if _build_failed:
        return None
    with _build_lock:
        if _extensions is not None:
            return _extensions
        if _build_failed:
            return None
        try:
            decode_ext = _compile_extension(_DECODE_EXT_NAME, _DECODE_SOURCE)
            prefill_ext = _compile_extension(_PREFILL_EXT_NAME, _PREFILL_SOURCE)
            _extensions = (decode_ext, prefill_ext)
        except Exception as err:
            _build_failed = True
            logger.warning(
                "Muse Spark Hopper NVFP4 MoE: building extensions failed "
                "(%s: %s); falling back to Marlin.",
                type(err).__name__,
                err,
            )
            return None
        return _extensions


@functools.cache
def _fp4_pack_lut_cpu() -> torch.Tensor:
    """Build a `[4, 256]` `int32` lookup table for bit-packing FP4 pairs.

    Each entry `lut[p, c0 | (c1 << 4)]` places the 4 bits of FP4 code `c0`
    at `_FP4_BIT_POSITIONS[(p, 0)]` and `c1` at `_FP4_BIT_POSITIONS[(p, 1)]`.
    Because all 32 target bit positions across `p in 0..3` are pairwise
    disjoint, bitwise-OR of the four looked-up words reproduces the exact
    32-bit packed layout without `int64` temporaries.
    """
    table = torch.zeros((4, 256), dtype=torch.int64)
    for pair_idx in range(4):
        pos0 = _FP4_BIT_POSITIONS[(pair_idx, 0)]
        pos1 = _FP4_BIT_POSITIONS[(pair_idx, 1)]
        for byte_val in range(256):
            c0 = byte_val & 0xF
            c1 = byte_val >> 4
            word = 0
            for bit, pos in enumerate(pos0):
                if (c0 >> bit) & 1:
                    word |= 1 << pos
            for bit, pos in enumerate(pos1):
                if (c1 >> bit) & 1:
                    word |= 1 << pos
            if word >= (1 << 31):
                word -= 1 << 32
            table[pair_idx, byte_val] = word
    return table.to(torch.int32)


def w13_row_permutation(device: torch.device | str) -> torch.Tensor:
    """Return the `[8192]` row permutation interleaving gate and up rows."""
    rows = torch.arange(2 * INTERMEDIATE_DIM, device=device)
    mb = rows // 256
    tile = (rows % 256) // 64
    wq = (rows % 64) // 16
    hi = (rows % 16) // 8
    group = rows % 8
    return 128 * mb + 32 * tile + 8 * wq + group + INTERMEDIATE_DIM * hi


def encode_rows(
    wq: torch.Tensor,
    sf: torch.Tensor,
    row_perm: torch.Tensor | None = None,
    *,
    _lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode one expert's packed FP4 weights and block scales.

    Args:
        wq: Packed FP4 weights of shape `[R, 2048]` (`uint8` or
            `float4_e2m1fn_x2`), with `R` a positive multiple of 256.
        sf: Block scales of shape `[R, 256]` (typically `float8_e4m3fn`).
        row_perm: Optional `[R]` row permutation applied before packing.
        _lut: Optional pre-staged `[4, 256]` `int32` packing table on
            `wq.device`.

    Returns:
        Packed `uint8` tensor of shape `[R // 256, 64, 9216]`.

    Raises:
        ValueError: If shapes are invalid or any block scale is non-finite or
            cannot be represented exactly in the packed E5M3 format.

    """
    wq_u8 = wq.view(torch.uint8)
    if (
        wq_u8.ndim != 2
        or wq_u8.shape[0] <= 0
        or wq_u8.shape[0] % 256 != 0
        or wq_u8.shape[1] != HIDDEN_DIM // 2
    ):
        raise ValueError(
            f"Expected wq shape [256*k, {HIDDEN_DIM // 2}], got {tuple(wq_u8.shape)}."
        )
    if sf.shape != (wq_u8.shape[0], HIDDEN_DIM // 16):
        raise ValueError(
            f"Expected sf shape ({wq_u8.shape[0]}, {HIDDEN_DIM // 16}), "
            f"got {tuple(sf.shape)}."
        )

    if row_perm is not None:
        wq_u8 = wq_u8[row_perm]
        sf = sf[row_perm]

    num_rows = wq_u8.shape[0]
    codes = torch.stack((wq_u8 & 0xF, wq_u8 >> 4), dim=-1).view(num_rows, 256, 16)
    vals = sf.float()
    if not torch.isfinite(vals).all():
        raise ValueError("Non-finite NVFP4 block scale encountered.")

    neg = vals < 0
    zero = vals == 0
    codes = torch.where(neg.unsqueeze(-1), codes ^ 8, codes)
    codes = torch.where(zero.unsqueeze(-1), torch.zeros_like(codes), codes)

    mag = torch.where(zero, torch.ones_like(vals), vals.abs())
    mantissa, exp = torch.frexp(mag)
    biased_exp = exp - 1 + _E5_BIAS
    m3_float = (mantissa * 2.0 - 1.0) * 8.0
    m3_int = m3_float.round().to(torch.int32)
    if (
        not torch.equal(m3_int.float(), m3_float)
        or int(biased_exp.min().item()) < 0
        or int(biased_exp.max().item()) > _E5_MAX_BIASED_EXP
    ):
        raise ValueError("Block scale cannot be encoded exactly in E5M3.")
    scale_bytes = ((biased_exp << 3) | m3_int).to(torch.uint8)

    mb = num_rows // 256
    # Tile FP4 codes into [MB, 64, 4, 4, 8, 4, 4, 4, 2] matching the kernel's
    # 64-K-slice x 4-subtile swizzle, then bit-pack each (4, 2) group into i32.
    tiled = codes.view(mb, 4, 4, 2, 8, 64, 4, 2, 4, 2)
    tiled = tiled.permute(0, 5, 1, 2, 4, 8, 6, 7, 3, 9).contiguous()
    tiled = tiled.view(mb, 64, 4, 4, 8, 4, 4, 4, 2)

    lut = _fp4_pack_lut_cpu().to(wq_u8.device) if _lut is None else _lut
    pair_bytes = (tiled[..., 0] | (tiled[..., 1] << 4)).to(torch.int32)
    packed_words = lut[0][pair_bytes[..., 0]]
    for pair_idx in range(1, 4):
        packed_words.bitwise_or_(lut[pair_idx][pair_bytes[..., pair_idx]])

    fp4_bytes = packed_words.contiguous().view(mb, 64, 4, 512).view(torch.uint8)
    packed_scales = (
        scale_bytes.view(mb, 4, 4, 2, 8, 64, 4)
        .permute(0, 5, 1, 2, 4, 6, 3)
        .contiguous()
        .view(mb, 64, 4, 256)
    )
    return (
        torch.cat((fp4_bytes, packed_scales), dim=3)
        .contiguous()
        .view(mb, 64, PACKED_UNIT_BYTES)
    )


@torch.no_grad()
def encode_hopper_nvfp4_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    *,
    decode_ext: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode raw NVFP4 layer weights and scales into packed Hopper format.

    Uses the fused CUDA packer on `decode_ext` when CUDA tensors and supported
    block-scale dtypes are provided, and falls back to the pure-PyTorch
    per-expert encoder on CPU or when no CUDA packer is available.

    Args:
        w13: Raw gate/up FP4 weights of shape `[E, 8192, 2048]`.
        w13_scale: Raw gate/up block scales of shape `[E, 8192, 256]`.
        w13_scale_2: Gate/up global scales of shape `[E, 2]` (or `[E]`).
        w2: Raw down-proj FP4 weights of shape `[E, 4096, 2048]`.
        w2_scale: Raw down-proj block scales of shape `[E, 4096, 256]`.
        w2_scale_2: Down-proj global scales of shape `[E]` (or `[E, 1]`).
        decode_ext: Optional loaded Hopper decode extension providing
            `pack_experts`.

    Returns:
        Tuple `(w13_packed, w2_packed, gs13_gate, gs13_up, gs2)` where
        `w13_packed` is `uint8[E, 32, 64, 9216]`, `w2_packed` is
        `uint8[E, 16, 64, 9216]`, and the three global scales are contiguous
        `float32[E]` tensors scaled by `GS_FIX = 128.0`.

    Raises:
        ValueError: If any tensor shape or dtype is invalid, or any block or
            global scale is non-finite or cannot be encoded exactly in E5M3.

    """
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError(
            f"Expected 3D weights: w13={tuple(w13.shape)}, w2={tuple(w2.shape)}."
        )
    num_experts = w13.shape[0]
    if num_experts <= 0 or w2.shape[0] != num_experts:
        raise ValueError(
            f"Mismatched expert counts: w13={tuple(w13.shape)}, w2={tuple(w2.shape)}."
        )
    if w13.element_size() != 1 or w2.element_size() != 1:
        raise ValueError(
            f"Expected 1-byte FP4 weights, got w13={w13.dtype}, w2={w2.dtype}."
        )
    if w13.shape != (num_experts, 2 * INTERMEDIATE_DIM, HIDDEN_DIM // 2):
        raise ValueError(
            f"Expected w13 shape ({num_experts}, {2 * INTERMEDIATE_DIM}, "
            f"{HIDDEN_DIM // 2}), got {tuple(w13.shape)}."
        )
    if w2.shape != (num_experts, HIDDEN_DIM, INTERMEDIATE_DIM // 2):
        raise ValueError(
            f"Expected w2 shape ({num_experts}, {HIDDEN_DIM}, "
            f"{INTERMEDIATE_DIM // 2}), got {tuple(w2.shape)}."
        )
    if w13_scale.shape != (num_experts, 2 * INTERMEDIATE_DIM, HIDDEN_DIM // 16):
        raise ValueError(
            f"Expected w13_scale shape ({num_experts}, {2 * INTERMEDIATE_DIM}, "
            f"{HIDDEN_DIM // 16}), got {tuple(w13_scale.shape)}."
        )
    if w2_scale.shape != (num_experts, HIDDEN_DIM, INTERMEDIATE_DIM // 16):
        raise ValueError(
            f"Expected w2_scale shape ({num_experts}, {HIDDEN_DIM}, "
            f"{INTERMEDIATE_DIM // 16}), got {tuple(w2_scale.shape)}."
        )

    if w13_scale_2.ndim == 2 and w13_scale_2.shape == (num_experts, 2):
        gs13_gate = (w13_scale_2[:, 0].float() * GS_FIX).contiguous()
        gs13_up = (w13_scale_2[:, 1].float() * GS_FIX).contiguous()
    elif w13_scale_2.numel() == num_experts:
        gs13_gate = (w13_scale_2.reshape(num_experts).float() * GS_FIX).contiguous()
        gs13_up = gs13_gate.clone()
    else:
        raise ValueError(
            f"Expected w13_scale_2 shape ({num_experts}, 2) or "
            f"({num_experts},), got {tuple(w13_scale_2.shape)}."
        )

    if w2_scale_2.numel() != num_experts:
        raise ValueError(
            f"Expected w2_scale_2 with {num_experts} elements, "
            f"got {tuple(w2_scale_2.shape)}."
        )
    gs2 = (w2_scale_2.reshape(num_experts).float() * GS_FIX).contiguous()

    if not (
        torch.isfinite(gs13_gate).all()
        and torch.isfinite(gs13_up).all()
        and torch.isfinite(gs2).all()
    ):
        raise ValueError("Non-finite NVFP4 global scale encountered.")

    device = w13.device
    w13_packed = torch.empty(
        (num_experts, 2 * INTERMEDIATE_DIM // 256, 64, PACKED_UNIT_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    w2_packed = torch.empty(
        (num_experts, HIDDEN_DIM // 256, 64, PACKED_UNIT_BYTES),
        dtype=torch.uint8,
        device=device,
    )

    ext = (
        decode_ext
        if decode_ext is not None
        else (_extensions[0] if _extensions is not None else None)
    )
    if (
        ext is not None
        and hasattr(ext, "pack_experts")
        and w13.is_cuda
        and w2.is_cuda
        and w13_scale.is_cuda
        and w2_scale.is_cuda
        and w13.device == w13_scale.device == w2.device == w2_scale.device
        and w13_scale.dtype in _CUDA_PACK_SCALE_DTYPES
        and w2_scale.dtype in _CUDA_PACK_SCALE_DTYPES
    ):
        w13_c = w13.contiguous()
        w2_c = w2.contiguous()
        w13_sf_c = w13_scale.contiguous()
        w2_sf_c = w2_scale.contiguous()
        if (
            w13_c.data_ptr() % 8 == 0
            and w2_c.data_ptr() % 8 == 0
            and w13_packed.data_ptr() % 16 == 0
            and w2_packed.data_ptr() % 16 == 0
        ):
            err_flag = torch.zeros((1,), dtype=torch.int32, device=device)
            ext.pack_experts(w13_c, w13_sf_c, True, w13_packed, err_flag)
            ext.pack_experts(w2_c, w2_sf_c, False, w2_packed, err_flag)
            if int(err_flag.item()) != 0:
                raise ValueError("Block scales are not exactly representable in E5M3.")
            return w13_packed, w2_packed, gs13_gate, gs13_up, gs2

    lut = _fp4_pack_lut_cpu().to(device)
    perm = w13_row_permutation(device)
    for expert_idx in range(num_experts):
        w13_packed[expert_idx] = encode_rows(
            w13[expert_idx], w13_scale[expert_idx], perm, _lut=lut
        )
        w2_packed[expert_idx] = encode_rows(
            w2[expert_idx], w2_scale[expert_idx], None, _lut=lut
        )

    return w13_packed, w2_packed, gs13_gate, gs13_up, gs2


def build_static_routing_plan(
    expert_map: torch.Tensor | None,
    ep_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build static decode `plan` and prefill `remap` tables on `device`.

    Computes both tables on `device` with zero host-device synchronization.

    Args:
        expert_map: Optional global-to-local expert map (`-1` for off-rank).
            When `None`, maps `[ep_rank * 32, ep_rank * 32 + 32)` to `0..31`.
        ep_rank: Expert-parallel rank used when `expert_map` is `None`.
        device: Target tensor device.

    Returns:
        Tuple `(plan, remap)` where `plan[:256]` maps global expert IDs to local
        slot `0..31` (or `255` off-rank), `plan[256:288]` is `0..31`, and
        `remap` maps global expert IDs to local slot `0..31` (or `-1` off-rank).

    """
    plan = torch.full((PLAN_BYTES,), _OFF_RANK_PLAN, dtype=torch.uint8, device=device)
    remap = torch.full((MAX_GLOBAL_EXPERTS,), -1, dtype=torch.int32, device=device)
    local_slots = torch.arange(NUM_LOCAL_EXPERTS, dtype=torch.uint8, device=device)
    plan[MAX_GLOBAL_EXPERTS:] = local_slots

    if expert_map is None:
        start = ep_rank * NUM_LOCAL_EXPERTS
        end = min(start + NUM_LOCAL_EXPERTS, MAX_GLOBAL_EXPERTS)
        count = max(0, end - start)
        if count > 0:
            plan[start:end] = local_slots[:count]
            remap[start:end] = torch.arange(count, dtype=torch.int32, device=device)
        return plan, remap

    emap = expert_map.to(device=device, dtype=torch.int32).reshape(-1)
    num_global = min(emap.numel(), MAX_GLOBAL_EXPERTS)
    if num_global > 0:
        head = emap[:num_global]
        valid = (head >= 0) & (head < NUM_LOCAL_EXPERTS)
        plan[:num_global] = torch.where(
            valid,
            head.clamp(0, NUM_LOCAL_EXPERTS - 1).to(torch.uint8),
            plan[:num_global],
        )
        remap[:num_global] = torch.where(valid, head, remap[:num_global])
    return plan, remap


def _detect_contiguous_local_start(
    expert_map: torch.Tensor | None,
    ep_rank: int,
) -> int:
    """Return `start` if `[start, start+32)` maps identically to `0..31`, else -1."""
    if expert_map is None:
        start = ep_rank * NUM_LOCAL_EXPERTS
        if start >= 0 and start + NUM_LOCAL_EXPERTS <= MAX_GLOBAL_EXPERTS:
            return start
        return -1
    if expert_map.is_meta or _is_capturing(expert_map.device):
        return -1
    vals = expert_map.detach().reshape(-1).to(device="cpu", dtype=torch.int64).tolist()
    if len(vals) > MAX_GLOBAL_EXPERTS:
        return -1
    start_idx: int | None = None
    for idx, val in enumerate(vals):
        if val == -1:
            continue
        if start_idx is None:
            if val != 0 or idx + NUM_LOCAL_EXPERTS > len(vals):
                return -1
            start_idx = idx
        if idx < start_idx + NUM_LOCAL_EXPERTS:
            if val != idx - start_idx:
                return -1
        else:
            return -1
    if start_idx is None:
        return -1
    if vals[start_idx : start_idx + NUM_LOCAL_EXPERTS] != list(
        range(NUM_LOCAL_EXPERTS)
    ):
        return -1
    return start_idx


def _is_post_norm_meta(post_expert_finalize: Any) -> bool:
    """Return whether `post_expert_finalize`'s norm weight still resides on meta."""
    norm = getattr(post_expert_finalize, "_norm", None)
    return (
        norm is not None
        and getattr(norm, "has_weight", False)
        and getattr(norm, "weight", None) is not None
        and norm.weight.is_meta
    )


def _compute_post_norm_params(
    post_expert_finalize: Any,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Extract owned FP32 effective post-expert norm weight and epsilon."""
    norm = post_expert_finalize._norm
    if norm.has_weight:
        assert norm.weight is not None
        raw = norm.weight.detach()
        weight = raw.to(device=device, dtype=torch.float32)
        gain_center = float(norm.fused_gain_center)
        if gain_center != 0.0:
            weight = weight + gain_center
        weight = weight.contiguous()
        if weight.data_ptr() == raw.data_ptr():
            weight = weight.clone()
    else:
        weight = torch.ones((HIDDEN_DIM,), dtype=torch.float32, device=device)
    return weight, float(norm.variance_epsilon)


@dataclass
class _DeviceState:
    """Persistent per-device Hopper NVFP4 MoE control and scratch buffers."""

    num_sms: int
    slot_of: torch.Tensor
    ctrl: torch.Tensor
    part: torch.Tensor


_device_states: dict[int, _DeviceState] = {}
_configured_devices: set[int] = set()
_configured_ext_devices: set[tuple[int, int]] = set()
_device_states_lock = threading.Lock()


def _is_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _configure_device_extensions(
    dev_idx: int,
    device: torch.device,
    decode_ext: Any | None,
    prefill_ext: Any | None,
) -> None:
    """Eagerly configure per-device extension state outside CUDA graph capture."""
    if (
        (decode_ext is None and prefill_ext is None)
        or dev_idx in _configured_devices
        or _is_capturing(device)
    ):
        return
    with _device_states_lock:
        if dev_idx in _configured_devices:
            return
        for ext in (decode_ext, prefill_ext):
            if ext is None or not hasattr(ext, "configure_device"):
                continue
            key = (dev_idx, id(ext))
            if key not in _configured_ext_devices:
                ext.configure_device(dev_idx)
                _configured_ext_devices.add(key)
        if decode_ext is not None and prefill_ext is not None:
            _configured_devices.add(dev_idx)


def _get_device_state(
    device: torch.device,
    decode_ext: Any | None = None,
    prefill_ext: Any | None = None,
) -> _DeviceState:
    """Return cached per-device control/scratch buffers and configure extensions."""
    dev_idx = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    state = _device_states.get(dev_idx)
    if state is None:
        with _device_states_lock:
            state = _device_states.get(dev_idx)
            if state is None:
                num_sms = int(
                    torch.cuda.get_device_properties(dev_idx).multi_processor_count
                )
                state = _DeviceState(
                    num_sms=num_sms,
                    slot_of=torch.empty(
                        (MAX_DECODE_ROWS,), dtype=torch.int32, device=device
                    ),
                    ctrl=torch.zeros((_CTRL_INTS,), dtype=torch.int32, device=device),
                    part=torch.empty(
                        (num_sms, _PART_COLS), dtype=torch.float32, device=device
                    ),
                )
                if not _is_capturing(device):
                    _device_states[dev_idx] = state
    _configure_device_extensions(dev_idx, device, decode_ext, prefill_ext)
    return state


def _tensors_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True if two contiguous tensors on the same device overlap."""
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False
    a_start = a.data_ptr()
    a_end = a_start + a.numel() * a.element_size()
    b_start = b.data_ptr()
    b_end = b_start + b.numel() * b.element_size()
    return a_start < b_end and b_start < a_end


class MuseSparkHopperNvFp4Experts(mk.FusedMoEExpertsModular):
    """Hopper (SM90a) NVFP4 fused MoE experts for Muse Spark.

    Fuses W13 + SwiGLU + W2 + weight-first post-expert RMSNorm + FP32 weighted
    top-k reduction into a single persistent decode/verify kernel (`T <= 1536`)
    and a chunked prefill pipeline (`T >= 1537`). Per-layer constants are kept
    on the parent layer via `keep_constants` / `refill_derived` so layerwise
    reloads preserve tensor addresses for captured CUDA graphs.
    """

    kept_constants: tuple[str, ...] = (
        "_gs13_gate",
        "_gs13_up",
        "_gs2",
        "_plan",
        "_remap",
        "_post_weight",
    )

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        *,
        gs13_gate: torch.Tensor | None = None,
        gs13_up: torch.Tensor | None = None,
        gs2: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
        post_expert_finalize: "MusePostExpertFinalize | None" = None,
        decode_ext: Any | None = None,
        prefill_ext: Any | None = None,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ) -> None:
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self._decode_ext = decode_ext
        self._prefill_ext = prefill_ext

        # Global scales passed via quant_config (or explicit kwargs) must carry
        # the GS_FIX = 128.0 exponent bias expected by decode_scale_pair.
        if gs13_gate is None or gs13_up is None or gs2 is None:
            from vllm.models.muse_spark.nvidia.muse_moe_runner import (
                MuseSparkNvFp4MoEQuantConfig,
            )

            if not isinstance(quant_config, MuseSparkNvFp4MoEQuantConfig):
                raise TypeError(
                    "MuseSparkHopperNvFp4Experts requires "
                    "MuseSparkNvFp4MoEQuantConfig with GS_FIX-scaled "
                    "g1_alphas, g1_scale_c, and g2_alphas (or explicit "
                    "gs13_gate, gs13_up, and gs2 tensors)."
                )
            if gs13_gate is None:
                gs13_gate = quant_config.g1_alphas
            if gs13_up is None:
                gs13_up = quant_config.g1_scale_c
            if gs2 is None:
                gs2 = quant_config.g2_alphas
            if gs13_gate is None or gs13_up is None or gs2 is None:
                raise ValueError(
                    "MuseSparkHopperNvFp4Experts requires non-None "
                    "gs13_gate, gs13_up, and gs2 global scales."
                )

        self._gs13_gate: torch.Tensor | None = gs13_gate.float().contiguous()
        self._gs13_up: torch.Tensor | None = gs13_up.float().contiguous()
        self._gs2: torch.Tensor | None = gs2.float().contiguous()

        device = self._infer_device()
        self._plan_expert_map: torch.Tensor | None = expert_map
        self._plan: torch.Tensor | None = None
        self._remap: torch.Tensor | None = None
        self._local_start: int = _detect_contiguous_local_start(
            expert_map, self.ep_rank
        )
        if device.type != "meta":
            self._plan, self._remap = build_static_routing_plan(
                expert_map, self.ep_rank, device
            )
            if device.type == "cuda":
                _get_device_state(
                    device, decode_ext=decode_ext, prefill_ext=prefill_ext
                )

        self._post_weight: torch.Tensor | None = None
        self._post_eps: float = _DEFAULT_POST_EPS
        if post_expert_finalize is not None:
            self.post_gemm2_hook = post_expert_finalize
            if device.type != "meta" and not _is_post_norm_meta(post_expert_finalize):
                self._post_weight, self._post_eps = _compute_post_norm_params(
                    post_expert_finalize, device
                )

    def _infer_device(self) -> torch.device:
        for tensor in (self._gs13_gate, self._gs2):
            if tensor is not None:
                return tensor.device
        return torch.device(self.moe_config.device)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(90)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def executed_activation_key(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> QuantKey | None:
        return None

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kNvfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            moe_parallel_config.tp_size == 1
            and moe_parallel_config.dp_size == 1
            and moe_parallel_config.pcp_size == 1
            and not moe_parallel_config.is_sequence_parallel
            and not moe_parallel_config.enable_eplb
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.use_passthrough_all2all
        )

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim == HIDDEN_DIM

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert a1.dim() == 2, f"Expected 2D hidden_states, got {a1.dim()}D."
        assert topk_ids.dim() == 2, f"Expected 2D topk_ids, got {topk_ids.dim()}D."
        assert topk_ids.size(0) == a1.size(0), (
            f"Token count mismatch: {topk_ids.size(0)} != {a1.size(0)}."
        )
        assert w1.shape == (NUM_LOCAL_EXPERTS, 32, 64, PACKED_UNIT_BYTES), (
            f"Unexpected packed w1 shape: {tuple(w1.shape)}."
        )
        assert w2.shape == (NUM_LOCAL_EXPERTS, 16, 64, PACKED_UNIT_BYTES), (
            f"Unexpected packed w2 shape: {tuple(w2.shape)}."
        )
        return (NUM_LOCAL_EXPERTS, a1.size(0), INTERMEDIATE_DIM, HIDDEN_DIM, TOP_K)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        m_eff = min(M, MAX_PREFILL_CHUNK)
        if m_eff <= DECODE_MAX_TOKENS:
            rows = max(m_eff, 1) * TOP_K
        else:
            rows = m_eff * TOP_K + NUM_LOCAL_EXPERTS * 64
        ws = (rows, HIDDEN_DIM)
        return (ws, ws, (M, HIDDEN_DIM))

    def refresh_post_norm(self, layer: torch.nn.Module) -> None:
        """Refresh the fused FP32 post-expert norm weight in place on `layer`.

        Preserves any existing `layer._experts_post_weight` address when the
        norm weight is still on `meta` during a layerwise reload.
        """
        if self.post_gemm2_hook is None:
            return
        if _is_post_norm_meta(self.post_gemm2_hook):
            existing = getattr(layer, "_experts_post_weight", None)
            if self._post_weight is None and isinstance(existing, torch.Tensor):
                self._post_weight = existing
            return
        w13 = getattr(layer, "w13_weight", None)
        device = w13.device if isinstance(w13, torch.Tensor) else self._infer_device()
        if device.type == "meta":
            return
        post_weight, post_eps = _compute_post_norm_params(self.post_gemm2_hook, device)
        self._post_weight = refill_derived(layer, "_experts_post_weight", post_weight)
        self._post_eps = post_eps

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Refresh per-layer derived tensors in place for CUDA-graph reloads."""
        w13 = getattr(layer, "w13_weight", None)
        device = w13.device if isinstance(w13, torch.Tensor) else self._infer_device()

        if (
            self._gs13_gate is not None
            and getattr(layer, "_experts_gs13_gate", None) is not self._gs13_gate
        ):
            self._gs13_gate = refill_derived(
                layer, "_experts_gs13_gate", self._gs13_gate
            )
        if (
            self._gs13_up is not None
            and getattr(layer, "_experts_gs13_up", None) is not self._gs13_up
        ):
            self._gs13_up = refill_derived(layer, "_experts_gs13_up", self._gs13_up)
        if (
            self._gs2 is not None
            and getattr(layer, "_experts_gs2", None) is not self._gs2
        ):
            self._gs2 = refill_derived(layer, "_experts_gs2", self._gs2)

        expert_map = getattr(layer, "expert_map", None)
        plan, remap = build_static_routing_plan(expert_map, self.ep_rank, device)
        self._plan = refill_derived(layer, "_experts_plan", plan)
        self._remap = refill_derived(layer, "_experts_remap", remap)
        self._plan_expert_map = expert_map
        self._local_start = _detect_contiguous_local_start(expert_map, self.ep_rank)

        self.refresh_post_norm(layer)

        if device.type == "cuda":
            decode_ext, prefill_ext = self._get_extensions()
            _get_device_state(
                device,
                decode_ext=decode_ext,
                prefill_ext=prefill_ext,
            )

    def _get_extensions(self) -> tuple[Any, Any]:
        if self._decode_ext is None or self._prefill_ext is None:
            exts = build_hopper_nvfp4_moe()
            if exts is None:
                raise RuntimeError(
                    "Hopper NVFP4 MoE CUDA extensions are unavailable; "
                    "see earlier warning."
                )
            self._decode_ext, self._prefill_ext = exts
        return self._decode_ext, self._prefill_ext

    def _resolve_routing_plan(
        self,
        expert_map: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if (
            self._plan is not None
            and self._remap is not None
            and expert_map is self._plan_expert_map
        ):
            return self._plan, self._remap, self._local_start

        plan, remap = build_static_routing_plan(expert_map, self.ep_rank, device)
        local_start = _detect_contiguous_local_start(expert_map, self.ep_rank)
        if not _is_capturing(device):
            if (
                self._plan is not None
                and self._plan.device == device
                and self._remap is not None
                and self._remap.device == device
            ):
                self._plan.copy_(plan)
                self._remap.copy_(remap)
                self._plan_expert_map = expert_map
                self._local_start = local_start
                return self._plan, self._remap, self._local_start
            self._plan = plan
            self._remap = remap
            self._plan_expert_map = expert_map
            self._local_start = local_start
        return plan, remap, local_start

    def _resolve_post_norm(self, device: torch.device) -> tuple[torch.Tensor, float]:
        if self._post_weight is not None:
            return self._post_weight, self._post_eps
        if self.post_gemm2_hook is None:
            raise RuntimeError(
                "MuseSparkHopperNvFp4Experts requires an attached post_gemm2_hook."
            )
        post_weight, post_eps = _compute_post_norm_params(self.post_gemm2_hook, device)
        if not _is_capturing(device):
            self._post_weight = post_weight
            self._post_eps = post_eps
        return post_weight, post_eps

    def _run_chunk(
        self,
        decode_ext: Any,
        prefill_ext: Any,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gs13_gate: torch.Tensor,
        gs13_up: torch.Tensor,
        gs2: torch.Tensor,
        x_chunk: torch.Tensor,
        ids_chunk: torch.Tensor,
        weights_chunk: torch.Tensor,
        post_weight: torch.Tensor,
        post_eps: float,
        out_chunk: torch.Tensor,
        out_full: torch.Tensor,
        ws1_2d: torch.Tensor,
        ws2_2d: torch.Tensor,
        plan: torch.Tensor,
        remap: torch.Tensor,
        local_start: int,
        state: _DeviceState,
    ) -> None:
        num_tokens = x_chunk.shape[0]
        device = x_chunk.device
        grid = state.num_sms

        if num_tokens <= DECODE_MAX_TOKENS:
            rows_needed = num_tokens * TOP_K
            act = ws1_2d[:rows_needed]
            y = ws2_2d[:rows_needed]
            # FusedMoEKernel.apply aliases output to a fresh allocation; guard
            # against direct _allocate_buffers callers without output_alias.
            if act.shape[0] < rows_needed or _tensors_overlap(act, out_full):
                act = torch.empty(
                    (rows_needed, HIDDEN_DIM), dtype=x_chunk.dtype, device=device
                )
            if (
                y.shape[0] < rows_needed
                or _tensors_overlap(y, out_full)
                or _tensors_overlap(y, act)
            ):
                y = torch.empty(
                    (rows_needed, HIDDEN_DIM), dtype=x_chunk.dtype, device=device
                )

            decode_ext.run(
                w1,
                w2,
                gs13_gate,
                gs13_up,
                gs2,
                x_chunk,
                ids_chunk,
                weights_chunk,
                post_weight,
                post_eps,
                act,
                y,
                state.slot_of,
                state.ctrl,
                out_chunk,
                grid,
                plan,
                local_start,
            )
            return

        if ids_chunk.data_ptr() % 16 != 0:
            ids_chunk = ids_chunk.clone()

        n_routes = num_tokens * TOP_K
        rp = n_routes + NUM_LOCAL_EXPERTS * 64
        np_tiles = (rp + 255) // 256 + 2 * NUM_LOCAL_EXPERTS
        sizes = (rp, n_routes, 4 * np_tiles, 4 * (grid + 1), 2 * grid + 96)
        offs: list[int] = []
        total = 0
        for sz in sizes:
            offs.append(total)
            total += ((sz + 63) // 64) * 64
        route_ws = torch.empty((total,), dtype=torch.int32, device=device)
        rows, rpos, pinfo, seg, flags = (
            route_ws[off : off + sz] for off, sz in zip(offs, sizes)
        )

        in2 = ws1_2d[:rp]
        y = ws2_2d[:rp]
        if in2.shape[0] < rp or _tensors_overlap(in2, out_full):
            in2 = torch.empty((rp, HIDDEN_DIM), dtype=x_chunk.dtype, device=device)
        if y.shape[0] < rp or _tensors_overlap(y, out_full) or _tensors_overlap(y, in2):
            y = torch.empty((rp, HIDDEN_DIM), dtype=x_chunk.dtype, device=device)

        prefill_ext.route(
            ids_chunk,
            0,
            rows,
            rpos,
            pinfo,
            seg,
            flags,
            grid,
            _PREFILL_COST,
            NUM_LOCAL_EXPERTS,
            remap,
        )
        prefill_ext.gemm(
            0,
            w1,
            gs13_gate,
            gs13_up,
            x_chunk,
            rows,
            pinfo,
            seg,
            state.part,
            flags,
            in2,
            grid,
        )
        prefill_ext.gemm(
            1,
            w2,
            gs2,
            None,
            in2,
            rows,
            pinfo,
            seg,
            state.part,
            flags,
            y,
            grid,
        )
        prefill_ext.combine(
            y,
            rpos,
            weights_chunk,
            post_weight,
            post_eps,
            out_chunk,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self.post_gemm2_hook is not None and hasattr(
            self.post_gemm2_hook, "output_reduced"
        ):
            self.post_gemm2_hook.output_reduced = False  # type: ignore[attr-defined]

        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            return

        assert not apply_router_weight_on_input, (
            "MuseSparkHopperNvFp4Experts does not support apply_router_weight_on_input."
        )
        assert activation == MoEActivation.SILU, (
            "MuseSparkHopperNvFp4Experts requires MoEActivation.SILU, "
            f"got {activation}."
        )
        assert (
            self._gs13_gate is not None
            and self._gs13_up is not None
            and self._gs2 is not None
        ), "Hopper NVFP4 MoE global scales must be initialized before apply()."

        decode_ext, prefill_ext = self._get_extensions()
        device = hidden_states.device
        state = _get_device_state(
            device, decode_ext=decode_ext, prefill_ext=prefill_ext
        )
        post_weight, post_eps = self._resolve_post_norm(device)
        plan, remap, local_start = self._resolve_routing_plan(expert_map, device)

        x = (
            hidden_states
            if hidden_states.is_contiguous()
            else hidden_states.contiguous()
        )
        if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
            ids = topk_ids
        else:
            ids = topk_ids.to(dtype=torch.int32).contiguous()

        if topk_weights.dtype == torch.float32 and topk_weights.is_contiguous():
            weights = topk_weights
        else:
            weights = topk_weights.to(dtype=torch.float32).contiguous()

        out_contiguous = output.is_contiguous()
        out_buf = (
            output
            if out_contiguous
            else torch.empty(output.shape, dtype=output.dtype, device=output.device)
        )
        ws1_2d = workspace13.reshape(-1, HIDDEN_DIM)
        ws2_2d = workspace2.reshape(-1, HIDDEN_DIM)

        for start in range(0, num_tokens, MAX_PREFILL_CHUNK):
            end = min(start + MAX_PREFILL_CHUNK, num_tokens)
            full_slice = start == 0 and end == num_tokens
            self._run_chunk(
                decode_ext,
                prefill_ext,
                w1,
                w2,
                self._gs13_gate,
                self._gs13_up,
                self._gs2,
                x if full_slice else x[start:end],
                ids if full_slice else ids[start:end],
                weights if full_slice else weights[start:end],
                post_weight,
                post_eps,
                out_buf if full_slice else out_buf[start:end],
                out_buf,
                ws1_2d,
                ws2_2d,
                plan,
                remap,
                local_start,
                state,
            )

        if not out_contiguous:
            output.copy_(out_buf)


def _all_hopper_nvfp4_ranks_ok(ok: bool) -> bool:
    """Agree across EP ranks in the current PP stage on Hopper NVFP4 use."""
    if not (dist.is_available() and dist.is_initialized()):
        return ok
    ep_group = get_ep_group()
    if ep_group.world_size <= 1 or ep_group.cpu_group is None:
        return ok
    flag = torch.tensor(int(ok), dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group.cpu_group)
    return bool(flag.item())


def _make_hopper_quant_config(
    gs13_gate: torch.Tensor | None,
    gs13_up: torch.Tensor | None,
    gs2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FusedMoEQuantConfig:
    from vllm.models.muse_spark.nvidia.muse_moe_runner import (
        MuseSparkNvFp4MoEQuantConfig,
    )

    assert gs13_gate is not None and gs2 is not None
    base_config = nvfp4_w4a16_moe_quant_config(
        g1_alphas=gs13_gate,
        g2_alphas=gs2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    return MuseSparkNvFp4MoEQuantConfig(**vars(base_config), g1_scale_c=gs13_up)


def _replace_packed_weights(
    layer: torch.nn.Module,
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
) -> None:
    replace_parameter(layer, "w13_weight", w13_packed)
    replace_parameter(layer, "w2_weight", w2_packed)
    # Keep each block scale in its own storage: the direct-reload load plan
    # labels writes into a shared storage with its first checkpoint tensor
    # (reload/bake.py `Targets.by_storage`), so a scale viewed into the packed
    # weight would be reported as that weight. Under direct reload this keeps
    # the checkpoint scale bytes (96 MiB/layer) for the next reload.
    for name in ("w13_weight_scale", "w2_weight_scale"):
        scale = cast(torch.Tensor, getattr(layer, name))
        replace_parameter(
            layer, name, torch.empty(0, dtype=scale.dtype, device=scale.device)
        )


def try_process_hopper_nvfp4_weights(
    quant_method: "MuseSparkModelOptNvFp4FusedMoE",
    layer: "MuseSparkRoutedExperts",
) -> bool:
    """Pack and bind Hopper NVFP4 MoE weights on `layer`, or return False.

    Handles cold start (with EP/TP-group agreement and clean fallback to Marlin
    on JIT/encoding failure or OOM), idempotent repeat calls on already-packed
    weights, and in-place layerwise/direct weight reloads without collectives
    or shape-breaking fallback.

    Args:
        quant_method: `MuseSparkModelOptNvFp4FusedMoE` instance for `layer`.
        layer: `MuseSparkRoutedExperts` module holding the MoE weights.

    Returns:
        True if the layer is bound to `MuseSparkHopperNvFp4Experts`, or False
        on cold start to fall back to Hopper Marlin.

    Raises:
        RuntimeError: If weights are already packed without a bound Hopper
            NVFP4 kernel, or if reloading an already-bound layer fails.

    """
    existing_experts = (
        quant_method.moe_kernel.fused_experts
        if quant_method.moe_kernel is not None
        else None
    )
    bound_experts = (
        existing_experts
        if isinstance(existing_experts, MuseSparkHopperNvFp4Experts)
        else None
    )
    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale)
    w13_weight_scale_2 = cast(torch.Tensor, layer.w13_weight_scale_2)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale)
    w2_weight_scale_2 = cast(torch.Tensor, layer.w2_weight_scale_2)

    if w13_weight.ndim == 4 and w13_weight_scale.numel() == 0:
        if bound_experts is not None:
            bound_experts.refresh_post_norm(layer)
            return True
        raise RuntimeError(
            "MuseSpark NVFP4 layer already holds packed Hopper NVFP4 weights "
            "without a bound MuseSparkHopperNvFp4Experts kernel; cannot fall "
            "back to Marlin."
        )

    if bound_experts is not None:
        decode_ext, _ = bound_experts._get_extensions()
        w13_packed, w2_packed, gs13_gate, gs13_up, gs2 = encode_hopper_nvfp4_weights(
            w13_weight.data,
            w13_weight_scale.data,
            w13_weight_scale_2.data,
            w2_weight.data,
            w2_weight_scale.data,
            w2_weight_scale_2.data,
            decode_ext=decode_ext,
        )
        _replace_packed_weights(layer, w13_packed, w2_packed)
        bound_experts._gs13_gate = gs13_gate.float().contiguous()
        bound_experts._gs13_up = gs13_up.float().contiguous()
        bound_experts._gs2 = gs2.float().contiguous()
        if bound_experts.post_gemm2_hook is None:
            bound_experts.post_gemm2_hook = getattr(
                quant_method, "muse_post_expert_finalize", None
            )
        bound_experts.process_weights_after_loading(layer)
        quant_config = _make_hopper_quant_config(
            bound_experts._gs13_gate,
            bound_experts._gs13_up,
            bound_experts._gs2,
            cast(torch.Tensor, layer.w13_weight_scale),
            cast(torch.Tensor, layer.w2_weight_scale),
        )
        quant_method.moe_quant_config = quant_config
        bound_experts.quant_config = quant_config
        return True

    if existing_experts is not None:
        # Bound to another kernel (Marlin) at cold start: keep it, and the
        # graphs captured with it, on reload, without a collective.
        return False

    supported = is_hopper_nvfp4_moe_supported(
        quant_method.moe, getattr(quant_method, "muse_post_expert_finalize", None)
    )
    exts = build_hopper_nvfp4_moe() if supported else None
    encoded: _EncodedWeights | None = None
    if exts is not None:
        try:
            encoded = encode_hopper_nvfp4_weights(
                w13_weight.data,
                w13_weight_scale.data,
                w13_weight_scale_2.data,
                w2_weight.data,
                w2_weight_scale.data,
                w2_weight_scale_2.data,
                decode_ext=exts[0],
            )
        except (ValueError, RuntimeError) as err:
            if torch.cuda.is_available():
                torch.accelerator.empty_cache()
            # warning_once caches its args; pass str so tracebacks/packed
            # buffers are not pinned in the LRU cache.
            logger.warning_once(
                "MuseSpark NVFP4: Hopper NVFP4 weight encoding failed (%s); "
                "falling back to Marlin.",
                str(err),
            )
    if not _all_hopper_nvfp4_ranks_ok(encoded is not None):
        return False

    assert exts is not None and encoded is not None
    w13_packed, w2_packed, gs13_gate, gs13_up, gs2 = encoded
    _replace_packed_weights(layer, w13_packed, w2_packed)

    initial_quant_config = _make_hopper_quant_config(
        gs13_gate,
        gs13_up,
        gs2,
        cast(torch.Tensor, layer.w13_weight_scale),
        cast(torch.Tensor, layer.w2_weight_scale),
    )
    quant_method.experts_cls = MuseSparkHopperNvFp4Experts
    quant_method.moe_kernel = make_nvfp4_moe_kernel(
        moe_quant_config=initial_quant_config,
        moe_config=quant_method.moe,
        experts_cls=MuseSparkHopperNvFp4Experts,
        backend=NvFp4MoeBackend.MARLIN,
        routing_tables=layer._expert_routing_tables(),
    )
    experts = cast(MuseSparkHopperNvFp4Experts, quant_method.moe_kernel.fused_experts)
    experts.post_gemm2_hook = getattr(quant_method, "muse_post_expert_finalize", None)
    experts.keep_constants(layer)
    experts.process_weights_after_loading(layer)

    final_quant_config = _make_hopper_quant_config(
        experts._gs13_gate,
        experts._gs13_up,
        experts._gs2,
        cast(torch.Tensor, layer.w13_weight_scale),
        cast(torch.Tensor, layer.w2_weight_scale),
    )
    quant_method.moe_quant_config = final_quant_config
    experts.quant_config = final_quant_config

    logger.info_once("MuseSpark NVFP4: Hopper SM90a fused NVFP4 MoE kernels.")
    return True
