# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accessors for the optional ``b12x`` package."""

import importlib
import importlib.util
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, fields, is_dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.quantization import PhaseSpec

logger = init_logger(__name__)

# Logical elements and encoded bytes per block.
B12X_BLOCK_CODECS = {"iq2_xs": (256, 74), "iq2_xxs": (256, 66), "q8_0": (32, 34)}


@dataclass(frozen=True)
class B12xWarmupUnit:
    name: str
    key: Hashable
    compile: Callable[[], None]


_HAS_B12X = importlib.util.find_spec("b12x") is not None


def _import_submodule(module_name: str) -> ModuleType | None:
    if not _HAS_B12X:
        return None
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


_B12X_SUBMODULES = {
    module_name: _import_submodule(module_name)
    for module_name in (
        "b12x.attention.paged",
        "b12x.gemm.blockscaled",
        # TODO: Remove once B12X exposes the scale-swizzle API publicly.
        "b12x._lib.intrinsics",
        "b12x.gemm.mxfp8_linear",
        "b12x.gemm.tensor_fp8_linear",
        "b12x.moe.fused_moe",
    )
}


def has_b12x() -> bool:
    """Return whether the B12X package is installed."""
    return _HAS_B12X


def _get_submodule(module_name: str) -> ModuleType | None:
    return _B12X_SUBMODULES.get(module_name)


def get_b12x_blockscaled() -> ModuleType | None:
    return _get_submodule("b12x.gemm.blockscaled")


def get_b12x_intrinsics() -> ModuleType | None:
    return _get_submodule("b12x._lib.intrinsics")


def get_b12x_mxfp8_linear() -> ModuleType | None:
    return _get_submodule("b12x.gemm.mxfp8_linear")


def get_b12x_tensor_fp8_linear() -> ModuleType | None:
    return _get_submodule("b12x.gemm.tensor_fp8_linear")


def get_b12x_fused_moe() -> ModuleType | None:
    return _get_submodule("b12x.moe.fused_moe")


def get_b12x_paged_attention() -> ModuleType | None:
    return _get_submodule("b12x.attention.paged")


def b12x_warmup_token_counts(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
) -> tuple[int, ...]:
    # B12X deduplicates shapes that select the same internal kernel policy.
    # Keep the complete serving shape set here rather than duplicating its
    # policy-selection heuristics in vLLM.
    counts = {1}
    counts.update(int(size) for size in cudagraph_capture_sizes if int(size) > 0)
    if int(max_tokens) > 0:
        counts.add(int(max_tokens))
    return tuple(sorted(counts))


def _same_packed_layout(current: Any, replacement: Any) -> bool:
    if type(current) is not type(replacement):
        return False
    if isinstance(current, torch.Tensor):
        return (
            current.shape == replacement.shape
            and current.stride() == replacement.stride()
            and current.dtype == replacement.dtype
            and current.device == replacement.device
        )
    if is_dataclass(current):
        return all(
            _same_packed_layout(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )
            for field in fields(current)
        )
    return bool(current == replacement)


def _copy_packed_tensors(current: Any, replacement: Any) -> None:
    if isinstance(current, torch.Tensor):
        current.copy_(replacement)
    elif is_dataclass(current):
        for field in fields(current):
            _copy_packed_tensors(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )


@torch.no_grad()
def reuse_packed_weight_storage(current: Any, replacement: Any) -> Any:
    """Reuse packed tensor addresses when a compatible weight is reloaded."""
    if current is None or not _same_packed_layout(current, replacement):
        return replacement
    _copy_packed_tensors(current, replacement)
    return current


@dataclass(frozen=True)
class B12xLinearPlans:
    """Prepared b12x plans for one packed linear weight.

    Calls with at most ``cutoff`` rows, or every call when it is None, are
    decode calls; ``quantized`` says whether decode and prefill calls quantize
    their activations.
    """

    query: Any
    quantized: tuple[bool, bool]
    cutoff: int | None
    plan: Any
    # plan_regimes only forces A16 up to a row count, so quantized decode calls
    # with A16 prefill calls have their own plan.
    decode_plan: Any = None

    @property
    def plans(self) -> tuple[Any, ...]:
        return (
            (self.plan,) if self.decode_plan is None else (self.plan, self.decode_plan)
        )

    def select(self, rows: int, quantize: bool) -> tuple[Any, bool]:
        """The plan of a call with ``rows`` rows, and whether it quantizes its
        activations when its phase asks to (``quantize``)."""
        decode = self.cutoff is None or rows <= self.cutoff
        quantized = quantize and self.quantized[0 if decode else 1]
        if quantized and self.decode_plan is not None:
            return self.decode_plan, True
        return self.plan, quantized


def plan_b12x_linear(
    blockscaled: ModuleType,
    packed: Any,
    recipe: str,
    phase: "PhaseSpec | None",
    previous: B12xLinearPlans | None = None,
) -> B12xLinearPlans:
    """Declare every serving shape of a packed b12x linear weight, with the
    activation precision ``phase`` gives each call; None keeps BF16
    activations. An undeclared token count is prepared on first use, and that
    preparation evicts kernels compiled earlier, including ones already
    captured in CUDA graphs. ``previous`` is returned when nothing changed, e.g.
    on a weight reload."""
    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config()
    capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
    capacity = max(
        config.scheduler_config.max_num_batched_tokens,
        config.scheduler_config.max_num_scheduled_tokens or 0,
        *capture_sizes,
    )
    decode = prefill = False
    cutoff = None
    if phase is not None:
        decode, prefill = (mode is not None for mode in phase.activation_modes())
        cutoff = phase.max_decode_tokens
        if (decode or prefill) and packed.padded_in_features % 128:
            logger.warning_once(
                "b12x %s keeps BF16 activations for input size %d, which is not "
                "a multiple of 128",
                recipe.upper(),
                packed.padded_in_features,
            )
            decode = prefill = False
        if cutoff is not None and cutoff >= capacity and not decode:
            # Every planned call is a decode call; larger ones run in
            # capacity-sized chunks, so prefill never quantizes.
            if prefill:
                logger.info_once(
                    "max_decode_tokens=%d covers every step of at most %d "
                    "tokens, so prefill activations stay unquantized",
                    cutoff,
                    capacity,
                )
            prefill = False
    if decode == prefill:
        cutoff = None
    common = dict(
        recipe=recipe,
        in_features=packed.in_features,
        padded_in_features=packed.padded_in_features,
        out_features=packed.out_features,
        # The A16 regimes of NVFP4 plans with quantized ones take an unused
        # activation scale.
        activation_scale_available=recipe == "nvfp4" and (decode or prefill),
        global_scale_kind=getattr(packed, "global_scale_kind", None),
    )
    query = blockscaled.BlockscaledQuery(
        num_tokens=capacity,
        activation_mode="quantized" if decode or prefill else "a16",
        **common,
    )
    if (
        previous is not None
        and previous.query == query
        and (previous.quantized, previous.cutoff) == ((decode, prefill), cutoff)
    ):
        return previous
    exact_m = sorted({m for m in capture_sizes if 0 < m < capacity})
    if decode and not prefill:
        assert cutoff is not None
        plan = blockscaled.plan_regimes(
            blockscaled.BlockscaledQuery(
                num_tokens=capacity, activation_mode="a16", **common
            ),
            exact_m=tuple(m for m in exact_m if m > cutoff),
        )
        decode_tokens = min(cutoff, capacity)
        decode_plan = blockscaled.plan_regimes(
            blockscaled.BlockscaledQuery(
                num_tokens=decode_tokens, activation_mode="quantized", **common
            ),
            exact_m=tuple(m for m in exact_m if m < decode_tokens),
        )
        return B12xLinearPlans(query, (decode, prefill), cutoff, plan, decode_plan)
    plan = blockscaled.plan_regimes(
        query, exact_m=tuple(exact_m), a16_max_tokens=cutoff or 0
    )
    return B12xLinearPlans(query, (decode, prefill), cutoff, plan)
