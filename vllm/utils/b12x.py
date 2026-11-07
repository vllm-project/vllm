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
    from vllm.model_executor.kernels.linear.phase_split import PhasePlans
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

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


def b12x_activation_mode(mode: "QuantKey | None") -> str:
    """b12x's name for activations in ``mode``; None is unquantized."""
    return "a16" if mode is None else "quantized"


def plan_b12x_linear(
    blockscaled: ModuleType,
    packed: Any,
    recipe: str,
    phase: "PhaseSpec | None",
    previous: "PhasePlans | None" = None,
) -> "PhasePlans":
    """Declare every serving shape of a packed b12x linear weight, with the
    activations ``phase`` gives each call; None keeps BF16 activations. An
    undeclared token count is prepared on first use, and that preparation
    evicts kernels compiled earlier, including ones already captured in CUDA
    graphs. ``previous`` is returned when nothing changed, e.g. on a weight
    reload."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.kernels.linear.phase_split import PhasePlans

    config = get_current_vllm_config()
    capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
    capacity = max(
        config.scheduler_config.max_num_batched_tokens,
        config.scheduler_config.max_num_scheduled_tokens or 0,
        *capture_sizes,
    )
    decode: QuantKey | None = None
    prefill: QuantKey | None = None
    cutoff = None
    if phase is not None:
        decode, prefill = phase.activation_modes()
        cutoff = phase.max_decode_tokens
        if (decode, prefill) != (None, None) and packed.padded_in_features % 128:
            logger.warning_once(
                "b12x %s keeps BF16 activations for input size %d, which is not "
                "a multiple of 128",
                recipe.upper(),
                packed.padded_in_features,
            )
            decode = prefill = None
        if cutoff is not None and cutoff >= capacity and decode is None:
            # Every planned call is a decode call; larger ones run in
            # capacity-sized chunks, so prefill never quantizes.
            if prefill is not None:
                logger.info_once(
                    "max_decode_tokens=%d covers every step of at most %d "
                    "tokens, so prefill activations stay unquantized",
                    cutoff,
                    capacity,
                )
            prefill = None
    if decode == prefill:
        cutoff = None
    modes = (decode, prefill)
    quantized = decode if decode is not None else prefill
    common = dict(
        recipe=recipe,
        in_features=packed.in_features,
        padded_in_features=packed.padded_in_features,
        out_features=packed.out_features,
        # The A16 regimes of NVFP4 plans with quantized ones take an unused
        # activation scale.
        activation_scale_available=recipe == "nvfp4" and quantized is not None,
        global_scale_kind=getattr(packed, "global_scale_kind", None),
    )
    query = blockscaled.BlockscaledQuery(
        num_tokens=capacity, activation_mode=b12x_activation_mode(quantized), **common
    )
    if (
        previous is not None
        and previous.key == query
        and (previous.modes, previous.cutoff) == (modes, cutoff)
    ):
        return previous
    exact_m = sorted({m for m in capture_sizes if 0 < m < capacity})
    if decode is not None and prefill is None:
        # plan_regimes only forces A16 up to a row count, so quantized decode
        # calls get their own plan.
        assert cutoff is not None
        plan = blockscaled.plan_regimes(
            blockscaled.BlockscaledQuery(
                num_tokens=capacity,
                activation_mode=b12x_activation_mode(None),
                **common,
            ),
            exact_m=tuple(m for m in exact_m if m > cutoff),
        )
        decode_tokens = min(cutoff, capacity)
        decode_plan = blockscaled.plan_regimes(
            blockscaled.BlockscaledQuery(
                num_tokens=decode_tokens,
                activation_mode=b12x_activation_mode(decode),
                **common,
            ),
            exact_m=tuple(m for m in exact_m if m < decode_tokens),
        )
        return PhasePlans(query, capacity, modes, cutoff, plan, decode_plan)
    plan = blockscaled.plan_regimes(
        query, exact_m=tuple(exact_m), a16_max_tokens=cutoff or 0
    )
    return PhasePlans(query, capacity, modes, cutoff, plan)
