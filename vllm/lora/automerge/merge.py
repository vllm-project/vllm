# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 LoRA merge/restore via the vLLM LoRA model manager.

Golden copy approach:
    On first merge, clone each base weight tensor. All subsequent merges
    compute merged = golden + delta and overwrite the base weight.
    Restore simply copies the golden back. No W -= delta, no drift.

TP-awareness:
    LoRA weights in lora_model.loras are stored in their *original* (full)
    shape. We use module.slice_lora_a() / slice_lora_b() to shard them
    for the current TP rank before computing delta = B @ A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class MergeResult:
    ok: bool
    reason: str = ""
    merged_modules: int = 0
    skipped_modules: int = 0


class BF16GoldenCache:
    """Clones of base weights for LoRA target modules.

    Populated lazily on first merge. The golden copies are never modified,
    so restore is a simple copy with zero numerical drift.

    Supports three storage modes:
      - 'gpu': clones stay on the same GPU device (fastest restore)
      - 'cpu': clones are moved to host RAM (saves GPU memory)
      - 'off': no clones stored (subtract-to-restore, zero extra memory)
    """

    def __init__(self, device: str = "cpu") -> None:
        self._cache: dict[str, torch.Tensor] = {}
        self._initialized = False
        self._device = device  # "cpu", "gpu", or "off"

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def is_off(self) -> bool:
        return self._device == "off"

    def ensure_populated(
        self,
        modules: dict[str, Any],
        lora_module_names: set[str],
    ) -> int:
        if self._initialized:
            return len(self._cache)

        if self._device == "off":
            self._initialized = True
            return 0

        cached = 0
        for name in lora_module_names:
            if name in self._cache:
                cached += 1
                continue
            module = modules.get(name)
            if module is None:
                continue
            W = _get_base_weight_tensor(module)
            if W is None:
                continue
            if self._device == "cpu":
                self._cache[name] = W.detach().clone().cpu()
            else:
                # "gpu" — keep on same device
                self._cache[name] = W.clone()
            cached += 1

        self._initialized = True
        total_mb = (
            sum(t.numel() * t.element_size() for t in self._cache.values())
            / 1024
            / 1024
        )
        logger.info(
            "automerge: cached %d golden copies on %s (%.1f MB)",
            cached,
            self._device,
            total_mb,
        )
        return cached

    def get(self, module_name: str) -> torch.Tensor | None:
        return self._cache.get(module_name)

    def clear(self) -> None:
        self._cache.clear()
        self._initialized = False


def _get_base_weight_tensor(module: Any) -> torch.Tensor | None:
    """Extract the base weight tensor from a vLLM LoRA wrapper module."""
    if hasattr(module, "base_layer"):
        module = module.base_layer
    w = getattr(module, "weight", None)
    if w is None:
        return None
    return w if isinstance(w, torch.Tensor) else None


def _get_scaling(lora_layer: Any) -> float:
    """Get the LoRA scaling factor.

    After _create_merged_loras_inplace -> optimize(), the scaling is
    already baked into lora_b, so this typically returns 1.0.
    """
    for name in ("scaling", "scale", "lora_scaling"):
        if hasattr(lora_layer, name):
            v = getattr(lora_layer, name)
            try:
                return float(v)
            except Exception:
                pass
    if hasattr(lora_layer, "lora_alpha") and hasattr(lora_layer, "rank"):
        try:
            alpha = float(lora_layer.lora_alpha)
            r = float(lora_layer.rank)
            if r != 0:
                return alpha / r
        except Exception:
            pass
    return 1.0


def _is_supported_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16, torch.float32)


def _slice_and_compute_delta(
    *,
    module: Any,
    A: torch.Tensor,
    B: torch.Tensor,
    W: torch.Tensor,
    scale: float,
    module_name: str,
) -> torch.Tensor | None:
    """Slice LoRA A/B for TP, compute delta, match to base weight shape."""
    tp_size = getattr(module, "tp_size", 1)
    try:
        if tp_size > 1:
            A_sliced = module.slice_lora_a(A)
            B_sliced = module.slice_lora_b(B)
        else:
            A_sliced, B_sliced = A, B
    except Exception:
        logger.debug("automerge: skip %s — slice failed", module_name, exc_info=True)
        return None

    dev = W.device
    try:
        delta = torch.matmul(B_sliced.to(device=dev), A_sliced.to(device=dev))
        if scale != 1.0:
            delta = delta * scale
    except Exception:
        logger.debug("automerge: skip %s — matmul failed", module_name, exc_info=True)
        return None

    if delta.shape != W.shape:
        logger.debug(
            "automerge: skip %s — shape mismatch delta=%s vs W=%s",
            module_name,
            list(delta.shape),
            list(W.shape),
        )
        return None

    if delta.dtype != W.dtype:
        delta = delta.to(dtype=W.dtype)
    return delta


def merge_lora_into_base(
    *,
    model_manager: Any,
    lora_model: Any,
    golden_cache: BF16GoldenCache,
    validate_dtypes: bool = True,
) -> MergeResult:
    """Merge LoRA weights into base model.

    When golden_cache has copies (cpu/gpu mode):
        W <- golden + scale * (B @ A)
    When golden_cache is off (subtract mode):
        W += scale * (B @ A)  (caller must subtract to restore)
    """
    modules: dict[str, Any] = getattr(model_manager, "modules", {})
    loras: dict[str, Any] = getattr(lora_model, "loras", {})

    if not modules or not loras:
        return MergeResult(ok=False, reason="missing modules/loras dict")

    golden_cache.ensure_populated(modules, set(loras.keys()))

    merged = 0
    skipped = 0

    for module_name, lora_layer in loras.items():
        module = modules.get(module_name)
        if module is None:
            skipped += 1
            continue

        W = _get_base_weight_tensor(module)
        if W is None:
            skipped += 1
            continue

        if validate_dtypes and not _is_supported_dtype(W.dtype):
            skipped += 1
            continue

        golden = golden_cache.get(module_name)

        A = getattr(lora_layer, "lora_a", None)
        B = getattr(lora_layer, "lora_b", None)
        is_packed = getattr(lora_layer, "is_packed", False)

        if is_packed and isinstance(A, list) and isinstance(B, list):
            ok = _merge_packed_layer(
                module=module,
                lora_layer=lora_layer,
                A_list=A,
                B_list=B,
                golden=golden,
                W=W,
            )
            if ok:
                merged += 1
            else:
                skipped += 1
            continue

        # Non-packed
        if not (torch.is_tensor(A) and torch.is_tensor(B)):
            skipped += 1
            continue

        scale = _get_scaling(lora_layer)
        # For golden mode, compute delta against golden; for off mode,
        # compute against current W (delta will be added in-place).
        ref = golden if golden is not None else W
        delta = _slice_and_compute_delta(
            module=module,
            A=A,
            B=B,
            W=W,  # always use W for device/shape reference
            scale=scale,
            module_name=module_name,
        )
        if delta is None:
            skipped += 1
            continue

        with torch.no_grad():
            if golden is not None:
                W.copy_(golden.to(device=W.device) + delta)
            else:
                # "off" mode: add delta in-place
                W.add_(delta)
        merged += 1

    return MergeResult(
        ok=merged > 0 and skipped == 0,
        reason=""
        if merged > 0 and skipped == 0
        else (
            f"partial merge: {merged} merged, {skipped} skipped"
            if merged > 0
            else "no layers merged"
        ),
        merged_modules=merged,
        skipped_modules=skipped,
    )


def _merge_packed_layer(
    *,
    module: Any,
    lora_layer: Any,
    A_list: list,
    B_list: list,
    golden: torch.Tensor | None,
    W: torch.Tensor,
) -> bool:
    """Merge a packed LoRA layer (merged QKV, gate_up, etc.)."""
    scalings = getattr(lora_layer, "scaling", None)
    if not isinstance(scalings, list):
        scalings = [_get_scaling(lora_layer)] * len(A_list)

    tp_size = getattr(module, "tp_size", 1)
    try:
        A_sliced = module.slice_lora_a(A_list) if tp_size > 1 else A_list
        B_sliced = module.slice_lora_b(B_list) if tp_size > 1 else B_list
    except Exception:
        return False

    output_slices = getattr(module, "output_slices", None)
    base = golden.to(device=W.device) if golden is not None else W
    merged_w = base.clone()
    dev = merged_w.device
    sub_ok = 0
    offset = 0

    for i, (a_i, b_i) in enumerate(zip(A_sliced, B_sliced)):
        if a_i is None or b_i is None:
            if output_slices is not None and i < len(output_slices):
                offset += output_slices[i]
            continue
        scale_i = float(scalings[i]) if i < len(scalings) else 1.0
        try:
            delta = torch.matmul(b_i.to(dev), a_i.to(dev))
            if scale_i != 1.0:
                delta = delta * scale_i
            if delta.dtype != merged_w.dtype:
                delta = delta.to(dtype=merged_w.dtype)
            shard_size = delta.shape[0]
            if output_slices is not None and i < len(output_slices):
                expected = output_slices[i]
                if shard_size != expected:
                    offset += expected
                    continue
                shard_size = expected
            merged_w[offset : offset + shard_size].add_(delta)
            offset += shard_size
            sub_ok += 1
        except Exception:
            if output_slices is not None and i < len(output_slices):
                offset += output_slices[i]

    if sub_ok > 0:
        with torch.no_grad():
            W.copy_(merged_w)
        return True
    return False


def _restore_packed_layer_subtract(
    *,
    module: Any,
    lora_layer: Any,
    A_list: list,
    B_list: list,
    W: torch.Tensor,
) -> bool:
    """Subtract a packed LoRA layer delta (reverse of _merge_packed_layer)."""
    scalings = getattr(lora_layer, "scaling", None)
    if not isinstance(scalings, list):
        scalings = [_get_scaling(lora_layer)] * len(A_list)

    tp_size = getattr(module, "tp_size", 1)
    try:
        A_sliced = module.slice_lora_a(A_list) if tp_size > 1 else A_list
        B_sliced = module.slice_lora_b(B_list) if tp_size > 1 else B_list
    except Exception:
        return False

    output_slices = getattr(module, "output_slices", None)
    dev = W.device
    sub_ok = 0
    offset = 0

    for i, (a_i, b_i) in enumerate(zip(A_sliced, B_sliced)):
        if a_i is None or b_i is None:
            if output_slices is not None and i < len(output_slices):
                offset += output_slices[i]
            continue
        scale_i = float(scalings[i]) if i < len(scalings) else 1.0
        try:
            delta = torch.matmul(b_i.to(dev), a_i.to(dev))
            if scale_i != 1.0:
                delta = delta * scale_i
            if delta.dtype != W.dtype:
                delta = delta.to(dtype=W.dtype)
            shard_size = delta.shape[0]
            if output_slices is not None and i < len(output_slices):
                expected = output_slices[i]
                if shard_size != expected:
                    offset += expected
                    continue
                shard_size = expected
            with torch.no_grad():
                W[offset : offset + shard_size].sub_(delta)
            offset += shard_size
            sub_ok += 1
        except Exception:
            if output_slices is not None and i < len(output_slices):
                offset += output_slices[i]

    return sub_ok > 0


def bf16_restore_base(
    *,
    model_manager: Any,
    golden_cache: BF16GoldenCache,
    lora_module_names: set[str],
    lora_model: Any | None = None,
) -> MergeResult:
    """Restore base weights.

    With golden copies (cpu/gpu): W <- golden (exact restore).
    Without golden copies (off): W -= scale * (B @ A) (subtract delta).
    """
    modules: dict[str, Any] = getattr(model_manager, "modules", {})
    restored = 0
    skipped = 0

    if golden_cache.is_off:
        # Subtract mode: recompute and subtract the delta
        if lora_model is None:
            return MergeResult(ok=False, reason="off mode needs lora_model")
        loras: dict[str, Any] = getattr(lora_model, "loras", {})
        for module_name in lora_module_names:
            module = modules.get(module_name)
            if module is None:
                skipped += 1
                continue
            W = _get_base_weight_tensor(module)
            if W is None:
                skipped += 1
                continue
            lora_layer = loras.get(module_name)
            if lora_layer is None:
                skipped += 1
                continue

            A = getattr(lora_layer, "lora_a", None)
            B = getattr(lora_layer, "lora_b", None)
            is_packed = getattr(lora_layer, "is_packed", False)

            if is_packed and isinstance(A, list) and isinstance(B, list):
                # Subtract packed layer delta
                ok = _restore_packed_layer_subtract(
                    module=module,
                    lora_layer=lora_layer,
                    A_list=A,
                    B_list=B,
                    W=W,
                )
                if ok:
                    restored += 1
                else:
                    skipped += 1
                continue

            if not (torch.is_tensor(A) and torch.is_tensor(B)):
                skipped += 1
                continue
            scale = _get_scaling(lora_layer)
            delta = _slice_and_compute_delta(
                module=module,
                A=A,
                B=B,
                W=W,
                scale=scale,
                module_name=module_name,
            )
            if delta is None:
                skipped += 1
                continue
            with torch.no_grad():
                W.sub_(delta)
            restored += 1
    else:
        # Golden copy mode: exact restore
        for module_name in lora_module_names:
            module = modules.get(module_name)
            if module is None:
                skipped += 1
                continue
            W = _get_base_weight_tensor(module)
            if W is None:
                skipped += 1
                continue
            golden = golden_cache.get(module_name)
            if golden is None:
                skipped += 1
                continue
            with torch.no_grad():
                W.copy_(golden.to(device=W.device))
            restored += 1

    logger.info("automerge: restored %d modules", restored)

    return MergeResult(
        ok=restored > 0,
        reason="" if restored > 0 else "no layers restored",
        merged_modules=restored,
        skipped_modules=skipped,
    )
