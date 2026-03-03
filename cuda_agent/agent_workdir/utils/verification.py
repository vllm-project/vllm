"""
utils/verification.py

Verifies correctness of the agent's CUDA extension model (ModelNew)
against the reference PyTorch model (Model).

Do NOT modify this file.

Anti-reward-hacking safeguards:
  - Runs 5 independent verification passes with freshly sampled inputs.
  - Patches torch.nn.functional to prevent ModelNew from delegating to
    standard PyTorch ops — the CUDA extension must perform genuine work.
  - File permissions prevent the agent from altering this script.

Usage:
    python3 -m utils.verification
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def transform_tensors(obj: Any, fn) -> Any:
    """Recursively apply fn to every tensor in a nested container."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: transform_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        transformed = [transform_tensors(v, fn) for v in obj]
        return type(obj)(transformed)
    return obj


def check_equal(actual: Any, expected: Any, label: str = "") -> None:
    """Assert that actual and expected tensors (or containers) are close."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, atol=1e-2, rtol=1e-2,
            msg=f"Output mismatch{' (' + label + ')' if label else ''}"
        )
    elif isinstance(expected, (list, tuple)):
        for i, (a, e) in enumerate(zip(actual, expected)):
            check_equal(a, e, label=f"{label}[{i}]")
    elif isinstance(expected, dict):
        for k in expected:
            check_equal(actual[k], expected[k], label=f"{label}[{k!r}]")


@contextlib.contextmanager
def block_torch_functional():
    """Context manager: raise AttributeError for any torch.functional access."""
    original = torch.nn.functional
    mock = types.ModuleType("torch.nn.functional")

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "ModelNew must not call torch.nn.functional operations. "
            "Use custom CUDA kernels instead."
        )

    for attr in dir(original):
        if not attr.startswith("_"):
            setattr(mock, attr, _blocked)

    torch.nn.functional = mock  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.nn.functional = original


def initialize_models():
    """Import and instantiate both models, move them to GPU."""
    # Fresh import to avoid cached state
    for mod_name in ["model", "model_new"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    model_mod = importlib.import_module("model")
    model_new_mod = importlib.import_module("model_new")

    init_inputs = model_mod.get_init_inputs()
    ref_model = model_mod.Model(*init_inputs).cuda().eval()
    new_model = model_new_mod.ModelNew(*init_inputs).cuda().eval()
    return ref_model, new_model, model_mod


def build_inputs(model_mod):
    """Build a fresh set of GPU inputs."""
    inputs = model_mod.get_inputs()
    return transform_tensors(inputs, lambda t: t.cuda())


# ---------------------------------------------------------------------------
# Main verification loop
# ---------------------------------------------------------------------------

NUM_PASSES = 5


def main() -> int:
    print(f"[VERIFY] Running {NUM_PASSES} verification passes …")
    ref_model, new_model, model_mod = initialize_models()

    all_passed = True
    for i in range(NUM_PASSES):
        inputs = build_inputs(model_mod)

        with torch.no_grad():
            expected = ref_model(*inputs)

        with torch.no_grad(), block_torch_functional():
            try:
                actual = new_model(*inputs)
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL]  Pass {i + 1}/{NUM_PASSES}: ModelNew raised {exc}")
                all_passed = False
                continue

        try:
            check_equal(actual, expected, label=f"pass {i + 1}")
            print(f"[PASS]  Pass {i + 1}/{NUM_PASSES}: outputs match")
        except AssertionError as exc:
            print(f"[FAIL]  Pass {i + 1}/{NUM_PASSES}: {exc}")
            all_passed = False

    if all_passed:
        print(f"\n[VERIFY] All {NUM_PASSES} passes succeeded. ✓")
        return 0
    else:
        print("\n[VERIFY] One or more verification passes FAILED. ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
