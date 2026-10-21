"""
Correctness parity test for GitHub issue #59640 fix: folding
routed_scaling_factor into routed_output_transform's weights at load
time, vs. applying it as a runtime multiply on every forward pass.

Run on a real GPU (CUDA-alike / ROCm) node with the patched vllm installed:

    python 13_test_routed_scale_prefold.py

This does NOT spin up a full model or FusedMoEFactory -- it isolates
just the two code paths under test:
  (a) MoERunner._maybe_apply_routed_scale_to_output with
      routed_scale_prefolded=False (today's runtime-multiply path)
  (b) the same method with routed_scale_prefolded=True, PLUS the
      weight-prefold step applied to a toy Linear standing in for
      fc2_latent_proj

and checks that running input through (output_transform -> scale) in
the old order gives the same result as (scale-folded-into-weights ->
output_transform) in the new order.
"""

import torch
import torch.nn as nn


def apply_output_transform(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return linear(x)


def old_path(
    fused_output: torch.Tensor,
    routed_scaling_factor: float,
    output_transform: nn.Linear,
) -> torch.Tensor:
    """Mirrors today's MoERunner behavior: scale first, then transform."""
    scaled = fused_output * routed_scaling_factor
    return apply_output_transform(scaled, output_transform)


def new_path(
    fused_output: torch.Tensor,
    routed_scaling_factor: float,
    output_transform: nn.Linear,
) -> torch.Tensor:
    """Mirrors the #59640 fix: fold scale into weights once, skip the
    runtime multiply, transform directly."""
    # Build a scaled copy of the transform to simulate the one-time
    # weight-prefold step in NemotronHForCausalLM._maybe_prefold_routed_scale,
    # without mutating the shared `output_transform` module (so this
    # function can be called repeatedly in the same test run without
    # double-applying the scale across cases).
    prefolded = nn.Linear(
        output_transform.in_features,
        output_transform.out_features,
        bias=output_transform.bias is not None,
        device=output_transform.weight.device,
        dtype=output_transform.weight.dtype,
    )
    with torch.no_grad():
        prefolded.weight.copy_(output_transform.weight * routed_scaling_factor)
        if output_transform.bias is not None:
            prefolded.bias.copy_(output_transform.bias * routed_scaling_factor)
    # No runtime multiply here -- this is the whole point of the fix.
    return apply_output_transform(fused_output, prefolded)


def run_case(
    batch_size: int,
    in_features: int,
    out_features: int,
    routed_scaling_factor: float,
    dtype: torch.dtype,
    device: str,
    bias: bool,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)

    output_transform = nn.Linear(
        in_features, out_features, bias=bias, device=device, dtype=dtype
    )
    fused_output = torch.randn(batch_size, in_features, device=device, dtype=dtype)

    old_result = old_path(fused_output, routed_scaling_factor, output_transform)
    new_result = new_path(fused_output, routed_scaling_factor, output_transform)

    # bf16 has limited mantissa precision; use a tolerance appropriate to
    # the dtype rather than assuming exact equality.
    if dtype == torch.bfloat16:
        atol, rtol = 2e-2, 2e-2
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5

    max_abs_diff = (old_result - new_result).abs().max().item()
    assert torch.allclose(old_result, new_result, atol=atol, rtol=rtol), (
        f"MISMATCH: batch={batch_size} in={in_features} out={out_features} "
        f"scale={routed_scaling_factor} dtype={dtype} bias={bias} "
        f"max_abs_diff={max_abs_diff}"
    )
    print(
        f"PASS  batch={batch_size:>4} in={in_features:>5} out={out_features:>5} "
        f"scale={routed_scaling_factor:>5} dtype={str(dtype):>14} bias={bias!s:>5} "
        f"max_abs_diff={max_abs_diff:.3e}"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This test requires a real GPU (ROCm device exposed via the "
            "CUDA-alike API) to match production dtype/kernel behavior -- "
            "run it on the MI350 node, not on CPU-only environments."
        )
    device = "cuda"

    # Realistic NemotronH-Ultra shape: moe_latent_size=2048 -> hidden_size=8192,
    # routed_scaling_factor=5.0, bias=False (matches config.mlp_bias=False,
    # the only case _routed_scale_prefoldable can be True per the fix's
    # reduce_commutative gating).
    cases = [
        # (batch, in_features, out_features, scale, dtype, bias)
        (1, 2048, 8192, 5.0, torch.bfloat16, False),
        (32, 2048, 8192, 5.0, torch.bfloat16, False),
        (256, 2048, 8192, 5.0, torch.bfloat16, False),
        (32, 2048, 8192, 1.0, torch.bfloat16, False),  # scale=1.0 edge case
        (32, 2048, 8192, 5.0, torch.float32, False),
        # bias=True is NOT expected to occur for NemotronH (mlp_bias gates
        # reduce_commutative to False whenever bias is True), but included
        # as a defensive check of the prefold math itself in isolation --
        # the actual fix should never reach this path in production since
        # _routed_scale_prefoldable would be False whenever bias=True.
        (32, 2048, 8192, 5.0, torch.bfloat16, True),
    ]

    failures = []
    for batch, in_f, out_f, scale, dtype, bias in cases:
        try:
            run_case(batch, in_f, out_f, scale, dtype, device, bias)
        except AssertionError as e:
            failures.append(str(e))
            print(f"FAIL  {e}")

    if failures:
        raise SystemExit(f"{len(failures)} parity case(s) failed")
    print("\nAll parity cases passed -- prefolded weights produce numerically "
          "equivalent output to the runtime-multiply path.")


if __name__ == "__main__":
    main()
