# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.gemma4 import (
    gemma4_fused_routing_kernel_triton,
    gemma4_routing_function_torch,
)


def sort_by_id(w, ids):
    order = ids.argsort(dim=-1)
    return w.gather(1, order), ids.gather(1, order)


# Gemma4 MoE Model has context length of 250K
# the minus 1 is to ensure that edge cases are tested
@pytest.mark.parametrize("num_tokens", [1, 2, 2048, 250000])
@pytest.mark.parametrize("num_experts", [128])  # gemma4 moe experts
@pytest.mark.parametrize("topk", [8])  # gemma4 topk
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_gemma4_routing_kernel_triton(
    num_tokens: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.manual_seed(0)

    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")
    scales = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    tri_w, tri_ids = gemma4_fused_routing_kernel_triton(gating, topk, scales)

    # Used with gamma4_routing_function_torch, it doesn't use the
    # first return value and the second value needs to be a LongTensor.
    def topk_function(x, k, dim):
        return None, tri_ids.long()

    # The two properties needed will be checked separately, in this order:
    # 1) Check that the tri_ids do constitute a valid top-k set.
    #
    # Use the top-k indices to check that the gating values were actually
    # a valid top-k
    #
    # 2) Check that the Triton implementation computes weights correctly.
    #
    # To check that the weights returned have been computed correctly,
    # take the returned ids, which were already checked and return them
    # into the Torch implementation using the topk_function above.
    #
    # This process is used  because torch.topk is unstable and tl.sort
    # uses Bitonic Mergesort, which is also unstable.
    #
    # Since scales are applied after the top-k computation, the
    # scale * weight computation could be different for two unshared experts
    # that tied for the same place since the scale could be different,
    # but gating value the same.

    ref_w, ref_ids = gemma4_routing_function_torch(
        gating, topk, scales, topk_function=topk_function
    )

    assert ref_ids.shape == tri_ids.shape, (
        f"Returned weights shape must match reference weights shape,"
        f"ref_ids.shape={ref_ids.shape}, tri_ids.shape={tri_ids.shape}."
    )
    assert ref_ids.dtype == tri_ids.dtype, (
        "Returned weights dtype must match reference dtype."
    )

    assert ref_w.shape == tri_w.shape, (
        f"Returned weights shape must match reference weights shape,"
        f"ref_w.shape={ref_w.shape}, tri_w.shape={tri_w.shape}."
    )
    assert ref_w.dtype == tri_w.dtype, (
        "Returned weights dtype must match reference dtype."
    )

    assert (tri_ids >= 0).all().item() and (tri_ids < num_experts).all().item(), (
        f"Returned indices must be within the range of 0 to {num_experts}"
    )

    # 1) Check for valid top-k.
    #
    # Get a stable sort of the gating values and take the top-k.
    topk_values, topk_indices = torch.sort(gating, descending=True, stable=True, dim=-1)
    topk_values = topk_values[:, :topk]
    topk_indices = topk_indices[:, :topk]

    # Gather all of the gating values corresponding to the selected top-k ids.
    # Since softmax preserves the ordering of a monotonic sequence, this can
    # be used do check that the ids returned from the Triton implementation
    # form a valid top-k.
    tri_gating_values = (
        gating.gather(dim=-1, index=tri_ids)
        .sort(descending=True, stable=True, dim=-1)
        .values
    )

    # Check that the top-k gating values returned for each token are a top-k.
    torch.testing.assert_close(topk_values, tri_gating_values, atol=1e-2, rtol=1e-2)

    # 2) Check for correct weight computation.
    ref_ws, ref_is = sort_by_id(ref_w, ref_ids)
    tri_ws, tri_is = sort_by_id(tri_w, tri_ids)

    weights_match = torch.allclose(ref_ws, tri_ws, atol=1e-2, rtol=1e-2)
    max_err = (ref_ws - tri_ws).abs().max().item()
    if not weights_match:
        bad = (ref_is != tri_is).any(dim=-1).nonzero(as_tuple=True)[0]
        if len(bad):
            r = bad[0].item()
            print(
                f"  first bad row {r}: ref_ids={ref_ids[r].tolist()} "
                f"tri_ids={tri_ids[r].tolist()}"
            )
        assert weights_match
