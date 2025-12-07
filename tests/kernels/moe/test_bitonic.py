# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.fused_moe.triton_bitonic_sort import (
    bitonic_compare_exchange_descending_wrapper,
    bitonic_sort_warp_size_descending,
    bitonic_sort_warp_size_descending_wrapper,
)
from vllm.triton_utils import tl, triton


def test_bitonic_descending():
    val = torch.arange(32, dtype=torch.float32, device="cuda")
    seq = torch.arange(32, dtype=torch.int32, device="cuda")
    new_val = torch.zeros(32, dtype=torch.float32, device="cuda")
    new_seq = torch.zeros(32, dtype=torch.int32, device="cuda")
    ref_1_seq = torch.tensor(
        [
            1,
            0,
            2,
            3,
            5,
            4,
            6,
            7,
            9,
            8,
            10,
            11,
            13,
            12,
            14,
            15,
            17,
            16,
            18,
            19,
            21,
            20,
            22,
            23,
            25,
            24,
            26,
            27,
            29,
            28,
            30,
            31,
        ],
        dtype=torch.int32,
        device="cuda",
    )

    # assert stride 1 is correct when constructing bitonic
    bitonic_compare_exchange_descending_wrapper[(1,)](val, seq, new_val, new_seq, 1, 1)
    torch.testing.assert_close(new_seq, ref_1_seq)

    # assert final sort result
    bitonic_sort_warp_size_descending_wrapper[(1,)](val, seq, new_val, new_seq)
    seq = seq.flip(0)
    torch.testing.assert_close(new_seq, seq)


@triton.jit
def test_bitonic_2d_kernel(
    in_ptr,
    out_val_ptr,
    out_idx_ptr,
    ROWS: tl.constexpr,
):
    offs_row = tl.arange(0, ROWS)
    offs_col = tl.arange(0, 32)

    vals = tl.load(in_ptr + offs_row[:, None] * 32 + offs_col[None, :])  # [ROWS, 32]

    idxs = tl.broadcast_to(offs_col[None, :], (ROWS, 32)).to(tl.int32)  # [ROWS, 32]

    sorted_vals, sorted_idxs = bitonic_sort_warp_size_descending(vals, idxs)

    tl.store(out_val_ptr + offs_row[:, None] * 32 + offs_col[None, :], sorted_vals)
    tl.store(out_idx_ptr + offs_row[:, None] * 32 + offs_col[None, :], sorted_idxs)


def test_bitonic_multirow():
    for ROWS in [1, 2, 4, 8]:
        torch.manual_seed(42)
        x = torch.randn(ROWS, 32, device="cuda", dtype=torch.float32)
        out_vals = torch.empty_like(x)
        out_idxs = torch.empty(ROWS, 32, device="cuda", dtype=torch.int32)

        # assumingly, num_warps >= ROWS
        test_bitonic_2d_kernel[(1,)](
            x,
            out_vals,
            out_idxs,
            ROWS=ROWS,
            num_warps=max(ROWS, 4),
        )

        expected_vals, expected_idxs = x.sort(dim=1, descending=True)

        vals_match = torch.allclose(out_vals, expected_vals)
        idxs_match = torch.equal(out_idxs, expected_idxs.to(torch.int32))

        print(f"values match: {vals_match}")
        print(f"indices match: {idxs_match}")

        if not vals_match or not idxs_match:
            print("input:")
            print(x)
            print("result vals:")
            print(out_vals)
            print("expected vals:")
            print(expected_vals)
            print("result idxs:")
            print(out_idxs)
            print("expected idxs:")
            print(expected_idxs)


if __name__ == "__main__":
    test_bitonic_multirow()
