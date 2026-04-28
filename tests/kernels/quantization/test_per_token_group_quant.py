# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils, int8_utils
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "shape", [(31, 128), (32, 128), (63, 256), (64, 256), (16, 512)]
)
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("tma_aligned", [False, True])
@pytest.mark.parametrize("scale_ue8m0", [False, True])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_per_token_group_quant_fp8(
    shape, column_major: bool, tma_aligned: bool, scale_ue8m0: bool, group_size: int
):
    device = "cuda"

    torch.manual_seed(42)
    num_tokens, hidden_dim = shape

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    # cuda path
    out_q, scale = fp8_utils.per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major,
        tma_aligned_scales=tma_aligned,
        use_ue8m0=scale_ue8m0,
    )

    # triton ref
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            column_major_scales=column_major,
            use_ue8m0=scale_ue8m0,
        )

    assert torch.allclose(out_q.float(), ref_q.float(), atol=0.15, rtol=0.15)
    assert torch.allclose(scale, ref_s, atol=0.01, rtol=0.01)


@pytest.mark.parametrize(
    "num_tokens,hidden_dim,group_size",
    [
        # No padding: mn=4 (mult of 4), groups_per_row=56 (mult of 4)
        (4, 7168, 128),
        # MN padding only: mn=1, tma_aligned_mn=4
        (1, 7168, 128),
        # MN padding only: mn=3, tma_aligned_mn=4
        (3, 7168, 128),
        # K padding only: groups_per_row=5 (5%4=1)
        (4, 640, 128),
        # K padding only: groups_per_row=6 (6%4=2)
        (4, 768, 128),
        # Single packed column, no padding: k_num_packed=1, mn%4=0
        (4, 384, 128),
        # Both MN and K padding
        (1, 384, 128),
        (3, 640, 128),
        # Larger shapes with no padding
        (64, 7168, 128),
        (128, 14336, 128),
        # Larger shapes with padding
        (127, 7168, 128),
        (253, 640, 128),
        # Non-power-of-2 group size
        (4, 768, 96),  # 768/96=8 groups, no padding
        (3, 768, 96),  # 768/96=8 groups, MN padding
        (4, 480, 96),  # 480/96=5 groups, K padding
        (1, 480, 96),  # both MN and K padding
    ],
)
@pytest.mark.parametrize("poisoned_scales", [False, True])
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="DeepGEMM not available on this platform"
)
def test_per_token_group_quant_fp8_packed(
    num_tokens, hidden_dim, group_size, poisoned_scales
):
    """Test the packed DeepGEMM quantization kernel against the Triton
    reference (row-major, UE8M0 scales)."""

    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    mn = num_tokens
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4
    num_scale_elems = mn + (k_num_packed - 1) * tma_aligned_mn

    if poisoned_scales:
        # Call the kernel with poisoned scale buffer to
        # ensure padded indices are correctly zeroed.
        fp8_dtype = torch.float8_e4m3fn
        finfo = torch.finfo(fp8_dtype)
        out_q = torch.empty_like(x, dtype=fp8_dtype)
        out_s_packed = torch.empty_strided(
            (mn, k_num_packed),
            (1, tma_aligned_mn),
            device=device,
            dtype=torch.int32,
        )
        torch.as_strided(out_s_packed, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)
        torch.ops._C.per_token_group_fp8_quant_packed(
            x,
            out_q,
            out_s_packed,
            group_size,
            1e-10,
            finfo.min,
            finfo.max,
        )
    else:
        out_q, out_s_packed = fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
            x,
            group_size=group_size,
            use_ue8m0=True,
        )

    # Triton reference (row-major float32 scales, UE8M0)
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            use_ue8m0=True,
        )

    # Quantized values must match.
    assert torch.equal(out_q, ref_q), "Quantized output mismatch"

    # Verify packed scales (valid exponents + padding zeros).
    ref_s_flat = ref_s.reshape(mn, groups_per_row)
    ref_exponents = (ref_s_flat.view(torch.int32) >> 23) & 0xFF

    expected = torch.zeros(num_scale_elems, dtype=torch.int32, device="cpu")
    for row in range(mn):
        for g in range(groups_per_row):
            pack_col = g // 4
            pos = g % 4
            idx = pack_col * tma_aligned_mn + row
            expected[idx] |= int(ref_exponents[row, g].item()) << (pos * 8)

    actual = torch.as_strided(out_s_packed, (num_scale_elems,), (1,)).cpu()
    assert torch.equal(actual, expected), (
        f"Packed scale storage mismatch.\n"
        f"First diff at index "
        f"{(actual != expected).nonzero(as_tuple=True)[0][0].item()}"
    )


@pytest.mark.parametrize("shape", [(32, 128), (64, 256), (16, 512)])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_per_token_group_quant_int8(shape, group_size: int):
    device = "cuda"

    torch.manual_seed(42)
    num_tokens, hidden_dim = shape

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    # cuda path
    out_q, scale = int8_utils.per_token_group_quant_int8(
        x,
        group_size,
    )

    # triton ref
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        ref_q, ref_s = int8_utils.per_token_group_quant_int8(
            x,
            group_size,
        )

    assert torch.allclose(out_q.float(), ref_q.float(), atol=0.15, rtol=0.15)
    assert torch.allclose(scale, ref_s, atol=0.01, rtol=0.01)
