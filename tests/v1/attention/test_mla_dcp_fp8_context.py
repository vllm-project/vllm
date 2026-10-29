# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fp8 MLA DCP chunked-context path against the dequantizing path.

The row gather must dequantize fp8 rows exactly as the default path's
dequantizing cache gather does.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.mla_dcp_context import gather_dcp_context_rows

_KV_LORA_RANK = 512
_QK_ROPE = 64
_ENTRY = _KV_LORA_RANK + _QK_ROPE


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "k_scale",
    [1.0, 0.37, 300.0, pytest.param(2.0**-120, id="subnormal-products")],
)
# The dequantizing cache gather takes rows of 320, 512 or 576 values; 448 + 64
# has a latent width that is not a power of two.
@pytest.mark.parametrize("dims", [(512, 64), (448, 64)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_rows", [0, 1, 33, 1000])
def test_gather_dcp_context_rows_dequantizes_like_the_cache_gather(
    num_rows, index_dtype, dims, k_scale, out_dtype
):
    """The rows come back exactly as the default path's dequantizing cache
    gather writes them, for any k_scale, either workspace dtype and a latent
    width that is not a power of two. The rows cover every fp8 code, NaN
    included."""
    from vllm import _custom_ops as ops

    kv_lora_rank, rope_dim = dims
    width = kv_lora_rank + rope_dim
    generator = torch.Generator(device="cuda").manual_seed(num_rows)
    raw = torch.randint(
        0, 256, (1200, width), dtype=torch.uint8, device="cuda", generator=generator
    )
    gathered = raw.view(current_platform.fp8_dtype())
    row_indices = torch.randperm(1200, device="cuda", generator=generator)
    row_indices = row_indices[:num_rows].to(index_dtype)
    # 0-dim, as layers hold their k_scale.
    scale = torch.tensor(k_scale, dtype=torch.float32, device="cuda")

    latent, k_pe = gather_dcp_context_rows(
        gathered, row_indices, kv_lora_rank, k_scale=scale, out_dtype=out_dtype
    )

    # Dequantize every row with the default path's gather, reading the rows
    # as a paged cache of 75 blocks of 16.
    dequantized = torch.empty((1200, width), dtype=out_dtype, device="cuda")
    ops.gather_and_maybe_dequant_cache(
        src_cache=gathered.view(75, 16, width),
        dst=dequantized,
        block_table=torch.arange(75, dtype=torch.int32, device="cuda")[None],
        cu_seq_lens=torch.tensor([0, 1200], dtype=torch.int32, device="cuda"),
        token_to_seq=torch.zeros(1200, dtype=torch.int32, device="cuda"),
        num_tokens=1200,
        kv_cache_dtype="fp8",
        scale=scale,
    )
    expected = dequantized[row_indices.long()].unsqueeze(1)
    assert latent.shape == (num_rows, 1, kv_lora_rank)
    assert k_pe.shape == (num_rows, 1, rope_dim)
    assert latent.dtype == k_pe.dtype == out_dtype
    assert latent.is_contiguous() and k_pe.is_contiguous()
    assert torch.equal(_bits(latent), _bits(expected[..., :kv_lora_rank]))
    assert torch.equal(_bits(k_pe), _bits(expected[..., kv_lora_rank:]))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize(
    "case",
    [
        "no_out_dtype",
        "no_k_scale",
        "fp32_out",
        "fp16_scale",
        "two_scales",
        "host_scale",
        "bf16_rows",
        "strided_rows",
    ],
)
def test_gather_dcp_context_rows_rejects_bad_dequant_inputs(case):
    """Dequantizing takes contiguous fp8 rows, a one-element fp32 k_scale on
    their device and a bf16 or fp16 output dtype, the last two together."""
    fp8 = current_platform.fp8_dtype()
    rows = torch.zeros((8, 2 * _ENTRY), dtype=torch.uint8, device="cuda").view(fp8)
    gathered = rows[:, :_ENTRY].contiguous()
    row_indices = torch.zeros(4, dtype=torch.int64, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device="cuda")
    kwargs = {"k_scale": scale, "out_dtype": torch.bfloat16}
    if case == "no_out_dtype":
        del kwargs["out_dtype"]
    elif case == "no_k_scale":
        del kwargs["k_scale"]
    elif case == "fp32_out":
        kwargs["out_dtype"] = torch.float32
    elif case == "fp16_scale":
        kwargs["k_scale"] = scale.half()
    elif case == "two_scales":
        kwargs["k_scale"] = scale.expand(2).contiguous()
    elif case == "host_scale":
        kwargs["k_scale"] = scale.cpu()
    elif case == "bf16_rows":
        gathered = gathered.to(torch.bfloat16)
    elif case == "strided_rows":
        gathered = rows[:, :_ENTRY]
    with pytest.raises(AssertionError):
        gather_dcp_context_rows(gathered, row_indices, _KV_LORA_RANK, **kwargs)
