# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify native mixed FMHA against the real vLLM writer, not a simulated pack."""

import pytest
import torch


@pytest.mark.parametrize("dim,layout", [(128, "HND"), (256, "NHD")])
def test_vllm_packed_writer_to_qmul(dim, layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires Blackwell")
    native = pytest.importorskip("flashinfer.attention.mixed_kv")
    from vllm import _custom_ops  # noqa: F401 (register the compiled cache ops)
    from vllm.utils.torch_utils import fp8_k_nvfp4_v_cache_split_views

    device = "cuda"
    pages, heads, page, tokens = 3, 2, 64, 69
    width = dim + dim // 2 + dim // 16
    shape = (pages, heads, page, width)
    if layout == "HND":
        writer_cache = torch.zeros(shape, device=device, dtype=torch.uint8).transpose(
            1, 2
        )
    else:
        writer_cache = torch.zeros(
            (pages, page, heads, width), device=device, dtype=torch.uint8
        )
    # Scale variation across tokens AND channel groups exposes a linear/swizzled
    # mismatch. Six times a power of two is exactly representable by the writer.
    rng = torch.Generator(device=device).manual_seed(23)
    exponents = torch.randint(
        -3, 4, (tokens, heads, dim // 16), device=device, generator=rng
    )
    v = (6 * (2.0**exponents)).repeat_interleave(16, -1).bfloat16()
    k = torch.zeros_like(v)
    slots = torch.arange(tokens, device=device)
    slots = torch.where(slots < page, 2 * page + slots, slots - page)
    scale = torch.tensor(0.5, device=device)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        k, v, writer_cache, writer_cache, slots, "fp8_k_nvfp4_v", scale, scale
    )
    views = fp8_k_nvfp4_v_cache_split_views(writer_cache, dim)
    k_cache, v_cache, sf = (x.transpose(1, 2) for x in views)
    q = torch.ones((tokens, heads * 4, dim), device=device).to(torch.float8_e4m3fn)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    table = torch.tensor([[2, 0]], device=device, dtype=torch.int32)
    length = torch.tensor([tokens], device=device, dtype=torch.int32)
    for n in (tokens, 1):
        ptr = torch.tensor([0, n], device=device, dtype=torch.int32)
        native.paged_fp8_k_nvfp4_v_fmha(
            q[:n],
            k_cache,
            v_cache,
            sf,
            table,
            length,
            ptr,
            max_q_len=n,
            out=out[:n],
            qk_scale=0.5 / dim**0.5,
            v_scale=0.5,
        )
        # K=0 makes causal attention a uniform average. Compute the answer
        # directly from the original values, without reading any packed bytes.
        denom = torch.arange(1, tokens + 1, device=device)[:, None, None]
        expected = (v.float().cumsum(0) / denom).repeat_interleave(4, dim=1)
        torch.testing.assert_close(
            out[:n].float(), expected[-n:], rtol=0.006, atol=0.01
        )
