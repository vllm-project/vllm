# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL sparse MLA over fp8_ds_mla records (Kimi-K3.1 DSA layers, gfx950)."""

import itertools

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(not _on_gfx950(), reason="needs gfx950 (MI350/MI355)")

KV_LORA_RANK = 512
ROPE_DIM = 64
D_QK = KV_LORA_RANK + ROPE_DIM
RECORD_BYTES = 656
BLOCK_SIZE = 64


@pytest.fixture(scope="module")
def flydsl_mla():
    pytest.importorskip("flydsl")
    from vllm.models.kimi_k3.amd.ops import flydsl_sparse_mla

    return flydsl_sparse_mla


def _make_cache(num_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Write random latents through vLLM's fp8_ds_mla writer.

    Returns the [num_blocks, BLOCK_SIZE, 656] uint8 pool and its fp32
    dequantized [num_slots, 576] rows.
    """
    import vllm._C_stable_libtorch  # noqa: F401  (registers _C_cache_ops)

    from vllm import _custom_ops as ops

    device = torch.device("cuda")
    num_blocks = -(-num_slots // BLOCK_SIZE)
    pool = torch.zeros(
        num_blocks, BLOCK_SIZE, RECORD_BYTES, dtype=torch.uint8, device=device
    )
    kv_c = torch.randn(num_slots, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(num_slots, ROPE_DIM, dtype=torch.bfloat16, device=device)
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        pool.view(torch.float8_e4m3fn),
        torch.arange(num_slots, dtype=torch.int64, device=device),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, dtype=torch.float32, device=device),
    )
    rows = pool.view(-1, RECORD_BYTES)[:num_slots]
    latent = rows[:, :KV_LORA_RANK].view(torch.float8_e4m3fn).float()
    scales = (
        rows[:, KV_LORA_RANK : KV_LORA_RANK + 16]
        .view(torch.float32)
        .repeat_interleave(128, dim=-1)
    )
    rope = rows[:, KV_LORA_RANK + 16 :].view(torch.bfloat16).float()
    return pool, torch.cat([latent * scales, rope], dim=-1)


def _reference(q, k_all, indptr, indices, scale):
    """fp32 sparse MLA; rows with no valid key give out 0 and lse -inf.

    Unit-variance inputs keep outputs O(1), so the 2e-2 tolerance used below
    catches a 10% softmax-scale error or an off-by-one slot.
    """
    num_tokens, num_heads, _ = q.shape
    out = torch.zeros(num_tokens, num_heads, KV_LORA_RANK, device=q.device)
    lse = torch.full((num_tokens, num_heads), float("-inf"), device=q.device)
    for t in range(num_tokens):
        idx = indices[indptr[t] : indptr[t + 1]].long()
        idx = idx[idx >= 0]
        if idx.numel() == 0:
            continue
        keys = k_all[idx]
        s = (q[t].float() @ keys.T) * scale
        out[t] = torch.softmax(s, dim=-1) @ keys[:, :KV_LORA_RANK]
        lse[t] = torch.logsumexp(s, dim=-1)
    return out, lse


def _index_stream(lens, num_slots, invalid_every=0):
    device = torch.device("cuda")
    rows = []
    for n in lens:
        sel = torch.randperm(num_slots, device=device)[:n].to(torch.int32)
        if invalid_every:
            sel[::invalid_every] = -1
        rows.append(sel)
    indices = torch.cat(rows) if rows else torch.empty(0, dtype=torch.int32)
    indptr = torch.tensor(
        [0, *itertools.accumulate(lens)], dtype=torch.int32, device=device
    )
    return indptr, indices.to(device)


# Row lengths straddle the kernel's 128-token tiles and include an empty row.
LENS = [0, 1, 127, 128, 129, 300, 640]


@pytest.mark.parametrize("num_heads", [1, 8, 16])
@pytest.mark.parametrize("has_invalid", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("kv_layout", ["paged", "flat"])
def test_matches_reference(flydsl_mla, num_heads, has_invalid, return_lse, kv_layout):
    set_random_seed(0)
    num_slots = 1000
    pool, k_all = _make_cache(num_slots)
    kv = pool if kv_layout == "paged" else pool.view(-1, RECORD_BYTES)
    lens = LENS + ([5] if has_invalid else [])
    indptr, indices = _index_stream(lens, num_slots, 3 if has_invalid else 0)
    if has_invalid:
        # A row whose every entry is masked.
        indices[indptr[-2] : indptr[-1]] = -1
    q = torch.randn(len(lens), num_heads, D_QK, dtype=torch.bfloat16, device="cuda")
    scale = D_QK**-0.5

    out, lse = flydsl_mla.flydsl_sparse_mla_fp8ds_fwd(
        q, kv, indptr, indices, scale, has_invalid=has_invalid, return_lse=return_lse
    )
    ref_out, ref_lse = _reference(q, k_all, indptr, indices, scale)

    assert out.shape == (len(lens), num_heads, KV_LORA_RANK)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=0)
    if return_lse:
        assert lse is not None and lse.dtype == torch.float32
        empty = torch.isinf(ref_lse)
        assert torch.equal(torch.isneginf(lse), empty)
        torch.testing.assert_close(lse[~empty], ref_lse[~empty], atol=2e-2, rtol=0)
    else:
        assert lse is None


def test_int64_indices_and_strided_out(flydsl_mla):
    """int64 CSR inputs are accepted; a strided `out` view only gets 512 cols."""
    set_random_seed(1)
    num_slots, num_heads = 500, 4
    pool, k_all = _make_cache(num_slots)
    indptr, indices = _index_stream([3, 200, 64], num_slots)
    q = torch.randn(3, num_heads, D_QK, dtype=torch.bfloat16, device="cuda")
    wide = torch.full((3, num_heads, D_QK), 7.0, dtype=torch.bfloat16, device="cuda")
    scale = D_QK**-0.5

    out, _ = flydsl_mla.flydsl_sparse_mla_fp8ds_fwd(
        q, pool, indptr.long(), indices.long(), scale, out=wide[..., :KV_LORA_RANK]
    )
    ref_out, _ = _reference(q, k_all, indptr, indices, scale)

    assert out.data_ptr() == wide.data_ptr()
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=0)
    assert torch.all(wide[..., KV_LORA_RANK:] == 7.0)


def test_empty_batch_and_empty_index_stream(flydsl_mla):
    pool, _ = _make_cache(64)
    q = torch.zeros(0, 4, D_QK, dtype=torch.bfloat16, device="cuda")
    indptr = torch.zeros(1, dtype=torch.int32, device="cuda")
    indices = torch.empty(0, dtype=torch.int32, device="cuda")
    out, lse = flydsl_mla.flydsl_sparse_mla_fp8ds_fwd(
        q, pool, indptr, indices, 1.0, return_lse=True
    )
    assert out.shape == (0, 4, KV_LORA_RANK) and lse.shape == (0, 4)

    q = torch.randn(2, 4, D_QK, dtype=torch.bfloat16, device="cuda")
    indptr = torch.zeros(3, dtype=torch.int32, device="cuda")
    out, lse = flydsl_mla.flydsl_sparse_mla_fp8ds_fwd(
        q, pool, indptr, indices, 1.0, return_lse=True
    )
    assert torch.all(out == 0)
    assert torch.all(torch.isneginf(lse))


def _supported_args():
    device = torch.device("cuda")
    q = torch.zeros(2, 8, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.zeros(4, BLOCK_SIZE, RECORD_BYTES, dtype=torch.uint8, device=device)
    indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    indices = torch.zeros(2, dtype=torch.int32, device=device)
    return q, kv, indptr, indices


def test_supported_accepts_baseline(flydsl_mla):
    assert flydsl_mla.flydsl_sparse_mla_fp8ds_supported(*_supported_args())


@pytest.mark.parametrize(
    "case",
    [
        "too_many_heads",
        "fp32_q",
        "fp8_q_scale",
        "kv_scale",
        "attn_sink",
        "extra_segment",
        "kv_splits",
        "skip_reduce",
        "fp8_dots",
        "rope_128",
        "padded_page_pitch",
        "wrong_record_width",
        "indptr_length",
        "float_indices",
        "out_too_narrow",
        "cpu_q",
    ],
)
def test_supported_rejects(flydsl_mla, case):
    q, kv, indptr, indices = _supported_args()
    kwargs = {}
    if case == "too_many_heads":
        q = torch.zeros(2, 17, D_QK, dtype=torch.bfloat16, device="cuda")
    elif case == "fp32_q":
        q = q.float()
    elif case == "fp8_q_scale":
        kwargs["q_scale"] = torch.ones(1, device="cuda")
    elif case == "kv_scale":
        kwargs["kv_scale"] = torch.ones(1, device="cuda")
    elif case == "attn_sink":
        kwargs["attn_sink"] = torch.zeros(8, device="cuda")
    elif case == "extra_segment":
        kwargs["extra_kv"] = kv
    elif case == "kv_splits":
        kwargs["kv_splits"] = 2
    elif case == "skip_reduce":
        kwargs["skip_reduce"] = True
    elif case == "fp8_dots":
        kwargs["dot_precision"] = "fp8"
    elif case == "rope_128":
        kwargs["qk_rope_head_dim"] = 128
    elif case == "padded_page_pitch":
        kv = torch.zeros(4, BLOCK_SIZE, 704, dtype=torch.uint8, device="cuda")
        kv = kv[..., :RECORD_BYTES]
    elif case == "wrong_record_width":
        kv = torch.zeros(4, BLOCK_SIZE, 576, dtype=torch.uint8, device="cuda")
    elif case == "indptr_length":
        indptr = indptr[:2]
    elif case == "float_indices":
        indices = indices.float()
    elif case == "out_too_narrow":
        kwargs["out"] = torch.empty(2, 8, 256, dtype=torch.bfloat16, device="cuda")
    elif case == "cpu_q":
        q = q.cpu()
    assert not flydsl_mla.flydsl_sparse_mla_fp8ds_supported(
        q, kv, indptr, indices, **kwargs
    )


def test_fwd_raises_on_unsupported_call(flydsl_mla):
    q, kv, indptr, indices = _supported_args()
    with pytest.raises(ValueError, match="unsupported call"):
        flydsl_mla.flydsl_sparse_mla_fp8ds_fwd(q.float(), kv, indptr, indices, 1.0)


def test_warmup_then_graph_capture(flydsl_mla):
    """After warm-up the launch is capturable and replays with fresh inputs."""
    set_random_seed(2)
    num_slots, num_heads = 300, 8
    pool, k_all = _make_cache(num_slots)
    flydsl_mla.flydsl_sparse_mla_fp8ds_warmup(
        heads=(num_heads,), has_invalid=(False,), return_lse=(False,)
    )
    indptr, indices = _index_stream([10, 200], num_slots)
    q = torch.randn(2, num_heads, D_QK, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(2, num_heads, KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
    scale = D_QK**-0.5

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_mla.flydsl_sparse_mla_fp8ds_fwd(q, pool, indptr, indices, scale, out=out)

    new_indptr, new_indices = _index_stream([150, 60], num_slots)
    indptr.copy_(new_indptr)
    indices.copy_(new_indices)
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.accelerator.synchronize()

    ref_out, _ = _reference(q, k_all, indptr, indices, scale)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=0)
