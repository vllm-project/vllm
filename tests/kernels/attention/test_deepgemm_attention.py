# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _ceil_to_ue8m0,
    calc_diff,
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    # x: (num_blocks, block_size, 1, head_dim)
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: tuple, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _generate_cp_test_data(seq_len: int, seq_len_kv: int):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    seq_len_kv = kv.shape[0]

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
@pytest.mark.parametrize("clean_logits", [True, False])
def test_deepgemm_fp8_mqa_logits(clean_logits: bool):
    torch.manual_seed(0)
    random.seed(0)
    num_heads, head_dim = 32, 128
    for seq_len in (512,):
        for seq_len_kv in (1024,):
            for disable_cp in (False, True):
                q = torch.randn(
                    seq_len,
                    num_heads,
                    head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv = torch.randn(
                    seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16
                )
                weights = torch.randn(
                    seq_len, num_heads, device="cuda", dtype=torch.float32
                )

                if disable_cp:
                    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
                    ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
                        seq_len_kv - seq_len
                    )
                else:
                    ks, ke = _generate_cp_test_data(seq_len, seq_len_kv)

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)
                logits = fp8_fp4_mqa_logits(
                    (q_fp8, None), kv_fp8, weights, ks, ke, clean_logits=clean_logits
                )

                ref_logits = _ref_fp8_mqa_logits(
                    q=q,
                    kv=kv,
                    weights=weights,
                    cu_seqlen_ks=ks,
                    cu_seqlen_ke=ke,
                )
                ref_neginf_mask = ref_logits == float("-inf")

                if clean_logits:
                    neginf_mask = logits == float("-inf")
                    assert torch.equal(neginf_mask, ref_neginf_mask)

                ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                logits = logits.masked_fill(ref_neginf_mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"


def _ref_fp8_fp4_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device="cuda",
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
def test_deepgemm_fp8_fp4_paged_mqa_logits():
    # NOTE: clean_logits=True is incompatible with the 2D context_lens
    # required by csrc/apis/attention.hpp; only the False path is exercised.
    clean_logits = False
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    for batch_size, next_n in [(4, 1), (2, 2)]:
        for heads, index_dim in [(32, 128)]:
            for avg_kv in (2048,):
                num_blocks, blocksize = max_model_len * 2, 64

                q = torch.randn(
                    (batch_size, next_n, heads, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv_cache = torch.randn(
                    (num_blocks, blocksize, 1, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                weights = torch.randn(
                    (batch_size * next_n, heads),
                    device="cuda",
                    dtype=torch.float32,
                )

                context_lens = (
                    torch.randint(int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,))
                    .cuda()
                    .to(torch.int32)
                )
                max_block_len = (
                    (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                )
                block_tables = torch.zeros(
                    (batch_size, max_block_len),
                    device="cuda",
                    dtype=torch.int32,
                )

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = int(context_lens[i].item())
                    for j in range((ctx_len + blocksize - 1) // blocksize):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

                # deep_gemm paged MQA logits requires 2D context_lens of
                # shape (B, next_n) (csrc/apis/attention.hpp:332-335);
                # see indexer.py:607-608. For each batch/next_n token, the
                # effective context length is context_lens[b] - next_n + j + 1.
                next_n_arange = torch.arange(next_n, device="cuda", dtype=torch.int32)
                context_lens_2d = (
                    context_lens.unsqueeze(-1) - next_n + 1 + next_n_arange
                ).contiguous()
                schedule_metadata = get_paged_mqa_logits_metadata(
                    context_lens_2d, blocksize, get_num_sms()
                )
                logits = fp8_fp4_paged_mqa_logits(
                    (q_fp8, None),
                    kv_cache_fp8,
                    weights,
                    context_lens_2d,
                    block_tables,
                    schedule_metadata,
                    max_model_len,
                    clean_logits=clean_logits,
                )

                ref_logits = _ref_fp8_fp4_paged_mqa_logits(
                    q,
                    kv_cache,
                    weights,
                    context_lens,
                    block_tables,
                    max_model_len,
                )

                positions = (
                    torch.arange(max_model_len, device="cuda")
                    .unsqueeze(0)
                    .expand(batch_size * next_n, -1)
                )
                row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
                next_n_offset = (
                    torch.arange(batch_size * next_n, device="cuda") % next_n
                )
                mask = positions <= (
                    context_lens[row_indices] - next_n + next_n_offset
                ).unsqueeze(1)

                logits = logits.masked_fill(~mask, 0)
                ref_logits = ref_logits.masked_fill(~mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
@pytest.mark.parametrize("context_len", [512, 513, 2052])
def test_deepgemm_paged_and_contiguous_indexer_logits_exact(context_len: int):
    """P/D gate: identical FP8 Q/K bytes must produce identical logits.

    The 513 case is the first DS4 position where top-512 selection observes
    logits instead of returning every candidate. 2052 covers the exact first
    mismatch seen in the fixed DAPO 2K capsule.
    """
    torch.manual_seed(20260815)
    # DeepGEMM consumes the cache's storage block size (64), not the
    # uncompressed sparse-attention scheduling page size (256).
    heads, head_dim, block_size = 64, 128, 64
    max_model_len = 2304
    num_blocks = cdiv(context_len, block_size)
    q = torch.randn(
        (1, 1, heads, head_dim), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(
        (num_blocks, block_size, 1, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.randn((1, heads), device="cuda", dtype=torch.float32)
    packed = kv_cache_cast_to_fp8(kv)
    packed_bytes = packed.view(num_blocks, -1)
    contiguous_k = packed_bytes[:, : block_size * head_dim].reshape(
        num_blocks * block_size, head_dim
    ).view(torch.float8_e4m3fn)[:context_len]
    contiguous_scale = packed_bytes[:, block_size * head_dim :].reshape(
        num_blocks * block_size, 4
    ).view(torch.float32).reshape(-1)[:context_len]

    context_lens = torch.tensor([[context_len]], device="cuda", dtype=torch.int32)
    block_table = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    schedule = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms()
    )
    paged = fp8_fp4_paged_mqa_logits(
        (q, None),
        packed,
        weights,
        context_lens,
        block_table,
        schedule,
        max_model_len,
        clean_logits=False,
    )[0, :context_len]
    # The target request must also be invariant to a co-batched request and
    # the resulting scheduler metadata/layout.
    q_batched = torch.cat(
        [
            q,
            torch.randn_like(q.to(torch.bfloat16)).to(torch.float8_e4m3fn),
        ],
        dim=0,
    )
    weights_batched = torch.cat(
        [weights, torch.randn_like(weights)], dim=0
    )
    context_lens_batched = torch.tensor(
        [[context_len], [max(1, context_len - 7)]],
        device="cuda",
        dtype=torch.int32,
    )
    block_table_batched = block_table.expand(2, -1).contiguous()
    schedule_batched = get_paged_mqa_logits_metadata(
        context_lens_batched, block_size, get_num_sms()
    )
    paged_batched = fp8_fp4_paged_mqa_logits(
        (q_batched, None),
        packed,
        weights_batched,
        context_lens_batched,
        block_table_batched,
        schedule_batched,
        max_model_len,
        clean_logits=False,
    )[0, :context_len]
    assert torch.equal(paged, paged_batched), (
        context_len,
        "paged target changed under co-batching",
        int((paged != paged_batched).sum().item()),
        float((paged - paged_batched).abs().max().item()),
    )
    contiguous = fp8_fp4_mqa_logits(
        (q.reshape(1, heads, head_dim), None),
        (contiguous_k, contiguous_scale),
        weights,
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.tensor([context_len], device="cuda", dtype=torch.int32),
        clean_logits=False,
    )[0, :context_len]
    assert torch.equal(paged, contiguous), (
        context_len,
        int((paged != contiguous).sum().item()),
        float((paged - contiguous).abs().max().item()),
    )
