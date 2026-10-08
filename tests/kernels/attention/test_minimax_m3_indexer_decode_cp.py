# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MSA indexer decode context parallelism (indexer_decode_cp.py).

The TP ranks are simulated on one GPU: every rank's compact score row is a
strided slice of one full score tensor, and the all-gather is a stack. The
merged top-k of every simulated rank must equal the full-context top-k.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("MiniMax-M3 MSA indexer requires CUDA.", allow_module_level=True)

from vllm.models.minimax_m3.nvidia.indexer_decode_cp import (  # noqa: E402
    TOPK,
    local_topk_keys,
    merge_topk,
)

MAX_K_TILES = 8192
NUM_INDEX_HEADS = 4
FLT_MAX = torch.finfo(torch.float32).max


def _reference_topk(
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    force_begin: int,
    force_end: int,
) -> torch.Tensor:
    """Full-context top-k, ``sparse_topk_select`` contract: per (token, head)
    the TOPK best blocks below the token's page count, init/local blocks
    forced, ascending, ``-1`` padded. Exact ties: the lower block id wins."""
    scores, num_valid_pages = scores.cpu(), num_valid_pages.cpu()
    num_tokens, num_heads, max_k_tiles = scores.shape
    out = torch.full((num_tokens, num_heads, TOPK), -1, dtype=torch.int32)
    for t in range(num_tokens):
        nvp = min(max(int(num_valid_pages[t]), 0), max_k_tiles)
        force_end_start = nvp - force_end if nvp >= force_end else 0
        blocks = torch.arange(nvp)
        forced = (blocks < force_begin) | (blocks >= force_end_start)
        for h in range(num_heads):
            row = torch.where(forced, FLT_MAX, scores[t, h, :nvp])
            # Stable sort keeps ascending block ids among equal scores.
            order = torch.sort(-row, stable=True).indices[:TOPK]
            sel = torch.sort(order).values.to(torch.int32)
            out[t, h, : sel.numel()] = sel
    return out


def _cp_topk(
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    cp_size: int,
    force_begin: int,
    force_end: int,
) -> list[torch.Tensor]:
    """Steps 3-5 of the CP decode on ``cp_size`` simulated ranks; returns each
    rank's merged top-k for the index heads it owns."""
    num_tokens, num_heads, max_k_tiles = scores.shape
    width = max_k_tiles // cp_size
    nvp = num_valid_pages.clamp(0, max_k_tiles)
    j = torch.arange(width, device=scores.device)
    keys = []
    for rank in range(cp_size):
        compact = scores[:, :, rank::cp_size].contiguous()
        # Columns past the rank's owned page count are never written by the
        # score kernel: poison them so a read would change the result.
        n_local = (nvp + (cp_size - 1 - rank)) // cp_size
        unowned = j[None, :] >= n_local[:, None]  # [T, width]
        compact.masked_fill_(unowned[:, None, :], 1e30)
        k = torch.zeros(
            num_tokens, num_heads, TOPK, dtype=torch.int64, device=scores.device
        )
        local_topk_keys(
            compact,
            num_valid_pages,
            k,
            cp_size=cp_size,
            cp_rank=rank,
            max_k_tiles=max_k_tiles,
            force_begin=force_begin,
            force_end=force_end,
        )
        keys.append(k)
    gathered = torch.stack(keys)  # the all-gather: [cp, T, H, TOPK]

    head_replicas = max(1, cp_size // num_heads)
    num_local_heads = max(1, num_heads // cp_size)
    outs = []
    for rank in range(cp_size):
        head_base = (rank // head_replicas) * num_local_heads
        out = torch.full(
            (num_tokens, num_local_heads, TOPK),
            -7,
            dtype=torch.int32,
            device=scores.device,
        )
        merge_topk(gathered, out, head_base=head_base)
        outs.append((head_base, out))
    return outs


def _make_case(kind: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(0)
    # Page counts: empty, shorter than TOPK, around the forced window, chunk
    # and group edges, the 1M-token maximum, and past it (clamped).
    nvp = torch.tensor(
        [0, 1, 3, 15, 16, 17, 129, 2048, 2049, 5078, MAX_K_TILES, MAX_K_TILES + 5],
        dtype=torch.int32,
    )
    shape = (nvp.numel(), NUM_INDEX_HEADS, MAX_K_TILES)
    if kind == "random":
        scores = torch.randn(shape, generator=gen)
    elif kind == "ties":
        # Few distinct values: exact ties straddle the TOPK-th place.
        scores = torch.randint(0, 4, shape, generator=gen).float()
    elif kind == "clustered":
        # All of a row's winners in one region (one local-top-k group).
        scores = torch.randn(shape, generator=gen) - 100.0
        scores[:, :, 300:340] += 200.0
    else:
        raise ValueError(kind)
    return scores.to(device), nvp.to(device)


@pytest.mark.parametrize("cp_size", [4, 8])
@pytest.mark.parametrize("kind", ["random", "ties", "clustered"])
@pytest.mark.parametrize(("force_begin", "force_end"), [(0, 0), (1, 2)])
def test_cp_topk_matches_full_topk(
    cp_size: int, kind: str, force_begin: int, force_end: int
):
    scores, nvp = _make_case(kind, "cuda")
    expected = _reference_topk(scores, nvp, force_begin, force_end)
    for head_base, out in _cp_topk(scores, nvp, cp_size, force_begin, force_end):
        torch.testing.assert_close(
            out.cpu(),
            expected[:, head_base : head_base + out.shape[1]],
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="fmha_sm100 sparse_topk_select requires Blackwell.",
)
@pytest.mark.parametrize("cp_size", [4, 8])
def test_cp_topk_matches_sparse_topk_select(cp_size: int):
    """Against the default decode top-k on tie-free scores."""
    api = pytest.importorskip("vllm.third_party.fmha_sm100.api")
    force_begin, force_end = 1, 2
    scores, nvp = _make_case("random", "cuda")
    stock = torch.full(
        (scores.shape[0], NUM_INDEX_HEADS, TOPK), -7, dtype=torch.int32, device="cuda"
    )
    api.sparse_topk_select(
        scores,
        TOPK,
        num_valid_pages=nvp,
        force_begin_blocks=force_begin,
        force_end_blocks=force_end,
        output=stock,
        max_score_layout="THK",
    )
    stock = torch.sort(stock, dim=-1).values
    for head_base, out in _cp_topk(scores, nvp, cp_size, force_begin, force_end):
        torch.testing.assert_close(
            torch.sort(out, dim=-1).values,
            stock[:, head_base : head_base + out.shape[1]],
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="CuteDSL index decode score requires Blackwell.",
)
@pytest.mark.parametrize("cp_size", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(("decode_query_len", "max_decode_query_len"), [(1, 1), (3, 8)])
def test_cp_score_columns_match_full_score(
    cp_size: int,
    dtype: torch.dtype,
    decode_query_len: int,
    max_decode_query_len: int,
):
    """Rank r's compact column j is bitwise the full score of block j*cp+r."""
    pytest.importorskip("cutlass")
    from vllm.models.minimax_m3.nvidia.ops import (
        minimax_m3_index_decode_score_cutedsl,
    )

    torch.manual_seed(0)
    block, head_dim = 128, 128
    seq_lens = torch.tensor((5, 1025, 4097, 9000), device="cuda", dtype=torch.int32)
    batch = seq_lens.numel()
    total_q = batch * decode_query_len
    max_seq_len = int(seq_lens.max())
    max_blocks = (max_seq_len + block - 1) // block
    num_pages = batch * max_blocks
    block_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.randn(total_q, NUM_INDEX_HEADS, head_dim, device="cuda").to(dtype)
    kv = torch.randn(num_pages, block, head_dim, device="cuda").to(dtype)

    def score(out: torch.Tensor, **cp) -> torch.Tensor:
        minimax_m3_index_decode_score_cutedsl(
            idx_q,
            kv,
            block_table,
            seq_lens,
            max_seq_len=max_seq_len,
            init_blocks=0,
            local_blocks=0,
            num_kv_heads=NUM_INDEX_HEADS,
            decode_query_len=decode_query_len,
            max_decode_query_len=max_decode_query_len,
            score_out=out.transpose(0, 1),
            **cp,
        )
        return out

    full = score(
        torch.full(
            (total_q, NUM_INDEX_HEADS, MAX_K_TILES), -float("inf"), device="cuda"
        )
    )
    num_blocks = ((seq_lens + block - 1) // block).repeat_interleave(decode_query_len)
    for rank in range(cp_size):
        compact = score(
            torch.full(
                (total_q, NUM_INDEX_HEADS, MAX_K_TILES // cp_size),
                -float("inf"),
                device="cuda",
            ),
            cp_size=cp_size,
            cp_rank=rank,
        )
        owned = (num_blocks + (cp_size - 1 - rank)) // cp_size
        for t in range(total_q):
            n = int(owned[t])
            torch.testing.assert_close(
                compact[t, :, :n],
                full[t, :, rank::cp_size][:, :n],
                rtol=0,
                atol=0,
            )
