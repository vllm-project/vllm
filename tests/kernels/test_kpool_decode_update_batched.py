# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the batched kpool decode-update kernel.

Validates ``kpool_decode_update_and_maybe_write_cache_batched`` against an
independent pure-torch reference that replicates the per-request, in-position
order semantics: stash each token into a paged tail ring; on pool completion
(``pos % pool_size == pool_size-1``) softmax(gate+ape)-weighted sum + Hadamard-128
+ per-vector fp8 absmax quant + write to the indexer K cache. Covers
no-completion, completion-at-end, completion-mid-batch, non-uniform padding,
plain decode, plus a randomized fuzz pass.

The kernel iterates each request's ``next_n`` tokens in position order inside
one program (grid = num_requests) to preserve the pool-completion
read-after-stash dependency; the reference mirrors that ordering.

``test_decode_writer_matches_prefill_writer`` is deliberately NOT
reference-based: it checks the decode writer against the *prefill* writer
(``kpool_compress_and_write_cache``), the invariant that actually matters in
production. A hand-written reference can drift to match a buggy kernel -- that
is exactly how the stash-gating bug (intra-pool tokens never entering the tail
ring, because the stash was gated on the pool-granular ``slot_mapping``) stayed
green here.
"""

import math

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.models.glm5next.amd.ops.kpool_compress import (
        kpool_compress_and_write_cache,
        kpool_decode_update_and_maybe_write_cache_batched,
        kpool_seed_tail_cache,
    )
else:
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_compress_and_write_cache,
        kpool_decode_update_and_maybe_write_cache_batched,
        kpool_seed_tail_cache,
    )

HEAD_DIM = 128
POOL_SIZE = 16
PAGE_SIZE = 64
NUM_BLOCKS = 32
ROUND_SCALE = True
FP8_DTYPE = current_platform.fp8_dtype()
FP8_MAX = torch.finfo(FP8_DTYPE).max


def _make_caches():
    kv = torch.zeros(
        NUM_BLOCKS, PAGE_SIZE, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    tail = torch.zeros(
        NUM_BLOCKS, 2, POOL_SIZE, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    return kv, tail


def _tail_slot_for(blocks, pos):
    """tail_slot = block*POOL + pos%POOL; each request owns a distinct tail block."""
    blk = torch.tensor(blocks, device=pos.device, dtype=torch.int32).unsqueeze(1)
    return (blk * POOL_SIZE + pos % POOL_SIZE).to(torch.int32)


def _seed_prior(tail, blocks, n_prior, seed=42):
    if n_prior <= 0:
        return
    g = torch.Generator(device=tail.device).manual_seed(seed)
    prior_k = torch.randn(
        len(blocks),
        n_prior,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=tail.device,
        generator=g,
    )
    prior_s = torch.randn(
        len(blocks),
        n_prior,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=tail.device,
        generator=g,
    )
    for i, blk in enumerate(blocks):
        tail[blk, 0, :n_prior, :] = prior_k[i]
        tail[blk, 1, :n_prior, :] = prior_s[i]


def _hadamard128_torch(x: torch.Tensor) -> torch.Tensor:
    """Reference Hadamard-128 on the last dim (must be 128)."""
    n = x.shape[-1]
    assert n == 128
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32, device=x.device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    h = h / math.sqrt(n)
    return x @ h


def _torch_reference(
    kv: torch.Tensor,
    tail: torch.Tensor,
    tail_slot: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    slot_map: torch.Tensor,
    pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent reference for the batched decode-update kernel.

    For each request, iterate its next_n tokens in order. On each token:
      - if pos%POOL == POOL-1 and pos_valid: compress the pool (slots
        [pool_start..pool_start+POOL-1], current token via is_current) and write
        fp8 K + fp32 scale to kv_cache at cache_loc.
      - always stash the current token's K/score into tail[block, pos%POOL].
    """
    kv = kv.clone()
    tail = tail.clone()
    B, next_n = pos.shape
    # The indexer K cache is [num_blocks, PAGE_SIZE, HEAD_DIM+4] uint8 but the
    # kernels interpret each page as [HEAD_DIM*PAGE_SIZE bytes of K (token-major)
    # | 4*PAGE_SIZE bytes of fp32 scale (token-major)]. Operate on a flat byte
    # view so the reference writes K and scale at the exact offsets the kernel
    # uses (page_base + tok*HEAD_DIM for K; page_base + HEAD_DIM*PAGE_SIZE +
    # tok*4 for scale).
    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    k_region = HEAD_DIM * PAGE_SIZE
    tail_slot_cpu = tail_slot.cpu().tolist()
    slot_map_cpu = slot_map.cpu().tolist()
    pos_cpu = pos.cpu().tolist()
    key_cpu = key.float().cpu()
    score_cpu = score.float().cpu()
    ape_cpu = ape.cpu()
    tail_cpu = tail.float().cpu()
    kv_flat = kv.view(torch.uint8).reshape(-1).cpu()

    for b in range(B):
        for t in range(next_n):
            cache_loc = slot_map_cpu[b][t]
            p = pos_cpu[b][t]
            pos_valid = cache_loc >= 0 and p >= 0
            safe_pos = max(p, 0)
            slot = safe_pos % POOL_SIZE
            phys_slot = safe_pos % POOL_SIZE
            # Per-token block derivation (a leading invalid sentinel must not
            # poison the base for the rest of the request); clamped like the
            # kernel so an invalid entry can't form a negative base.
            block = max(tail_slot_cpu[b][t], 0) // POOL_SIZE

            cur_key = key_cpu[b, t]
            cur_score = score_cpu[b, t]

            if pos_valid and slot == POOL_SIZE - 1:
                pool_logical_start = safe_pos - slot
                pool_scores = []
                pool_ks = []
                for ps in range(POOL_SIZE):
                    is_current = ps == slot
                    phys = (pool_logical_start + ps) % POOL_SIZE
                    if is_current:
                        s = cur_score
                        k = cur_key
                    else:
                        s = tail_cpu[block, 1, phys]
                        k = tail_cpu[block, 0, phys]
                    s = s + ape_cpu[ps]
                    pool_scores.append(s)
                    pool_ks.append(k)
                pool_scores = torch.stack(pool_scores)  # [POOL, D]
                pool_ks = torch.stack(pool_ks)  # [POOL, D]
                max_score = pool_scores.max(dim=0).values
                prob = torch.exp(pool_scores - max_score)
                denom = prob.sum(dim=0)
                acc = (pool_ks * prob).sum(dim=0)
                x = (acc / denom).to(torch.bfloat16).to(torch.float32)
                x = _hadamard128_torch(x).to(torch.bfloat16).to(torch.float32)
                absmax = torch.clamp(x.abs().max(), min=1e-4)
                if ROUND_SCALE:
                    scale = torch.exp2(torch.ceil(torch.log2(absmax / FP8_MAX)))
                else:
                    scale = absmax / FP8_MAX
                quantized = torch.clamp(x / scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
                # write K and scale at the separated-layout offsets
                loc = cache_loc
                loc_page_index = loc // PAGE_SIZE
                loc_tok = loc % PAGE_SIZE
                page_base = loc_page_index * page_bytes
                if current_platform.is_rocm():
                    dims = torch.arange(HEAD_DIM)
                    k_off = (
                        page_base
                        + (loc_tok // 16) * 16 * HEAD_DIM
                        + (dims // 16) * 16 * 16
                        + (loc_tok % 16) * 16
                        + dims % 16
                    )
                else:
                    k_off = page_base + loc_tok * HEAD_DIM + torch.arange(HEAD_DIM)
                s_off = page_base + k_region + loc_tok * 4
                kv_flat[k_off] = quantized.view(torch.uint8)
                kv_flat[s_off : s_off + 4] = scale.detach().reshape(1).view(torch.uint8)

            # stash -- gated on the TOKEN-granular tail slot, not on pos_valid.
            # pos_valid keys off the pool-granular cache_loc, which is -1 for
            # every token that is not the pool's last, so gating the stash on it
            # would drop all intra-pool tokens.
            if p >= 0 and tail_slot_cpu[b][t] >= 0:
                tail_cpu[block, 0, phys_slot] = cur_key
                tail_cpu[block, 1, phys_slot] = cur_score

    kv_out = kv_flat.view(NUM_BLOCKS, PAGE_SIZE, HEAD_DIM + 4).to(device="cuda")
    return kv_out, tail_cpu.to(torch.bfloat16).to(device="cuda")


def _assert_eq(r_ref, r_kern):
    kv_ref, tail_ref = r_ref
    kv_kern, tail_kern = r_kern
    assert torch.equal(kv_ref, kv_kern), (
        "kv_cache differs: max diff "
        f"{(kv_ref.int() - kv_kern.int()).abs().max().item()}"
    )
    assert torch.equal(tail_ref, tail_kern), (
        "tail_kv_cache differs: max diff "
        f"{(tail_ref.float() - tail_kern.float()).abs().max().item()}"
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_amd_prefill_writer_uses_preshuffled_cache_layout():
    from vllm.models.glm5next.amd.ops.kpool_compress import (
        kpool_compress_and_write_cache as amd_kpool_compress,
    )

    torch.manual_seed(0)
    token_offset = 17
    kv = torch.zeros(1, PAGE_SIZE, HEAD_DIM + 4, dtype=torch.uint8, device="cuda")
    key = torch.randn(1, POOL_SIZE, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    score = torch.randn_like(key)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, dtype=torch.float32, device="cuda")
    compressed_k, compressed_scale = amd_kpool_compress(
        kv,
        key,
        score,
        ape,
        torch.tensor([token_offset], dtype=torch.int64, device="cuda"),
        pool_size=POOL_SIZE,
        head_dim=HEAD_DIM,
        round_scale=ROUND_SCALE,
        return_compressed=True,
    )

    dim = torch.arange(HEAD_DIM, device="cuda")
    offsets = (
        (token_offset // 16) * 16 * HEAD_DIM
        + (dim // 16) * 16 * 16
        + (token_offset % 16) * 16
        + dim % 16
    )
    flat = kv.view(torch.uint8).reshape(-1)
    stored_k = flat[offsets].view(compressed_k.dtype)
    scale_offset = PAGE_SIZE * HEAD_DIM + token_offset * 4
    stored_scale = flat[scale_offset : scale_offset + 4].view(torch.float32)

    assert torch.equal(stored_k, compressed_k[0])
    assert torch.equal(stored_scale, compressed_scale)


def _run_kernel(kv, tail, tail_slot, key, score, ape, slot_map, pos):
    kv = kv.clone()
    tail = tail.clone()
    kpool_decode_update_and_maybe_write_cache_batched(
        kv,
        tail,
        tail_slot,
        key,
        score,
        ape,
        slot_map,
        pos,
        POOL_SIZE,
        HEAD_DIM,
        round_scale=ROUND_SCALE,
    )
    return kv, tail


@pytest.mark.parametrize("pool_size", [4, 16])
def test_decode_writer_matches_prefill_writer(pool_size):
    """Compare production decode and prefill writers for pool sizes 4 and 16."""
    n_pools, page, nblk = 8, 64, 4
    n_tok = n_pools * pool_size
    dev = "cuda"
    torch.manual_seed(0)
    k = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    score = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(pool_size, HEAD_DIM, dtype=torch.float32, device=dev)

    kv_prefill = torch.zeros(nblk, page, HEAD_DIM + 4, dtype=torch.uint8, device=dev)
    kpool_compress_and_write_cache(
        kv_prefill,
        k.view(n_pools, pool_size, HEAD_DIM),
        score.view(n_pools, pool_size, HEAD_DIM),
        ape,
        torch.arange(n_pools, dtype=torch.int64, device=dev),
        pool_size=pool_size,
        head_dim=HEAD_DIM,
        round_scale=ROUND_SCALE,
    )

    # One request owning tail block 0, fed one token per decode step.
    kv_decode = torch.zeros_like(kv_prefill)
    tail = torch.zeros(nblk, 2, pool_size, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    for t in range(n_tok):
        completes = t % pool_size == pool_size - 1
        kpool_decode_update_and_maybe_write_cache_batched(
            kv_decode,
            tail,
            # token-granular: every token has a valid tail slot
            torch.tensor([[t % pool_size]], dtype=torch.int32, device=dev),
            k[t].view(1, 1, HEAD_DIM),
            score[t].view(1, 1, HEAD_DIM),
            ape,
            # pool-granular: only the pool's last token carries a cache slot
            torch.tensor(
                [[t // pool_size if completes else -1]], dtype=torch.int32, device=dev
            ),
            torch.tensor([[t]], dtype=torch.int32, device=dev),
            pool_size,
            HEAD_DIM,
            round_scale=ROUND_SCALE,
        )

    differing = [
        p
        for p in range(n_pools)
        if not torch.equal(
            kv_prefill[p // page, p % page], kv_decode[p // page, p % page]
        )
    ]
    assert not differing, (
        f"decode-written pools differ from prefill-written pools: "
        f"{len(differing)}/{n_pools} (pool_size={pool_size}, first={differing[:5]})"
    )


def test_leading_invalid_tail_slot():
    """A request whose FIRST token carries an invalid (-1) tail slot while a
    later token is a real pool completion.

    The tail block must be derived per token, not from token 0: a leading
    invalid sentinel would otherwise poison the base address for the whole
    request (out-of-bounds tail reads on the completion).
    """
    torch.manual_seed(0)
    B, next_n, blocks = 2, 4, [3, 5]
    # req 0: token 0 invalid (pos -1), tokens 1..3 valid, completion at pos 15
    # req 1: all valid, no completion
    pos = torch.tensor(
        [[-1, 13, 14, 15], [4, 5, 6, 7]], dtype=torch.int32, device="cuda"
    )
    safe_pos = torch.where(pos >= 0, pos, 0)
    tail_slot = _tail_slot_for(blocks, safe_pos)
    # leading invalid entry carries the -1 sentinel, as the scatter path emits
    tail_slot[0, 0] = -1
    slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="cuda")
    slot_map[0, 3] = 15  # req 0 completes its pool on the last verify token

    key = torch.randn(B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    score = torch.randn(B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    ape = torch.randn(POOL_SIZE, HEAD_DIM, dtype=torch.float32, device="cuda")

    kv, tail = _make_caches()
    _seed_prior(tail, blocks, 13)
    r_ref = _torch_reference(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    r_kern = _run_kernel(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    _assert_eq(r_ref, r_kern)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_amd_prefill_seed_honors_padded_tail_block_stride():
    """The tail shares a padded indexer allocation in production."""
    kpool = 4
    num_blocks = 6
    logical_block_elems = 2 * kpool * HEAD_DIM
    padded_block_elems = logical_block_elems + 256
    sentinel = -123.0
    backing = torch.full(
        (num_blocks * padded_block_elems,),
        sentinel,
        dtype=torch.bfloat16,
        device="cuda",
    )
    tail = torch.as_strided(
        backing,
        size=(num_blocks, 2, kpool, HEAD_DIM),
        stride=(padded_block_elems, kpool * HEAD_DIM, HEAD_DIM, 1),
    )

    block = 3
    ring_slot = 2
    key = torch.arange(HEAD_DIM, dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    score = (key + 256).to(torch.bfloat16)
    tail_slot = torch.tensor(
        [block * kpool + ring_slot], dtype=torch.int32, device="cuda"
    )

    kpool_seed_tail_cache(tail, key, score, tail_slot, kpool, HEAD_DIM)
    torch.accelerator.synchronize()

    assert torch.equal(tail[block, 0, ring_slot], key[0])
    assert torch.equal(tail[block, 1, ring_slot], score[0])

    compact_offset = (block * 2 * kpool + ring_slot) * HEAD_DIM
    assert torch.all(backing[compact_offset : compact_offset + HEAD_DIM] == sentinel)


@pytest.mark.parametrize(
    "case_id",
    [
        "no_completion",
        "completion_at_end",
        "completion_mid_batch",
        "non_uniform_padding",
        "plain_decode",
    ],
)
def test_batched_matches_reference(case_id):
    torch.manual_seed(0)
    if case_id == "no_completion":
        B, next_n, blocks = 3, 4, [0, 1, 2]
        pos = (
            torch.arange(next_n, device="cuda", dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="cuda")
        n_prior = 0
    elif case_id == "completion_at_end":
        B, next_n, blocks = 2, 4, [0, 1]
        pos = torch.tensor(
            [[12, 13, 14, 15], [12, 13, 14, 15]], dtype=torch.int32, device="cuda"
        )
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="cuda")
        slot_map[:, 3] = torch.tensor(
            [15, PAGE_SIZE + 15], dtype=torch.int32, device="cuda"
        )
        n_prior = POOL_SIZE - next_n
    elif case_id == "completion_mid_batch":
        B, next_n, blocks = 3, 4, [0, 1, 2]
        pos = torch.tensor([[13, 14, 15, 16]] * B, dtype=torch.int32, device="cuda")
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="cuda")
        slot_map[:, 2] = torch.tensor(
            [15, PAGE_SIZE + 15, 2 * PAGE_SIZE + 15], dtype=torch.int32, device="cuda"
        )
        n_prior = 13
    elif case_id == "non_uniform_padding":
        B, next_n, blocks = 2, 4, [0, 1]
        pos = torch.tensor(
            [[12, 13, 14, 15], [12, 13, -1, -1]], dtype=torch.int32, device="cuda"
        )
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="cuda")
        slot_map[0, 3] = 15
        n_prior = POOL_SIZE - 4
    else:  # plain_decode
        B, next_n, blocks = 4, 1, [0, 1, 2, 3]
        pos = torch.tensor([[5], [6], [7], [8]], dtype=torch.int32, device="cuda")
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="cuda")
        n_prior = 0

    if case_id == "non_uniform_padding":
        safe_pos = torch.where(pos >= 0, pos, 0)
        tail_slot = torch.where(pos >= 0, _tail_slot_for(blocks, safe_pos), 0)
    else:
        tail_slot = _tail_slot_for(blocks, pos)

    key = torch.randn(B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    score = torch.randn(B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    ape = torch.randn(POOL_SIZE, HEAD_DIM, dtype=torch.float32, device="cuda")

    kv, tail = _make_caches()
    _seed_prior(tail, blocks, n_prior)
    r_ref = _torch_reference(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    r_kern = _run_kernel(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    _assert_eq(r_ref, r_kern)


@pytest.mark.parametrize("seed", list(range(20)))
def test_batched_matches_reference_fuzz(seed):
    """Random B / next_n / start positions; covers 0, 1, and multi completion."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = int(torch.randint(1, 6, (1,), generator=g, device="cuda").item())
    next_n = int(torch.randint(1, 8, (1,), generator=g, device="cuda").item())
    blocks = list(range(B))

    starts = torch.randint(0, 33, (B,), generator=g, device="cuda", dtype=torch.int32)
    pos = starts.unsqueeze(1) + torch.arange(
        next_n, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    tail_slot = _tail_slot_for(blocks, pos)

    is_completion = pos % POOL_SIZE == POOL_SIZE - 1
    blk = torch.tensor(blocks, device="cuda", dtype=torch.int32).unsqueeze(1)
    pool_slot = blk * PAGE_SIZE + (POOL_SIZE - 1)
    slot_map = torch.where(is_completion, pool_slot, torch.full_like(pos, -1))

    key = torch.randn(
        B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g
    )
    score = torch.randn(
        B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=g
    )
    ape = torch.randn(
        POOL_SIZE, HEAD_DIM, dtype=torch.float32, device="cuda", generator=g
    )

    kv, tail = _make_caches()
    prior_g = torch.Generator(device="cuda").manual_seed(seed + 1000)
    for b in range(B):
        n_prior = int(starts[b].item()) % POOL_SIZE
        if n_prior > 0:
            pk = torch.randn(
                n_prior,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
                generator=prior_g,
            )
            ps = torch.randn(
                n_prior,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
                generator=prior_g,
            )
            tail[blocks[b], 0, :n_prior, :] = pk
            tail[blocks[b], 1, :n_prior, :] = ps

    r_ref = _torch_reference(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    r_kern = _run_kernel(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    _assert_eq(r_ref, r_kern)
