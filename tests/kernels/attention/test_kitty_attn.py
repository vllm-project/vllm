# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Kitty backend in `vllm/v1/attention/backends/kitty_attn.py`.

`KittyMetadataBuilder.build` and `KittyAttentionImpl.forward` run a multi-step
schedule of mixed batches on one layer's cache. The K/V are values that sealing
stores exactly, so every query token must attend as dense causal attention
over its request's K/V, whatever the schedule. A step that reads a page before
it is sealed, a residual row after it is overwritten, or another request's
tokens fails the comparison.
"""

import numpy as np
import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.kitty_attn import (
    KittyMetadataBuilder,
    KittyResidualMetadata,
)
from vllm.v1.attention.ops.kitty.tiles import (
    KITTY_PAGE_SIZE,
    KITTY_RESIDUAL_SLOTS,
    kitty_d_boost,
    kitty_state_content_bytes,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="Kitty needs CUDA with compute capability >= 8.0",
)

PAGE = KITTY_PAGE_SIZE
D = 128
NUM_KV_HEADS = 2
NUM_HEADS = 8
SCALE = D**-0.5
NUM_PAD_TOKENS = 3
LAYER = "layers.0.attn"

# Each step is (decodes, continuations, fresh prefills), each a list of
# (request, query length), in the order the runner lays out the batch.
SCHEDULE = [
    # Fresh prefills only; A seals K block 1.
    ([], [], [("A", 300), ("B", 130)]),
    # C is new but sits among the continuations, as the runner's sort allows.
    # A crosses 384, sealing K block 2 and V block 1.
    ([("B", 1)], [("C", 383), ("A", 200)], [("E", 255)]),
    # E's decode seals K block 1, C's seals K block 2 and V block 1. B's
    # continuation reads its sink and tails and seals K blocks 1-2, V block 1.
    ([("E", 1), ("C", 1), ("A", 1)], [("B", 260)], []),
    # Decodes only.
    ([("A", 1), ("B", 1), ("C", 1), ("E", 1)], [], []),
    # G decodes from an empty cache. A's V tail wraps its 256-row ring.
    ([("C", 1), ("G", 1)], [("E", 300), ("A", 150)], [("F", PAGE)]),
]


def lossless_kv(length: int, dtype: torch.dtype, seed: int):
    """Random K/V `[length, H_KV, D]` that sealing stores exactly.

    Within a page, each K channel takes values in {0, 5, 10, 15} / 8 and each
    V token values in {0, 1, 2, 3} / 2, with both extremes present. These are
    exact `code * scale + min` values for 2-bit K, 4-bit (boosted) K and 2-bit
    V.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (length, NUM_KV_HEADS, D)
    k = torch.randint(0, 4, shape, generator=gen, device="cuda") * 5 / 8
    v = torch.randint(0, 4, shape, generator=gen, device="cuda") / 2
    k[0::PAGE], k[1::PAGE] = 0, 15 / 8
    v[..., 0], v[..., 1] = 0, 3 / 2
    return k.to(dtype), v.to(dtype)


def dense_causal_attention(q, k, v, old_len: int):
    """fp32 attention of queries `[n, H_Q, D]` at positions `old_len..` over
    K/V `[old_len + n, H_KV, D]`."""
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * SCALE
    q_pos = old_len + torch.arange(q.shape[0], device="cuda")
    future = torch.arange(k.shape[0], device="cuda")[None] > q_pos[:, None]
    scores.masked_fill_(future, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def batch_tokens(seqs: dict, rows: list[tuple[str, int, int]]) -> torch.Tensor:
    """Rows `[old, new)` of each request's sequence in batch order, then
    `NUM_PAD_TOKENS` NaN padding tokens."""
    seq = next(iter(seqs.values()))
    pad = torch.full((NUM_PAD_TOKENS, *seq.shape[1:]), float("nan"), dtype=seq.dtype)
    return torch.cat([*(seqs[r][o:n] for r, o, n in rows), pad.to(seq.device)])


def make_common_metadata(
    q_lens: list[int], seq_lens: list[int], block_table: torch.Tensor
) -> CommonAttentionMetadata:
    query_start_loc = torch.tensor(np.cumsum([0, *q_lens]), dtype=torch.int32)
    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc.cuda(),
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens_cpu.cuda(),
        num_reqs=len(q_lens),
        num_actual_tokens=sum(q_lens),
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
        seq_lens_cpu_upper_bound=seq_lens_cpu,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("boost_ratio", [0.0, 0.25])
def test_forward_matches_dense_attention_across_steps(dtype, boost_ratio):
    """Every step of `SCHEDULE` mixes decodes, continuations and fresh
    prefills in a new batch order, so each request moves between batch rows.
    Unwritten residual rows hold NaN, unused pages random bytes and padding
    tokens NaN."""
    total_lens: dict[str, int] = {}
    for step in SCHEDULE:
        for req, q_len in (item for group in step for item in group):
            total_lens[req] = total_lens.get(req, 0) + q_len
    names = list(total_lens)
    keys, values = {}, {}
    for seed, (r, n) in enumerate(total_lens.items()):
        keys[r], values[r] = lossless_kv(n, dtype, seed)
    gen = torch.Generator(device="cuda").manual_seed(0)
    queries = {
        r: torch.randn((n, NUM_HEADS, D), generator=gen, device="cuda").to(dtype)
        for r, n in total_lens.items()
    }

    vllm_config = VllmConfig(
        cache_config=CacheConfig(cache_dtype="kitty"),
        attention_config=AttentionConfig(kitty_boost_ratio=boost_ratio),
    )
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(dtype):
        layer = Attention(
            NUM_HEADS,
            D,
            SCALE,
            num_kv_heads=NUM_KV_HEADS,
            cache_config=vllm_config.cache_config,
            prefix=LAYER,
        )
    spec = AttentionSpec(
        block_size=PAGE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=D,
        dtype=dtype,
        kv_quant_mode=KVQuantMode.KITTY,
    )
    builder = KittyMetadataBuilder(spec, [LAYER], vllm_config, torch.device("cuda"))

    # Pages and residual blocks are assigned in shuffled order; page 0 and
    # residual block 0 are never assigned.
    d_boost = kitty_d_boost(D, boost_ratio)
    max_pages = cdiv(max(total_lens.values()), PAGE)
    num_pages = len(names) * max_pages + 1
    page_shape = (num_pages, NUM_KV_HEADS, PAGE, kitty_state_content_bytes(D, d_boost))
    packed = torch.randint(
        0, 256, page_shape, generator=gen, dtype=torch.uint8, device="cuda"
    )
    residual_shape = (len(names) + 1, NUM_KV_HEADS, KITTY_RESIDUAL_SLOTS, 2 * D)
    layer.kitty_residual.bind_kv_cache(
        torch.full(residual_shape, float("nan"), dtype=dtype, device="cuda")
    )
    pages = torch.randperm(num_pages - 1, generator=gen, device="cuda") + 1
    block_tables = dict(zip(names, pages.int().view(len(names), max_pages)))
    residual_ids = torch.randperm(len(names), generator=gen, device="cuda") + 1
    residual_blocks = dict(zip(names, residual_ids.int()))

    cache_lens = dict.fromkeys(names, 0)
    for i, (decodes, continuations, fresh) in enumerate(SCHEDULE):
        batch = decodes + continuations + fresh
        reqs = [r for r, _ in batch]
        q_lens = [n for _, n in batch]
        old_lens = [cache_lens[r] for r in reqs]
        new_lens = [o + n for o, n in zip(old_lens, q_lens)]
        rows = list(zip(reqs, old_lens, new_lens))
        common = make_common_metadata(
            q_lens, new_lens, torch.stack([block_tables[r] for r in reqs])
        )

        md = builder.build(common_prefix_len=0, common_attn_metadata=common)
        assert (md.num_decodes, md.num_continuations, md.num_fresh_prefills) == (
            len(decodes),
            len(continuations),
            len(fresh),
        ), f"step {i}"

        query = batch_tokens(queries, rows)
        output = torch.full_like(query, float("nan"))
        residual_md = KittyResidualMetadata(
            torch.stack([residual_blocks[r] for r in reqs])
        )
        attn_metadata = {LAYER: md, layer.kitty_residual.prefix: residual_md}
        with set_forward_context(attn_metadata, vllm_config):
            layer.impl.forward(
                layer,
                query,
                batch_tokens(keys, rows),
                batch_tokens(values, rows),
                packed,
                md,
                output=output,
            )

        qsl = common.query_start_loc_cpu.tolist()
        for b, (r, o, n) in enumerate(rows):
            want = dense_causal_attention(
                queries[r][o:n], keys[r][:n], values[r][:n], o
            )
            torch.testing.assert_close(
                output[qsl[b] : qsl[b + 1]].float(),
                want,
                atol=1.5e-2,
                rtol=1e-2,
                msg=lambda m, at=f"step {i}, request {r}, [{o}, {n})": f"{at}: {m}",
            )
        cache_lens.update(zip(reqs, new_lens))
