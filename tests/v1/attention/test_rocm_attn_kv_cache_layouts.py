# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_ATTN in every KV cache layout it declares.

The shared backend harness lays the cache out as [B, num_kv_heads, N, 2 * hs],
while ROCM_ATTN binds [B, 2, N, num_kv_heads * hs] (K and V as two head
groups), so this drives the backend's own cache writer and attention on a
per-layer view in each layout and checks them against SDPA.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCM_ATTN is ROCm-only", allow_module_level=True)

from vllm.v1.attention.backends.rocm_attn import (  # noqa: E402
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
)

NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE = 16, 4, 128
NUM_BLOCKS = 64
# (context_len, query_len) per sequence: decodes, a DFlash-style 8-token
# query and a prefill.
SEQS = [(37, 1), (0, 8), (100, 8), (21, 1), (0, 45)]


class _Layer:
    def __init__(self, device):
        self._q_scale = torch.ones(1, device=device)
        self._k_scale = torch.ones(1, device=device)
        self._v_scale = torch.ones(1, device=device)
        self._q_scale_float = self._k_scale_float = self._v_scale_float = 1.0


def _kv_cache_view(layout, block_size, dtype, device):
    """The per-layer [B, 2, N, H * hs] view bound to the layer under layout.

    Layouts with the layer inside the block get two layers, so the view keeps
    the gaps between one layer's blocks."""
    logical = (
        1 if layout.is_layer_compact else 2,
        NUM_BLOCKS,
        2,
        block_size,
        NUM_KV_HEADS * HEAD_SIZE,
    )
    physical = torch.zeros(
        [logical[i] for i in layout.stride_order], dtype=dtype, device=device
    )
    return physical.permute(*[layout.stride_order.index(i) for i in range(5)])[0]


def _reference(q, k, v, causal):
    out = []
    for i, (ctx, qlen) in enumerate(SEQS):
        qi = q[i].transpose(0, 1)
        ki = k[i].repeat_interleave(NUM_HEADS // NUM_KV_HEADS, 1).transpose(0, 1)
        vi = v[i].repeat_interleave(NUM_HEADS // NUM_KV_HEADS, 1).transpose(0, 1)
        mask = None
        if causal:
            mask = torch.ones(qlen, ctx + qlen, dtype=torch.bool, device=q[i].device)
            mask = mask.tril(ctx)
        o = F.scaled_dot_product_attention(
            qi.float(), ki.float(), vi.float(), attn_mask=mask, scale=HEAD_SIZE**-0.5
        )
        out.append(o.transpose(0, 1))
    return torch.cat(out)


@pytest.mark.parametrize(
    "layout", RocmAttentionBackend.supported_kv_cache_layouts(), ids=lambda x: x.name
)
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_rocm_attn_kv_cache_layouts(layout, block_size, causal, kv_cache_dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    fp8 = kv_cache_dtype == "fp8"
    layer = _Layer(device)
    impl = RocmAttentionImpl(
        NUM_HEADS,
        HEAD_SIZE,
        HEAD_SIZE**-0.5,
        NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
    )
    kv_cache = _kv_cache_view(layout, block_size, torch.uint8 if fp8 else dtype, device)

    q = [
        torch.randn(qlen, NUM_HEADS, HEAD_SIZE, dtype=dtype, device=device)
        for _, qlen in SEQS
    ]
    k = [
        torch.randn(ctx + qlen, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype, device=device)
        for ctx, qlen in SEQS
    ]
    v = [torch.randn_like(x) for x in k]
    if fp8:
        # The reference sees the values the cache stores.
        fp8_dtype = current_platform.fp8_dtype()
        k = [x.to(fp8_dtype).to(dtype) for x in k]
        v = [x.to(fp8_dtype).to(dtype) for x in v]

    # Scattered blocks; block 0 is the null block.
    perm = (torch.randperm(NUM_BLOCKS - 1) + 1).tolist()
    max_blocks = max((c + n + block_size - 1) // block_size for c, n in SEQS)
    block_table = torch.zeros(len(SEQS), max_blocks, dtype=torch.int32)
    for i, (ctx, qlen) in enumerate(SEQS):
        n = (ctx + qlen + block_size - 1) // block_size
        block_table[i, :n] = torch.tensor([perm.pop() for _ in range(n)])

    def slots(i, start, end):
        pos = torch.arange(start, end)
        return block_table[i, pos // block_size].long() * block_size + pos % block_size

    # Write the context with the backend's own writer, as earlier steps would.
    ctx_slots = torch.cat([slots(i, 0, c) for i, (c, _) in enumerate(SEQS)])
    impl.do_kv_cache_update(
        layer,
        torch.cat([x[:c] for x, (c, _) in zip(k, SEQS)]),
        torch.cat([x[:c] for x, (c, _) in zip(v, SEQS)]),
        kv_cache,
        ctx_slots.to(device),
    )

    new_k = torch.cat([x[c:] for x, (c, _) in zip(k, SEQS)])
    new_v = torch.cat([x[c:] for x, (c, _) in zip(v, SEQS)])
    new_slots = torch.cat([slots(i, c, c + n) for i, (c, n) in enumerate(SEQS)])
    impl.do_kv_cache_update(layer, new_k, new_v, kv_cache, new_slots.to(device))

    query_lens = torch.tensor([n for _, n in SEQS], dtype=torch.int32)
    seq_lens = torch.tensor([c + n for c, n in SEQS], dtype=torch.int32)
    query_start_loc = torch.zeros(len(SEQS) + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(0)
    metadata = RocmAttentionMetadata(
        num_actual_tokens=int(query_lens.sum()),
        max_query_len=int(query_lens.max()),
        query_start_loc=query_start_loc.to(device),
        max_seq_len=int(seq_lens.max()),
        seq_lens=seq_lens.to(device),
        block_table=block_table.to(device),
        slot_mapping=new_slots.to(device),
        causal=causal,
    )
    query = torch.cat(q)
    output = torch.empty_like(query)
    impl.forward(layer, query, new_k, new_v, kv_cache, metadata, output)

    torch.testing.assert_close(
        output.float(), _reference(q, k, v, causal), atol=2e-2, rtol=2e-2
    )
