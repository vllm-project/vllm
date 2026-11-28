# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph replay of FlyDSL UltraQuant decode matches eager launches.

The engine captures a decode graph that runs one FlyDSL launch per layer, and
each layer has its own query and KV cache. The graph is captured once at a
short context and replayed after the sequence lengths change in place. Every
replay must match an eager launch on the same buffers.
"""

import pytest
import torch

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    flydsl_ultraquant_decode_attention,
    is_ultraquant_flydsl_available,
    ultraquant_flydsl_decode_eligible,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import hadamard_matrix
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)

HEAD_SIZE = 256
QUERY_GROUP = 6
BLOCK_SIZE = 64
MAX_LEN = 4096


def _owned(out):
    # The launcher may return one pooled buffer for every layer, so copy each
    # result out before the next layer overwrites it.
    return out.clone()


def _paged_cache(key, value, block_size):
    batch, seq_len, kv_heads, head = key.shape
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    kv_cache = torch.zeros(
        batch * blocks_per_seq,
        block_size,
        kv_heads,
        slot_size(head),
        dtype=torch.uint8,
        device=key.device,
    )
    block_table = torch.arange(
        batch * blocks_per_seq, dtype=torch.int32, device=key.device
    ).view(batch, blocks_per_seq)
    slot_mapping = torch.arange(
        batch * blocks_per_seq * block_size, dtype=torch.int64, device=key.device
    ).view(batch, -1)[:, :seq_len]
    ultraquant_store(
        key.reshape(-1, kv_heads, head),
        value.reshape(-1, kv_heads, head),
        kv_cache,
        slot_mapping.reshape(-1),
    )
    return kv_cache, block_table


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("num_reqs", [1, 8])
def test_flydsl_graph_replay_matches_eager(width, num_reqs):
    if not ultraquant_flydsl_decode_eligible(
        head_size=HEAD_SIZE,
        num_kv_groups=QUERY_GROUP,
        has_sinks=False,
        sliding_window=None,
        flydsl_loaded=is_ultraquant_flydsl_available(),
    ):
        pytest.skip("UltraQuant D=256 FlyDSL kernel is unavailable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    layers = 4
    scale = HEAD_SIZE**-0.5
    hadamard = hadamard_matrix(HEAD_SIZE, device)
    set_random_seed(0x67A9 + width * 7 + num_reqs)

    caches = []
    for _ in range(layers):
        key = torch.randn(num_reqs, MAX_LEN, 1, HEAD_SIZE, dtype=dtype, device=device)
        value = torch.randn_like(key)
        caches.append(_paged_cache(key, value, BLOCK_SIZE))
    queries = [
        torch.randn(
            num_reqs * width, QUERY_GROUP, HEAD_SIZE, dtype=dtype, device=device
        )
        for _ in range(layers)
    ]
    seq_lens = torch.full((num_reqs,), 1000, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(
        0, (num_reqs + 1) * width, width, dtype=torch.int32, device=device
    )

    def run_all():
        outs = []
        for li in range(layers):
            kv_cache, block_table = caches[li]
            outs.append(
                _owned(
                    flydsl_ultraquant_decode_attention(
                        query=queries[li],
                        kv_cache=kv_cache,
                        block_table=block_table,
                        seq_lens=seq_lens,
                        scale=scale,
                        PiT=hadamard,
                        max_seq_len=MAX_LEN,
                        query_start_loc=query_start_loc if width > 1 else None,
                        num_verify_tokens=width,
                    )
                )
            )
        return outs

    # Warm up eagerly, then capture, as the engine does.
    run_all()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_all()
    torch.cuda.synchronize()

    for seq_len in (1000, 2100, 3000):
        seq_lens.fill_(seq_len)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed = [out.clone() for out in captured]
        eager = run_all()
        torch.cuda.synchronize()
        for layer, (got, want) in enumerate(zip(replayed, eager)):
            assert not torch.isnan(got.float()).any(), (seq_len, layer)
            cos = torch.nn.functional.cosine_similarity(
                got.float().reshape(-1), want.float().reshape(-1), dim=0
            ).item()
            assert cos > 0.9999, (seq_len, layer, cos)
