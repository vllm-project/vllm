# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global Kpool writes preserve request boundaries independently of local Q."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA Kpool kernels"
)


@pytest.mark.parametrize("decode_lengths", [(3, 1), (3, 2), (3, 3)])
def test_pcp_kpool_causal_writes_and_speculative_tail(decode_lengths):
    from vllm.models.glm5next.nvidia.ops import kpool_compress as ops
    from vllm.models.glm5next.nvidia.sparse_indexer import _write_kpool_cache
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    device = "cuda"
    pool, dim, page = 4, 128, 64
    lengths = (17, 19)  # Neither request boundary aligns with a PCP chunk.
    generator = torch.Generator(device=device).manual_seed(42)
    cache = torch.zeros(3, page, dim + 4, dtype=torch.uint8, device=device)
    tail = torch.zeros(3, 2, 8, dim, dtype=torch.bfloat16, device=device)
    expected_cache, expected_tail = cache.clone(), tail.clone()
    ape = torch.randn(pool, dim, generator=generator, device=device)

    def metadata(slots, positions, decode=None):
        return DeepseekV32IndexerMetadata(
            seq_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
            max_seq_len=max(lengths) + max(decode_lengths),
            slot_mapping=slots,
            num_decodes=0 if decode is None else 2,
            num_decode_tokens=0 if decode is None else sum(decode_lengths),
            num_prefills=2 if decode is None else 0,
            num_prefill_tokens=sum(lengths) if decode is None else 0,
            decode=decode,
            positions=positions,
        )

    for is_decode in (False, True):
        batch_lengths = decode_lengths if is_decode else lengths
        positions = torch.cat(
            [
                torch.arange(n, device=device) + (lengths[i] if is_decode else 0)
                for i, n in enumerate(batch_lengths)
            ]
        )
        owners = torch.cat(
            [
                torch.full((n,), i + 1, device=device)
                for i, n in enumerate(batch_lengths)
            ]
        )
        slots = torch.where(
            (positions + 1) % pool == 0, owners * page + positions // pool, -1
        )
        tail_slots = owners * tail.shape[2] + positions % tail.shape[2]
        # Match the strided K/gate views of a restored PCP exchange packet.
        packet = torch.randn(
            sum(batch_lengths),
            dim * 2,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        keys, gates = packet.split(dim, dim=-1)
        decode = None
        if is_decode:
            per_req = torch.tensor(batch_lengths, dtype=torch.int32, device=device)
            decode = DeepSeekV32IndexerDecodeMetadata(
                block_table=torch.tensor([[1], [2]], device=device, dtype=torch.int32),
                seq_lens=per_req,
                decode_lens=per_req,
                requires_padding=len(set(batch_lengths)) > 1,
                schedule_metadata=torch.empty(0, device=device, dtype=torch.int32),
                per_req_decode_lens=per_req,
                decode_is_uniform=len(set(batch_lengths)) == 1,
                write_max_decode_len=max(batch_lengths),
            )
        write_meta = metadata(slots, positions, decode)
        tail_meta = metadata(tail_slots, positions, decode)
        _write_kpool_cache(
            {"tail": tail_meta},
            write_meta,
            keys,
            gates,
            positions,
            ape,
            cache,
            tail,
            "tail",
            pool,
            dim,
            "ue8m0",
            dim,
            False,
            None,
        )

        offset = 0
        for request, n in enumerate(batch_lengths):
            rows = slice(offset, offset + n)
            key, gate = keys[rows].contiguous(), gates[rows].contiguous()
            if is_decode:
                # Reference one request at a time: no padded scatter or shared
                # request grouping can hide a cross-request tail dependency.
                ops.kpool_decode_update_and_maybe_write_cache_batched(
                    expected_cache,
                    expected_tail,
                    tail_slots[rows].view(1, n),
                    key.view(1, n, dim),
                    gate.view(1, n, dim),
                    ape,
                    slots[rows].view(1, n),
                    positions[rows].int().view(1, n),
                    pool,
                    dim,
                    round_scale=True,
                )
            else:
                complete = n // pool * pool
                ops.kpool_compress_and_write_cache(
                    expected_cache,
                    key[:complete].view(-1, pool, dim),
                    gate[:complete].view(-1, pool, dim),
                    ape,
                    slots[rows][pool - 1 : complete : pool].long(),
                    pool,
                    dim,
                    round_scale=True,
                )
                for pos in range(n - pool, n):
                    expected_tail[request + 1, 0, pos % tail.shape[2]] = key[pos]
                    expected_tail[request + 1, 1, pos % tail.shape[2]] = gate[pos]
            offset += n
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(tail, expected_tail, rtol=0, atol=0)
