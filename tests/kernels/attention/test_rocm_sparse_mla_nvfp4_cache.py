# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the DeepSeek-V4.1 NVFP4 compressed KV record in ROCm sparse
MLA decode, against a torch softmax over the rows the record stores."""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only used by ROCm"
)

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import _ON_GFX950

        return bool(_ON_GFX950)
    except Exception:
        return False


def _pack_nvfp4(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[N, 512]`` to the 288 B record: 256 bytes of e2m1 pairs, even element
    in the low nibble, then 32 e4m3 scales of 16 dims each."""
    levels = torch.tensor(E2M1, device=rows.device)
    tiles = rows.reshape(-1, 32, 16)
    scales = (tiles.abs().amax(-1) / 6.0).clamp(2**-9, 448.0).to(torch.float8_e4m3fn)
    scaled = (tiles / scales.float()[..., None]).reshape(rows.shape[0], -1)
    codes = (scaled.abs().unsqueeze(-1) - levels).abs().argmin(-1)
    nibbles = (codes | (scaled < 0).to(torch.int64) * 8).to(torch.uint8)
    stored = (levels[codes] * torch.sign(scaled)).reshape(-1, 32, 16)
    return (
        nibbles[:, 0::2] | (nibbles[:, 1::2] << 4),
        scales.view(torch.uint8),
        (stored * scales.float()[..., None]).reshape(rows.shape[0], -1),
    )


def _pack_v4(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[N, 512]`` to the 584 B record: 448 fp8 e4m3 NoPE with 7 UE8M0 scales
    of 64 dims (plus a pad byte), then 64 bf16 RoPE."""
    nope, rope = rows[:, :NOPE_DIM], rows[:, NOPE_DIM:].to(torch.bfloat16)
    tiles = nope.reshape(-1, NOPE_DIM // 64, 64)
    exponents = torch.ceil(torch.log2((tiles.abs().amax(-1) / 448.0).clamp_min(1e-4)))
    factors = torch.exp2(exponents)
    quantized = (tiles / factors[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    data = torch.zeros(rows.shape[0], 576, dtype=torch.uint8, device=rows.device)
    data[:, :NOPE_DIM] = quantized.reshape(-1, NOPE_DIM).view(torch.uint8)
    data[:, NOPE_DIM:] = rope.view(torch.uint8)
    scales = torch.zeros(rows.shape[0], 8, dtype=torch.uint8, device=rows.device)
    scales[:, : NOPE_DIM // 64] = (exponents + 127.0).to(torch.uint8)
    stored = (quantized.float() * factors[..., None]).reshape(-1, NOPE_DIM)
    return data, scales, torch.cat([stored, rope.float()], dim=1)


def _paged(data: torch.Tensor, scales: torch.Tensor, block_size: int) -> torch.Tensor:
    """A page holds its whole data region ahead of its whole scale region."""
    num_blocks = data.shape[0] // block_size
    width = data.shape[1] + scales.shape[1]
    cache = torch.empty(
        num_blocks, block_size * width, dtype=torch.uint8, device=data.device
    )
    split = block_size * data.shape[1]
    cache[:, :split] = data.reshape(num_blocks, -1)
    cache[:, split:] = scales.reshape(num_blocks, -1)
    return cache.view(num_blocks, block_size, width)


@pytest.mark.skipif(
    not _on_gfx950(), reason="The NVFP4 record is read by the gfx950 sparse decode"
)
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("num_heads", [16, 128])
@pytest.mark.parametrize("compressed_record", [288, 584])
def test_compressed_record_matches_reference(
    num_tokens: int, num_heads: int, compressed_record: int
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode

    set_random_seed(0)
    device = torch.device("cuda")
    block_size, num_slots = 64, 512
    num_compressed, num_swa = 64, 32
    scale = 1.0 / math.sqrt(HEAD_DIM)
    pack = _pack_nvfp4 if compressed_record == 288 else _pack_v4

    data, scales, compressed_stored = pack(
        torch.randn(num_slots, HEAD_DIM, device=device) * 0.7
    )
    compressed_cache = _paged(data, scales, block_size)
    assert compressed_cache.shape[-1] == compressed_record
    data, scales, swa_stored = _pack_v4(
        torch.randn(num_slots, HEAD_DIM, device=device) * 0.7
    )
    swa_cache = _paged(data, scales, block_size)

    query = torch.randn(
        num_tokens, num_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    sinks = torch.randn(num_heads, device=device, dtype=torch.float32)
    output = torch.empty_like(query)
    compressed_indices = torch.stack(
        [
            torch.randperm(num_slots, device=device)[:num_compressed]
            for _ in range(num_tokens)
        ]
    ).int()
    swa_indices = torch.stack(
        [torch.randperm(num_slots, device=device)[:num_swa] for _ in range(num_tokens)]
    ).int()

    rocm_sparse_attn_decode(
        q=query,
        kv_cache=compressed_cache,
        swa_k_cache=swa_cache,
        swa_only=False,
        topk_indices=compressed_indices,
        topk_lens=torch.full(
            (num_tokens,), num_compressed, device=device, dtype=torch.int32
        ),
        swa_indices=swa_indices,
        swa_lens=torch.full((num_tokens,), num_swa, device=device, dtype=torch.int32),
        swa_ragged_indices=None,
        swa_ragged_indptr=None,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=sinks,
        scale=scale,
        head_dim=HEAD_DIM,
        nope_head_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
        output=output,
    )

    for token in range(num_tokens):
        keys = torch.cat(
            [
                swa_stored[swa_indices[token].long()],
                compressed_stored[compressed_indices[token].long()],
            ]
        ).double()
        logits = query[token].double() @ keys.T * scale
        weights = torch.cat([logits, sinks[:, None].double()], dim=-1).softmax(dim=-1)
        expected = weights[:, :-1] @ keys
        torch.testing.assert_close(
            output[token].double(), expected, rtol=2e-2, atol=2e-2
        )
