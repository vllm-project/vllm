# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C128 prefill with shared compressed KV across adjacent query tokens."""

import torch

from vllm.triton_utils import tl, triton

try:
    import vllm._c128_C  # noqa: F401

    HAS_NATIVE_C128 = True
except ImportError:
    HAS_NATIVE_C128 = False


@triton.jit
def _pack_c128_queries(q, packed, lengths, seq_len, stride: tl.constexpr, n):
    token = tl.program_id(0)
    index = tl.arange(0, 4096)
    values = tl.load(q + token * stride + index, token < n, other=0)
    tl.store(packed + token * 4096 + index, values)
    if token % 8 == 0:
        seq = tl.load(seq_len)
        common = ((seq - n + token + 1) // 128) // 128 * 128
        tl.store(lengths + token // 8, common)


def c128_prefill_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_width: int,
    scale: float,
) -> None:
    """Attend to the causal compressed prefix and 128-token sliding window.

    Pack eight queries with eight active heads into each shared-prefix tile.
    Save unnormalized FP32 state before continuing each query's tail in CUDA,
    preserving the pinned Hopper FlashMLA accumulation and rounding order.
    The caller selects supported text prefills and otherwise uses FlashMLA.

    Args:
        q: BF16 queries, [tokens, 64, 512], with eight active heads.
        kv: Gathered BF16 KV, [1, workspace_width, 512]. The compressed
            prefix occupies [0, compressed_width), followed by gathered SWA KV.
        attn_sink: FP32 sink logits, at least eight elements.
        out: Output buffer with the same layout as q; writes the active heads.
        seq_lens: Sequence lengths including the current query tokens.
        gather_lens: Number of uncompressed KV rows gathered per request.
        compressed_width: Offset of the uncompressed region in each KV row.
        scale: Query-key softmax scale.

    """
    assert HAS_NATIVE_C128, "C128 prefill requires the native _c128_C extension"
    # Q packing runs before the C++ entry point validates the other tensors.
    assert q.is_cuda and seq_lens.device == q.device and seq_lens.numel() == 1
    assert q.ndim == 3 and q.shape[1:] == (64, 512)
    assert q.stride(1) == 512 and q.stride(2) == 1
    if q.shape[0] == 0:
        return
    num_tokens = q.shape[0]
    padded_tokens = triton.cdiv(num_tokens, 8) * 8
    packed_q = torch.empty((padded_tokens, 8, 512), device=q.device, dtype=q.dtype)
    raw_o = torch.empty((padded_tokens, 8, 512), device=q.device, dtype=torch.float32)
    raw_m = torch.empty((padded_tokens, 8), device=q.device, dtype=torch.float32)
    raw_l = torch.empty((padded_tokens, 8, 2, 4), device=q.device, dtype=torch.float32)
    lengths = torch.empty((padded_tokens // 8,), device=q.device, dtype=torch.int32)
    _pack_c128_queries[(padded_tokens,)](
        q, packed_q, lengths, seq_lens, q.stride(0), num_tokens, num_warps=4
    )
    torch.ops._c128_C.prefill(
        q,
        packed_q,
        kv,
        attn_sink,
        lengths,
        seq_lens,
        gather_lens,
        out,
        raw_o,
        raw_m,
        raw_l,
        compressed_width,
        scale * 1.4426950216293335,
    )
