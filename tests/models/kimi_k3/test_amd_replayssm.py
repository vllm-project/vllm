# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3 KDA ReplaySSM kernels vs a float32 recurrence."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 KDA ReplaySSM kernels are ROCm-only",
)

DEVICE = current_platform.device_type
NUM_HEADS = 2
HEAD_DIM = 128
LOWER_BOUND = -5.0


def _inputs(num_seqs: int, seq_len: int, seed: int):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    total = num_seqs * seq_len

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=DEVICE, dtype=torch.float32)

    return (
        randn(1, total, NUM_HEADS, HEAD_DIM),
        randn(1, total, NUM_HEADS, HEAD_DIM),
        randn(1, total, NUM_HEADS, HEAD_DIM),
        randn(1, total, NUM_HEADS, HEAD_DIM),
        randn(1, total, NUM_HEADS),
    )


def _gate(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    return LOWER_BOUND * torch.sigmoid(A_log.exp()[:, None] * (a + dt_bias))


def _reference(q, k, v, a, b, A_log, dt_bias, state, num_seqs, seq_len):
    """Output of every token, and the final state, per sequence."""
    scale = HEAD_DIM**-0.5
    out = torch.empty_like(v)
    final = state.clone()
    for seq in range(num_seqs):
        hidden = state[seq].clone()
        for token in range(seq * seq_len, (seq + 1) * seq_len):
            query = q[0, token]
            key = k[0, token]
            query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
            key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
            beta = torch.sigmoid(b[0, token])
            hidden = hidden * _gate(a[0, token], A_log, dt_bias).exp()[:, None, :]
            value = v[0, token] - (hidden * key[:, None, :]).sum(-1)
            value = value * beta[:, None]
            hidden = hidden + value[:, :, None] * key[:, None, :]
            out[0, token] = (hidden * (query * scale)[:, None, :]).sum(-1)
        final[seq] = hidden
    return out, final


def _split(x: torch.Tensor, num_seqs: int, seq_len: int, lo: int, hi: int):
    """Tokens [lo, hi) of each sequence, packed varlen."""
    rows = [x[:, s * seq_len + lo : s * seq_len + hi] for s in range(num_seqs)]
    return torch.cat(rows, dim=1).contiguous()


def _run_window(
    kernel, window, A_log, dt_bias, ckpt, bufs, write_pos, slots, window_len
):
    q, k, v, a, b = window
    num_seqs = slots.numel()
    cu_seqlens = torch.arange(
        0, (num_seqs + 1) * window_len, window_len, device=DEVICE, dtype=torch.int32
    )
    return kernel(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ckpt=ckpt,
        buf_k=bufs[0],
        buf_u=bufs[1],
        buf_g=bufs[2],
        write_pos=write_pos,
        slot_idx=slots,
        cu_seqlens=cu_seqlens,
        max_query_len=window_len,
        use_qk_l2norm_in_kernel=True,
        lower_bound=LOWER_BOUND,
    )


@pytest.mark.parametrize("cache_len", [16, 32], ids=["flush", "no-flush"])
@torch.inference_mode()
def test_replayssm_verify_window_matches_recurrence(cache_len: int) -> None:
    """A flushing window must see the records it replays, not its own appends.

    Eight primed records plus an eight-token window flush at capacity 16 and
    append at capacity 32. With V=128 the head is split across four V tiles
    that share its K/G rows, so a flush that overwrites those rows while a
    sibling tile is still replaying them shows up as an output mismatch.
    """
    from vllm.models.kimi_k3.amd.ops.third_party.replayssm import (
        replayssm_buffer_shapes,
        replayssm_sigmoid_gating_delta_rule,
    )

    num_seqs, window, repeats = 2, 8, 50
    seq_len = 2 * window
    q, k, v, a, b = _inputs(num_seqs, seq_len, seed=0)
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    A_log = torch.randn(NUM_HEADS, generator=gen, device=DEVICE)
    dt_bias = torch.randn(NUM_HEADS, HEAD_DIM, generator=gen, device=DEVICE)
    state0 = 0.1 * torch.randn(
        num_seqs, NUM_HEADS, HEAD_DIM, HEAD_DIM, generator=gen, device=DEVICE
    )
    ref_out, _ = _reference(q, k, v, a, b, A_log, dt_bias, state0, num_seqs, seq_len)
    ref_second = _split(ref_out, num_seqs, seq_len, window, seq_len)

    first = tuple(_split(x, num_seqs, seq_len, 0, window) for x in (q, k, v, a, b))
    second = tuple(
        _split(x, num_seqs, seq_len, window, seq_len) for x in (q, k, v, a, b)
    )
    # Slot 0 is the null block; sequences own slots 1..num_seqs.
    slots = torch.arange(1, num_seqs + 1, device=DEVICE, dtype=torch.int32)
    shapes = replayssm_buffer_shapes(
        cache_len, NUM_HEADS, HEAD_DIM, HEAD_DIM, is_kda=True
    )

    for _ in range(repeats):
        ckpt = torch.zeros(num_seqs + 1, NUM_HEADS, HEAD_DIM, HEAD_DIM, device=DEVICE)
        ckpt[1:] = state0
        bufs = tuple(
            torch.zeros(num_seqs + 1, *shape, device=DEVICE) for shape in shapes
        )
        write_pos = torch.zeros(num_seqs + 1, device=DEVICE, dtype=torch.int32)

        _run_window(
            replayssm_sigmoid_gating_delta_rule,
            first,
            A_log,
            dt_bias,
            ckpt,
            bufs,
            write_pos,
            slots,
            window,
        )
        write_pos[slots.long()] = window
        out = _run_window(
            replayssm_sigmoid_gating_delta_rule,
            second,
            A_log,
            dt_bias,
            ckpt,
            bufs,
            write_pos,
            slots,
            window,
        )
        torch.testing.assert_close(out, ref_second, atol=2e-3, rtol=2e-3)

    if cache_len == 16:
        key = second[1][0].view(num_seqs, window, NUM_HEADS, HEAD_DIM)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
        gate = _gate(
            second[3][0].view(num_seqs, window, NUM_HEADS, HEAD_DIM), A_log, dt_bias
        )
        torch.testing.assert_close(
            bufs[0][1:, :, :window], key.transpose(1, 2), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            bufs[2][1:, :, :window], gate.transpose(1, 2), atol=1e-5, rtol=1e-5
        )


def _make_replayssm_builder(num_speculative_tokens: int, num_slots: int):
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import SpeculativeConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.kimi_k3.amd.kda_metadata import KimiK3ROCmKDAMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    vllm_config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B", block_size=16)
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=num_speculative_tokens
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    vllm_config.cache_config.use_replayssm = True
    vllm_config.cache_config.num_gpu_blocks = num_slots
    builder = KimiK3ROCmKDAMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=16,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=0,
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=torch.device(DEVICE),
    )
    assert builder.use_kda_replayssm
    return builder


def _common(seq_lens: list[int], query_lens: list[int], is_prefilling: list[bool]):
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        16,
        torch.device(DEVICE),
        arange_block_indices=True,
    )
    # Block 0 is the null block.
    return common.replace(
        block_table_tensor=common.block_table_tensor + 1,
        is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool),
    )


@torch.inference_mode()
def test_replayssm_mixed_batch_folds_pending_plain_decode_record() -> None:
    """Prefill -> plain decode -> mixed batch must fold the plain decode's record.

    In the mixed batch the GDN builder reclassifies the plain decode as a
    prefill, so it takes the chunk path and needs its ring folded into the
    checkpoint. The record from the previous plain decode has to be committed
    first, otherwise the fold absorbs nothing and the token is lost.
    """
    from vllm.models.kimi_k3.amd.ops.third_party.replayssm import (
        replayssm_buffer_shapes,
        replayssm_fold,
        replayssm_sigmoid_gating_delta_rule,
    )

    num_spec, num_slots = 2, 16
    builder = _make_replayssm_builder(num_spec, num_slots)
    cap = builder.replayssm_cache_len
    t_max = builder.replayssm_max_query_len

    # Step A: both requests prefill. The chunk kernel would write their states.
    prefill = builder.build(
        0,
        _common([40, 20], [40, 20], [True, True]),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([-1, -1], dtype=torch.int32),
    )
    assert prefill.num_prefills == 2
    slots = prefill.prefill_state_indices.to(torch.int64)

    gen = torch.Generator(device=DEVICE).manual_seed(2)
    A_log = torch.randn(NUM_HEADS, generator=gen, device=DEVICE)
    dt_bias = torch.randn(NUM_HEADS, HEAD_DIM, generator=gen, device=DEVICE)
    state0 = 0.1 * torch.randn(
        2, NUM_HEADS, HEAD_DIM, HEAD_DIM, generator=gen, device=DEVICE
    )
    ckpt = torch.zeros(num_slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, device=DEVICE)
    ckpt[slots] = state0
    bufs = tuple(
        torch.zeros(num_slots, *shape, device=DEVICE)
        for shape in replayssm_buffer_shapes(
            cap, NUM_HEADS, HEAD_DIM, HEAD_DIM, is_kda=True
        )
    )

    # Step B: both requests plain-decode one token through the ReplaySSM kernel.
    decode = builder.build(
        0,
        _common([41, 21], [1, 1], [False, False]),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([-1, -1], dtype=torch.int32),
    )
    assert decode.num_decodes == 2 and decode.num_spec_decodes == 0
    assert decode.write_pos is not None and decode.slot_idx is not None
    torch.testing.assert_close(decode.slot_idx.to(torch.int64), slots)
    q, k, v, a, b = _inputs(2, 1, seed=3)
    replayssm_sigmoid_gating_delta_rule(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ckpt=ckpt,
        buf_k=bufs[0],
        buf_u=bufs[1],
        buf_g=bufs[2],
        write_pos=decode.write_pos,
        slot_idx=decode.slot_idx,
        cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE),
        max_query_len=t_max,
        use_qk_l2norm_in_kernel=True,
        lower_bound=LOWER_BOUND,
    )
    _, ref_state = _reference(q, k, v, a, b, A_log, dt_bias, state0, 2, 1)

    # Step C: request 0 plain-decodes again, request 1 verifies two drafts.
    mixed = builder.build(
        0,
        _common([42, 24], [1, 3], [False, False]),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device=DEVICE),
        num_decode_draft_tokens_cpu=torch.tensor([-1, 2], dtype=torch.int32),
    )
    assert mixed.num_prefills == 1 and mixed.num_spec_decodes == 1
    assert mixed.replayssm_fold_slots is not None
    assert mixed.replayssm_fold_len is not None
    torch.testing.assert_close(mixed.replayssm_fold_slots.to(torch.int64), slots[:1])
    assert mixed.replayssm_fold_len.tolist() == [1]
    assert builder.replayssm_write_pos[slots].tolist() == [0, 1]

    replayssm_fold(
        ckpt,
        bufs[0],
        bufs[1],
        bufs[2],
        mixed.replayssm_fold_len,
        mixed.replayssm_fold_slots,
    )
    torch.testing.assert_close(ckpt[slots[0]], ref_state[0], atol=2e-3, rtol=2e-3)
