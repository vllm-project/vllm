# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy test for the AITER ASM MLA prefill backend."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


def _asm_prefill_available() -> bool:
    from vllm.v1.attention.backends.mla.prefill.aiter_asm import (
        AiterAsmPrefillBackend,
    )

    return AiterAsmPrefillBackend.is_available()


pytestmark = pytest.mark.skipif(
    not _asm_prefill_available(),
    reason="The AITER ASM MLA prefill backend needs gfx950 (MI355) + an AITER "
    "build exporting mla_prefill_ps_asm_fwd / mla_reduce_v1",
)

# Test both padded and unpadded head counts
HEAD_COUNTS = [12, 16, 24]
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
V_HEAD_DIM = 128
KV_LORA_RANK = 512
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)

# Ok with fp8 (e4m3) rounding noise of q/k/v
ATOL, RTOL = 1e-1, 5e-2

ATTN_OUT_DTYPE = torch.bfloat16

MAX_NUM_BATCHED_TOKENS = 2048
MAX_NUM_SEQS = 4


@pytest.fixture(autouse=True)
def _workspace_manager():
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    init_workspace_manager(torch.device("cuda"))
    yield
    reset_workspace_manager()


def _make_backend(num_heads: int):
    from vllm.v1.attention.backends.mla.prefill.aiter_asm import (
        AiterAsmPrefillBackend,
    )

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        ),
    )
    return AiterAsmPrefillBackend(
        num_heads=num_heads,
        scale=SCALE,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        vllm_config=vllm_config,
    )


def _make_prefill_metadata(seq_lens: list[int], device: torch.device):
    """Stand in for MLACommonPrefillMetadata with the fields the backend reads."""
    qo_indptr_cpu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    qo_indptr_cpu[1:] = torch.tensor(seq_lens, dtype=torch.int32).cumsum(0)
    metadata = SimpleNamespace(
        query_start_loc=qo_indptr_cpu.to(device),
        query_start_loc_cpu=qo_indptr_cpu,
        max_query_len=max(seq_lens),
        chunked_context=None,
        output_dtype=ATTN_OUT_DTYPE,
    )
    return metadata, int(qo_indptr_cpu[-1].item())


def _make_context_chunk_metadata(q_len: int, context_len: int, device: torch.device):
    """Stand in for a single-request, single-chunk MLACommonPrefillMetadata."""
    qo_indptr_cpu = torch.tensor([0, q_len], dtype=torch.int32)
    kv_indptr_cpu = torch.tensor([0, context_len], dtype=torch.int32)
    chunk = SimpleNamespace(
        index=0,
        query_start_loc=qo_indptr_cpu.to(device),
        query_start_loc_cpu=qo_indptr_cpu,
        cu_seq_lens=kv_indptr_cpu.to(device),
        cu_seq_lens_cpu=kv_indptr_cpu,
        max_query_len=q_len,
    )
    metadata = SimpleNamespace(
        query_start_loc=qo_indptr_cpu.to(device),
        query_start_loc_cpu=qo_indptr_cpu,
        max_query_len=q_len,
        chunked_context=SimpleNamespace(chunks=[chunk]),
        output_dtype=ATTN_OUT_DTYPE,
    )
    return metadata, chunk


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
) -> torch.Tensor:
    """SDPA reference on the fp8 inputs the kernel is handed."""
    qf = q.float().transpose(0, 1)  # [H, S, Dqk]
    kf = k.float().transpose(0, 1)
    vf = v.float().transpose(0, 1)  # [H, S, Dv]
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=is_causal, scale=SCALE)
    return out.transpose(0, 1)  # [S, H, Dv]


def _reference_lse(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Natural-log softmax denominator of the non-causal scores, as [H, S_q]."""
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * SCALE
    return torch.logsumexp(scores, dim=-1)


def _make_qkv(num_tokens_q: int, num_tokens_kv: int, num_heads: int, device):
    fp8 = current_platform.fp8_dtype()
    kwargs = dict(dtype=ATTN_OUT_DTYPE, device=device)
    q = torch.randn(num_tokens_q, num_heads, QK_HEAD_DIM, **kwargs).to(fp8)
    k = torch.randn(num_tokens_kv, num_heads, QK_HEAD_DIM, **kwargs).to(fp8)
    v = torch.randn(num_tokens_kv, num_heads, V_HEAD_DIM, **kwargs).to(fp8)
    return q, k, v


def _run_context_chunk(num_heads: int, q_len: int, context_len: int):
    device = torch.device("cuda")
    torch.manual_seed(0)

    backend = _make_backend(num_heads)
    metadata, chunk = _make_context_chunk_metadata(q_len, context_len, device)
    backend.prepare_metadata(metadata)

    from vllm.v1.worker.workspace import lock_workspace

    lock_workspace()

    q, k, v = _make_qkv(q_len, context_len, num_heads, device)
    out, lse = backend.run_prefill_context_chunk(chunk, q, k, v)
    return out, lse, q, k, v


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
@pytest.mark.parametrize("seq_len", [128, 512])
@torch.inference_mode()
def test_asm_prefill_new_tokens_matches_causal_reference(
    seq_len: int, num_heads: int
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    backend = _make_backend(num_heads)
    metadata, total_q = _make_prefill_metadata([seq_len], device)
    backend.prepare_metadata(metadata)

    from vllm.v1.worker.workspace import lock_workspace

    lock_workspace()

    q, k, v = _make_qkv(total_q, total_q, num_heads, device)

    out = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=False)

    out_ref = _reference(q, k, v, is_causal=True)

    assert out.shape == (total_q, num_heads, V_HEAD_DIM)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
@pytest.mark.parametrize(("q_len", "context_len"), [(128, 384), (512, 512)])
@torch.inference_mode()
def test_asm_prefill_context_chunk_matches_non_causal_reference(
    q_len: int, context_len: int, num_heads: int
) -> None:
    """Context chunks attend to the whole chunk, with no causal mask."""
    out, _, q, k, v = _run_context_chunk(num_heads, q_len, context_len)

    out_ref = _reference(q, k, v, is_causal=False)

    assert out.shape == (q_len, num_heads, V_HEAD_DIM)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
@pytest.mark.parametrize(("q_len", "context_len"), [(128, 384), (512, 512)])
@torch.inference_mode()
def test_asm_prefill_context_chunk_lse_matches_reference(
    q_len: int, context_len: int, num_heads: int
) -> None:
    """The chunk LSE feeds merge_attn_states, so a wrong one corrupts silently.

    mla_reduce_v1 writes final_lse only for work units the PS scheduler split,
    so an un-split tile leaves whatever the workspace held. That is invisible in
    the output of a single chunk and only shows up once the chunk is merged.
    """
    _, lse, q, k, _ = _run_context_chunk(num_heads, q_len, context_len)

    lse_ref = _reference_lse(q, k)

    assert lse.shape == (num_heads, q_len)
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(lse.float(), lse_ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
def test_kernel_head_width_is_aligned(num_heads: int) -> None:
    """The PS metadata, the workspace and the kernel must share one head width."""
    from vllm.v1.attention.backends.mla.prefill.aiter_asm import (
        _FP8_PREFILL_TILE_Q,
        _HEAD_ALIGNMENT,
    )

    backend = _make_backend(num_heads)
    width = backend._kernel_num_heads
    assert width % _HEAD_ALIGNMENT == 0
    assert num_heads <= width < num_heads + _HEAD_ALIGNMENT

    from aiter import get_ps_metadata_info_v1

    seq_len = 512
    metadata, _ = _make_prefill_metadata([seq_len], torch.device("cuda"))
    backend.prepare_metadata(metadata)

    expected_tiles = get_ps_metadata_info_v1(
        batch_size=1,
        num_head_k=width,
        max_qlen=seq_len,
        qlen_granularity=_FP8_PREFILL_TILE_Q,
    )[5][0]
    assert backend._new_tokens_ps["reduce_partial_map"].numel() == expected_tiles
