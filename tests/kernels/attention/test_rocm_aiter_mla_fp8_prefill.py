# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy test for the ROCm AITER FP8 MLA prefill path.

The FP8 MLA prefill path
(``AiterMLAImpl._mla_fp8_prefill_attn`` -> ``mla_prefill_ps_asm_fwd`` +
``mla_reduce_v1``) is auto-enabled on gfx950 (MI355) when AITER ships those
kernels, and it has no other test coverage. This test drives the *real*
metadata builder and impl methods (via ``object.__new__``, mirroring
``test_rocm_aiter_mla_head_padding.py::test_h12_aiter_mla_decode_matches_reference``)
so it exercises the actual persistent-scheduling metadata contract
(``get_ps_metadata_v1``) plus both kernels -- rather than re-implementing them --
and compares the output against a causal SDPA reference.

gfx950-only: ``mla_prefill_ps_asm_fwd`` has no gfx942 build, so this is skipped
on every other platform.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


def _fp8_prefill_available() -> bool:
    if not (current_platform.is_rocm() and torch.cuda.is_available()):
        return False
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        _fp8_mla_prefill_supported,
    )

    return _fp8_mla_prefill_supported()


pytestmark = pytest.mark.skipif(
    not _fp8_prefill_available(),
    reason="FP8 MLA prefill needs gfx950 (MI355) + an AITER build exporting "
    "mla_prefill_ps_asm_fwd / mla_reduce_v1",
)

# DeepSeek-style MLA head dims after kv_b_proj decompression (q/k carry the
# nope+rope dims, v carries v_head_dim). FP8 prefill requires 16-aligned heads.
NUM_HEADS = 16
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
V_HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)

# Dominated by fp8 (e4m3) rounding of q/k/v. Measured on MI355: worst-element
# abs error ~0.054, so atol=0.1 is ~2x that noise floor; rtol=0.05 bounds larger
# elements.
ATOL, RTOL = 1e-1, 5e-2


@pytest.fixture(autouse=True)
def _workspace_manager():
    """The FP8 prefill path draws per-call scratch from the global workspace
    manager, which ``GPUModelRunner.__init__`` normally initializes. Set it up
    for the test and tear it down so no global state leaks to other tests.
    """
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    init_workspace_manager(torch.device("cuda"))
    yield
    reset_workspace_manager()


def _make_impl():
    """Minimal AiterMLAImpl exposing only what _mla_fp8_prefill_attn reads."""
    from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1

    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

    impl = object.__new__(AiterMLAImpl)
    impl.num_heads = NUM_HEADS
    impl.v_head_dim = V_HEAD_DIM
    impl.scale = SCALE
    impl._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
    impl._mla_reduce_v1 = mla_reduce_v1
    return impl


def _build_prefill_metadata(seq_lens: list[int], device: torch.device):
    """Build fp8_prefill_* metadata via the real builder (no hand-rolling)."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder

    qo_indptr_cpu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    qo_indptr_cpu[1:] = torch.tensor(seq_lens, dtype=torch.int32).cumsum(0)
    qo_indptr = qo_indptr_cpu.to(device)
    total_q = int(qo_indptr_cpu[-1].item())
    max_q = max(seq_lens)

    builder = object.__new__(AiterMLAMetadataBuilder)
    builder.num_heads = NUM_HEADS
    builder.mla_dims = SimpleNamespace(v_head_dim=V_HEAD_DIM)
    builder._init_fp8_prefill_ps_buffers(
        max_num_reqs=len(seq_lens),
        max_prefill_qlen=max_q,
        max_num_batched_tokens=total_q,
        device=device,
    )

    # SimpleNamespace stands in for the metadata dataclasses: the builder only
    # reads prefill.{query_start_loc,max_query_len}, num_decodes, and
    # common.query_start_loc_cpu, then populates metadata.fp8_prefill_*.
    prefill = SimpleNamespace(query_start_loc=qo_indptr, max_query_len=max_q)
    metadata = SimpleNamespace(prefill=prefill, num_decodes=0)
    common = SimpleNamespace(query_start_loc_cpu=qo_indptr_cpu)
    builder._build_fp8_prefill_ps_metadata(metadata, common)
    return metadata, total_q


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal SDPA reference on fp8-quantized inputs (matches kernel dtype)."""
    fp8 = current_platform.fp8_dtype()
    qf = q.to(fp8).float().transpose(0, 1)  # [H, S, Dqk]
    kf = k.to(fp8).float().transpose(0, 1)
    vf = v.to(fp8).float().transpose(0, 1)  # [H, S, Dv]
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=True, scale=SCALE)
    return out.transpose(0, 1)  # [S, H, Dv]


@pytest.mark.parametrize("seq_len", [128, 512])
@torch.inference_mode()
def test_fp8_prefill_matches_reference(seq_len: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    metadata, total_q = _build_prefill_metadata([seq_len], device)

    q = torch.randn(
        total_q, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        total_q, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(total_q, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device=device)
    out = torch.zeros(
        total_q, NUM_HEADS * V_HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    impl = _make_impl()
    impl._mla_fp8_prefill_attn(q, k, v, metadata, out)

    out_ref = _reference(q, k, v)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.view(total_q, NUM_HEADS, V_HEAD_DIM).float(),
        out_ref.float(),
        atol=ATOL,
        rtol=RTOL,
    )
