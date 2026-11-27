# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm custom op schema tests for AITER MLA decode.

`opcheck` verifies that the decode ops are registered and that their schemas
and fake implementations are consistent with the real kernels: fake-tensor
support for torch.compile tracing and `mutates_args=["o"]` in-place output
aliasing.
"""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

Q_HEAD_DIM = 576  # kv_lora_rank + qk_rope_head_dim
V_HEAD_DIM = 512  # kv_lora_rank


def _require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() not in (3, 4):
        pytest.skip("AITER MLA requires CDNA 3 or 4")

    if not is_aiter_found_and_supported():
        pytest.skip("aiter is required on supported ROCm hardware for this test")


@torch.inference_mode()
def test_gather_kv_b_proj_op_schema() -> None:
    """Validate fake tensors and mutation of both expanded K/V outputs."""
    _require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401

    rows, num_heads = 32, 12
    cache = torch.randn(rows, 1, 576, device="cuda").to(current_platform.fp8_dtype())
    scale = torch.tensor([0.07], device="cuda")
    cu_seq_lens = torch.tensor([0, 13, rows], dtype=torch.int32, device="cuda")
    indices = torch.arange(rows, dtype=torch.int32, device="cuda").flip(0)
    weight = torch.randn(num_heads * 256, 512, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(rows, num_heads, 192, dtype=torch.bfloat16, device="cuda")
    v = torch.empty(rows, num_heads, 128, dtype=torch.bfloat16, device="cuda")
    opcheck(
        torch.ops.vllm.rocm_aiter_gather_kv_b_proj,
        (cache, scale, cu_seq_lens, indices, cu_seq_lens, weight, k, v),
    )


@torch.inference_mode()
def test_mla_decode_fwd_op_schema() -> None:
    """Opcheck validates registration, schema, fake-tensor, and `o` aliasing.

    A single opcheck call covers that the op is registered/callable, that its
    fake implementation matches the real op (torch.compile tracing), and that
    the `mutates_args=["o"]` in-place output aliasing is declared correctly.
    """
    _require_aiter()
    # Import ensures the custom op is registered.
    from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401

    batch_size, nhead = 4, 128

    q = torch.randn(batch_size, nhead, Q_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kv_buffer = torch.randn(64, 1, 1, Q_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(batch_size, nhead, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * 16
    kv_indices = torch.arange(0, 64, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    opcheck(
        torch.ops.vllm.rocm_aiter_mla_decode_fwd,
        (q, kv_buffer, o, qo_indptr, 1),
        {
            "kv_indptr": kv_indptr,
            "kv_indices": kv_indices,
            "kv_last_page_lens": kv_last_page_lens,
            "sm_scale": Q_HEAD_DIM**-0.5,
            "logit_cap": 0.0,
            "q_scale": None,
            "kv_scale": None,
            "work_meta_data": None,
            "work_indptr": None,
            "work_info_set": None,
            "reduce_indptr": None,
            "reduce_final_map": None,
            "reduce_partial_map": None,
        },
    )


@torch.inference_mode()
def test_mla_decode_fwd_lse_op_schema() -> None:
    """Validate graph registration and mutation schema for LSE decode."""
    _require_aiter()
    # Import ensures the custom op is registered.
    from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401

    batch_size, nhead = 2, 16
    q = torch.randn(batch_size, nhead, Q_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kv_buffer = torch.randn(32, 1, 1, Q_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(batch_size, nhead, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * 16
    kv_indices = torch.arange(32, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    opcheck(
        torch.ops.vllm.rocm_aiter_mla_decode_fwd_lse,
        (q, kv_buffer, o, qo_indptr, 1),
        {
            "kv_indptr": kv_indptr,
            "kv_indices": kv_indices,
            "kv_last_page_lens": kv_last_page_lens,
            "sm_scale": Q_HEAD_DIM**-0.5,
            "logit_cap": 0.0,
            "q_scale": None,
            "kv_scale": None,
        },
    )


def test_sparse_attn_indexer_registers_without_aiter(monkeypatch) -> None:
    """`rocm_aiter_sparse_attn_indexer` registers on ROCm even without AITER."""
    import vllm._aiter_ops as aiter_ops_mod

    registered: list[str] = []

    def fake_register(*, op_name, **kwargs):
        registered.append(op_name)

    monkeypatch.setattr(aiter_ops_mod, "_OPS_REGISTERED", False)
    monkeypatch.setattr(aiter_ops_mod, "is_aiter_found_and_supported", lambda: False)
    monkeypatch.setattr(
        aiter_ops_mod, "is_aiter_found_and_supported_on_rdna", lambda: False
    )
    monkeypatch.setattr(aiter_ops_mod, "direct_register_custom_op", fake_register)

    aiter_ops_mod.rocm_aiter_ops.register_ops_once()
    assert registered == ["rocm_aiter_sparse_attn_indexer"]


def test_sparse_attn_indexer_forward_hip_dispatches_without_aiter(monkeypatch) -> None:
    """`SparseAttnIndexer.forward_hip` dispatches on non-AITER ROCm GPUs."""
    from types import SimpleNamespace

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    expected = torch.empty((2, 4), dtype=torch.int32)
    captured: dict[str, object] = {}

    def fake_indexer_op(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: False)
    monkeypatch.setattr(
        torch.ops.vllm, "rocm_aiter_sparse_attn_indexer", fake_indexer_op, raising=False
    )

    indexer = object.__new__(SparseAttnIndexer)
    indexer.use_fp4_cache = False
    indexer.k_cache = SimpleNamespace(prefix="layer.0.indexer", kv_cache=torch.empty(1))
    indexer.quant_block_size = 128
    indexer.scale_fmt = "ue8m0"
    indexer.topk_tokens = 4
    indexer.head_dim = 128
    indexer.max_model_len = 1024
    indexer.max_total_seq_len = 1024
    indexer.topk_indices_buffer = expected
    indexer.skip_k_cache_insert = False
    indexer.compress_ratio = 1
    indexer.candidate_blocks = None
    indexer.candidate_block_size = 0
    indexer.candidate_write = False
    indexer.use_pcp = False
    indexer.pcp_shard_decode_requests = False
    indexer.cache_layout = None
    indexer.dcp_rank = 0
    indexer.dcp_world_size = 1
    indexer._parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=1)

    hidden = torch.empty((2, 8))
    q_quant = torch.empty((2, 4, 128))
    k = torch.empty((2, 128))
    weights = torch.empty((2, 4))

    out = SparseAttnIndexer.forward_hip(indexer, hidden, q_quant, k, weights)
    assert out is expected
    assert captured["args"][0] is hidden
    assert captured["args"][8] == 4
