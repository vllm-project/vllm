# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm custom op schema tests for AITER MLA decode.

``opcheck`` verifies that the decode ops are registered and that their schemas
and fake implementations are consistent with the real kernels: fake-tensor
support for torch.compile tracing and ``mutates_args=["o"]`` in-place output
aliasing.
"""

from types import SimpleNamespace

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


@pytest.mark.parametrize("candidate_blocks", [False, True])
def test_sparse_indexer_forward_hip_schema(monkeypatch, candidate_blocks):
    """The ROCm caller must match the registered op and reject CUDA-only features."""
    _require_aiter()
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    with FakeTensorMode():
        tensor = torch.empty(1, device="cuda")
        indices = torch.empty((1, 1), dtype=torch.int32, device="cuda")
        indexer = SimpleNamespace(
            use_fp4_cache=False,
            candidate_blocks=indices if candidate_blocks else None,
            candidate_block_size=128 if candidate_blocks else 0,
            candidate_write=False,
            k_cache=SimpleNamespace(prefix="model.layers.0.indexer", kv_cache=tensor),
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=1,
            head_dim=128,
            max_model_len=128,
            max_total_seq_len=128,
            topk_indices_buffer=indices,
            skip_k_cache_insert=False,
            compress_ratio=1,
        )
        if candidate_blocks:
            with pytest.raises(NotImplementedError, match="candidate blocks"):
                SparseAttnIndexer.forward_hip(indexer, tensor, tensor, None, tensor)
        else:
            output = SparseAttnIndexer.forward_hip(
                indexer, tensor, tensor, None, tensor
            )
            assert output is indices


@torch.inference_mode()
def test_mla_decode_fwd_op_schema() -> None:
    """opcheck validates registration, schema, fake-tensor, and ``o`` aliasing.

    A single opcheck call covers that the op is registered/callable, that its
    fake implementation matches the real op (torch.compile tracing), and that
    the ``mutates_args=["o"]`` in-place output aliasing is declared correctly.
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
