# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm custom op schema test for AITER MLA decode.

A single ``opcheck`` call on ``torch.ops.vllm.rocm_aiter_mla_decode_fwd``
verifies that the custom op is registered and that its schema and fake
implementation are consistent with the real kernel: fake-tensor support for
torch.compile tracing and the ``mutates_args=["o"]`` in-place output aliasing.
"""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.platforms import current_platform

_SKIP_NON_MI3XX = True
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_mi3xx

    _SKIP_NON_MI3XX = not on_mi3xx()

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific tests"),
    pytest.mark.skipif(_SKIP_NON_MI3XX, reason="MI300/MI350 ROCm only"),
]

Q_HEAD_DIM = 576  # kv_lora_rank + qk_rope_head_dim
V_HEAD_DIM = 512  # kv_lora_rank


def _require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter is required on supported ROCm hardware for this test")


@torch.inference_mode()
def test_mla_decode_fwd_op_schema() -> None:
    """Opcheck validates registration, schema, fake-tensor, and ``o`` aliasing.

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
