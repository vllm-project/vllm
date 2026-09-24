# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm kernel correctness tests for AITER MLA decode.

Compares ``rocm_aiter_ops.mla_decode_fwd`` against a pure PyTorch reference
under the absorbed MLA formulation (DeepSeek-V3/V4) with varied batch sizes,
head counts, and sequence lengths.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

_SKIP_NON_MI3XX = True
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_mi3xx

    _SKIP_NON_MI3XX = not on_mi3xx()

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific tests"),
    pytest.mark.skipif(_SKIP_NON_MI3XX, reason="MI300/MI350 ROCm only"),
]

# DeepSeek-V3/V4 MLA config.
Q_HEAD_DIM = 576  # kv_lora_rank + qk_rope_head_dim
V_HEAD_DIM = 512  # kv_lora_rank
SM_SCALE = Q_HEAD_DIM**-0.5

NUM_HEADS = [16, 128]
BATCH_SIZES = [1, 4, 16]
KV_SEQ_LENS = [16, 256]

ATOL, RTOL = 1.5e-2, 1e-2


def _require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter is required on supported ROCm hardware for this test")


def _ref_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference for absorbed MLA decode.

    K uses full q_head_dim (576) for scoring, V uses first v_head_dim (512)
    for output. This mirrors the production kernel's absorbed formulation.
    """
    batch_size, num_heads, _ = q.shape
    output = torch.zeros(
        batch_size, num_heads, V_HEAD_DIM, dtype=q.dtype, device=q.device
    )

    for b in range(batch_size):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_indices = kv_indices[start:end]

        k = kv_buffer[token_indices].float()
        v = kv_buffer[token_indices, :V_HEAD_DIM].float()

        for h in range(num_heads):
            q_h = q[b, h, :].float()
            scores = torch.mv(k, q_h) * SM_SCALE
            weights = torch.softmax(scores, dim=0)
            output[b, h, :] = torch.mv(v.t(), weights).to(q.dtype)

    return output


def _make_inputs(
    batch_size: int,
    nhead: int,
    kv_seq_len: int,
    *,
    contiguous_indices: bool = True,
):
    """Build valid MLA decode inputs on the current CUDA device.

    When contiguous_indices=False, tokens are randomly scattered in a 2x pool
    to simulate real paged allocation.
    """
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

    actual_nhead = AiterMLAHelper.get_actual_mla_num_heads(nhead)
    total_kv_tokens = batch_size * kv_seq_len

    pool_size = total_kv_tokens if contiguous_indices else total_kv_tokens * 2

    q = torch.randn(
        batch_size, actual_nhead, Q_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    kv_buffer = torch.randn(
        pool_size, 1, Q_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    o = torch.zeros(
        batch_size, actual_nhead, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * kv_seq_len
    )
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    if contiguous_indices:
        kv_indices = torch.arange(0, total_kv_tokens, dtype=torch.int32, device="cuda")
    else:
        perm = torch.randperm(pool_size, device="cuda")[:total_kv_tokens]
        kv_indices = perm.to(torch.int32)

    return {
        "q": q,
        "kv_buffer": kv_buffer,
        "o": o,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "actual_nhead": actual_nhead,
    }


def _run_kernel(inputs: dict) -> torch.Tensor:
    """Run decode through the production rocm_aiter_ops path."""
    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.mla_decode_fwd(
        inputs["q"],
        inputs["kv_buffer"],
        inputs["o"],
        SM_SCALE,
        inputs["qo_indptr"],
        1,
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_lens"],
    )
    return inputs["o"]


def _ref_output(inputs: dict) -> torch.Tensor:
    """Compute reference output from the same inputs."""
    kv_flat = inputs["kv_buffer"].squeeze(1)
    return _ref_mla_decode(
        inputs["q"], kv_flat, inputs["kv_indptr"], inputs["kv_indices"]
    )


@pytest.mark.parametrize("nhead", NUM_HEADS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("kv_seq_len", KV_SEQ_LENS)
@torch.inference_mode()
def test_mla_decode_accuracy(
    nhead: int,
    batch_size: int,
    kv_seq_len: int,
) -> None:
    """BF16 decode accuracy vs PyTorch reference (contiguous indices)."""
    _require_aiter()
    set_random_seed(0)

    inputs = _make_inputs(batch_size, nhead, kv_seq_len)
    output = _run_kernel(inputs)
    output_ref = _ref_output(inputs)

    torch.testing.assert_close(output, output_ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("nhead", NUM_HEADS)
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("kv_seq_len", [16, 128])
@torch.inference_mode()
def test_mla_decode_noncontiguous_indices(
    nhead: int,
    batch_size: int,
    kv_seq_len: int,
) -> None:
    """BF16 decode accuracy with shuffled kv_indices (paged allocation)."""
    _require_aiter()
    set_random_seed(0)

    inputs = _make_inputs(batch_size, nhead, kv_seq_len, contiguous_indices=False)
    output = _run_kernel(inputs)
    output_ref = _ref_output(inputs)

    torch.testing.assert_close(output, output_ref, atol=ATOL, rtol=RTOL)


@torch.inference_mode()
def test_mla_decode_determinism() -> None:
    """Repeated decode calls produce bitwise-identical results."""
    _require_aiter()
    set_random_seed(0)

    inputs = _make_inputs(batch_size=4, nhead=128, kv_seq_len=16)
    out_first = _run_kernel(inputs).clone()

    for _ in range(3):
        inputs["o"] = torch.zeros_like(inputs["o"])
        out = _run_kernel(inputs)
        torch.testing.assert_close(out, out_first, atol=0, rtol=0)


@torch.inference_mode()
def test_mla_decode_smoke() -> None:
    """Basic sanity: correct shape, dtype, finite, non-zero."""
    _require_aiter()
    set_random_seed(0)

    inputs = _make_inputs(batch_size=4, nhead=128, kv_seq_len=64)
    output = _run_kernel(inputs)

    assert output.shape == (4, 128, V_HEAD_DIM)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
