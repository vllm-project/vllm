# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for ROCm AITER sparse MLA attention sinks."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

_SKIP_UNSUPPORTED_AITER_MLA = True
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_mi3xx

    _SKIP_UNSUPPORTED_AITER_MLA = not on_mi3xx()

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

Q_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = Q_HEAD_DIM**-0.5


def _require_aiter() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter is required on supported ROCm hardware for this test")


@pytest.mark.parametrize(
    ("real_heads", "cache_kind"),
    [
        (1, "bf16"),
        (4, "bf16"),
        (8, "bf16"),
        (8, "fp8"),
        (32, "fp8"),
        (64, "fp8"),
        (12, "bf16"),
        (16, "bf16"),
        (24, "bf16"),
        (32, "bf16"),
        (48, "bf16"),
        (64, "bf16"),
        (80, "bf16"),
    ],
)
@pytest.mark.skipif(_SKIP_UNSUPPORTED_AITER_MLA, reason="MI300/MI350 AITER MLA only")
@torch.inference_mode()
def test_sparse_mla_sink_matches_ragged_reference(
    real_heads: int, cache_kind: str
) -> None:
    """Exercise ragged sink decode, head padding, and FP8 scale forwarding."""
    _require_aiter()
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    if cache_kind == "bf16" and real_heads in (48, 64):
        from vllm.platforms.rocm import on_gfx942

        if on_gfx942():
            pytest.skip("This shape maps to gfx942's unsupported BF16 H64 kernel")

    set_random_seed(real_heads * 17 + (cache_kind == "fp8"))
    device = torch.device("cuda")
    seq_lens = [1, 5, 23, 17]
    batch_size = len(seq_lens)

    q_source = torch.randn(batch_size, real_heads, Q_HEAD_DIM, device=device) * 0.2

    pool_size = sum(seq_lens) + 52
    kv_source = torch.randn(pool_size, 1, Q_HEAD_DIM, device=device) * 0.2
    if cache_kind == "fp8":
        fp8_dtype = current_platform.fp8_dtype()
        q_scale = torch.tensor(0.5, dtype=torch.float32, device=device)
        kv_scale = torch.tensor(0.25, dtype=torch.float32, device=device)
        q_real = (q_source / q_scale).to(fp8_dtype)
        kv = (kv_source / kv_scale).to(fp8_dtype)
        q_ref = q_real.float() * q_scale
        kv_ref = kv.float() * kv_scale
    else:
        q_scale = kv_scale = None
        q_real = q_source.to(torch.bfloat16)
        kv = kv_source.to(torch.bfloat16)
        q_ref = q_real.float()
        kv_ref = kv.float()
    q = AiterMLAHelper.get_mla_padded_q(real_heads, q_real)

    indices = torch.randperm(pool_size, device=device)[: sum(seq_lens)].to(torch.int32)
    kv_indptr = torch.tensor(
        [0] + [sum(seq_lens[:i]) for i in range(1, len(seq_lens) + 1)],
        dtype=torch.int32,
        device=device,
    )

    # Non-None garbage proves the sink path cannot accidentally select the
    # gfx942 persistent kernel, which has no return-LSE code object.
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        qo_indptr=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.ones(batch_size, dtype=torch.int32, device=device),
        work_meta_data=torch.tensor([123], dtype=torch.int32),
        work_indptr=None,
        work_info_set=None,
        reduce_indptr=None,
        reduce_final_map=None,
        reduce_partial_map=None,
    )
    sinks = torch.linspace(-2.0, 6.0, real_heads, device=device)

    impl = object.__new__(ROCMAiterMLASparseImpl)
    impl.num_heads = real_heads
    impl.kv_lora_rank = V_HEAD_DIM
    impl.scale = SM_SCALE
    impl.sinks = sinks

    output, lse = impl._forward_mla(
        SimpleNamespace(_q_scale=q_scale, _k_scale=kv_scale), q, kv, metadata
    )
    assert lse is not None

    kv_flat = kv_ref[:, 0]
    ref_outputs = []
    ref_lses = []
    start = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        rows = kv_flat[indices[start : start + seq_len].long()].float()
        scores = q_ref[batch_idx] @ rows.T * SM_SCALE
        ref_outputs.append(torch.softmax(scores, dim=-1) @ rows[:, :V_HEAD_DIM])
        ref_lses.append(torch.logsumexp(scores, dim=-1))
        start += seq_len

    ref_output = torch.stack(ref_outputs)
    ref_lse = torch.stack(ref_lses)
    expected_lse = torch.logaddexp(ref_lse, sinks)
    expected_output = ref_output * torch.exp(ref_lse - expected_lse).unsqueeze(-1)

    assert output.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)
    output_atol = 1e-2 if cache_kind == "fp8" else 2e-3
    output_rtol = 3e-2 if cache_kind == "fp8" else 2e-2
    torch.testing.assert_close(
        output.float(), expected_output, atol=output_atol, rtol=output_rtol
    )


def _make_noncontiguous_sink() -> torch.Tensor:
    return torch.empty(8, dtype=torch.float32)[::2]


@pytest.mark.parametrize(
    ("sinks", "match"),
    [
        (torch.empty(4, dtype=torch.bfloat16), "must be float32"),
        (torch.empty(2, 2, dtype=torch.float32), "must have shape"),
        (_make_noncontiguous_sink(), "must be contiguous"),
    ],
)
def test_sparse_mla_sink_validation(sinks: torch.Tensor, match: str) -> None:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    with pytest.raises(ValueError, match=match):
        ROCMAiterMLASparseImpl(
            num_heads=4,
            head_size=Q_HEAD_DIM,
            scale=SM_SCALE,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="bfloat16",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            sinks=sinks,
            kv_lora_rank=V_HEAD_DIM,
        )


@pytest.mark.parametrize("real_heads", [48, 64])
def test_sparse_mla_sink_rejects_unsupported_gfx942_h64_at_runtime(
    real_heads: int,
) -> None:
    from vllm.platforms.rocm import on_gfx942
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    if not on_gfx942():
        pytest.skip("This AITER head-count constraint is gfx942-specific")

    device = torch.device("cuda")
    impl = object.__new__(ROCMAiterMLASparseImpl)
    impl.num_heads = real_heads
    impl.kv_lora_rank = V_HEAD_DIM
    impl.scale = SM_SCALE
    impl.sinks = torch.zeros(real_heads, dtype=torch.float32, device=device)
    metadata = SimpleNamespace(attn_out_dtype=torch.bfloat16)
    padded_heads = AiterMLAHelper.get_actual_mla_num_heads(real_heads)

    with pytest.raises(ValueError, match="increase tensor_parallel_size"):
        impl._forward_mla(
            SimpleNamespace(_q_scale=None, _k_scale=None),
            torch.empty(
                1,
                padded_heads,
                Q_HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
            ),
            torch.empty(1, 1, Q_HEAD_DIM, dtype=torch.bfloat16, device=device),
            metadata,
        )


def test_sparse_mla_backend_reports_sink_support_for_current_hardware() -> None:
    from vllm.platforms.rocm import on_mi3xx
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend.supports_sink() == on_mi3xx()


def test_sparse_mla_backend_rejects_dcp() -> None:
    from vllm.platforms.rocm import RocmPlatform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

    selector_config = AttentionSelectorConfig(
        head_size=Q_HEAD_DIM,
        dtype=torch.bfloat16,
        kv_cache_dtype="bfloat16",
        block_size=16,
        use_mla=True,
        has_sink=True,
        use_sparse=True,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        attn_type="decoder",
        use_dcp=True,
    )

    with pytest.raises(ValueError, match="DCP not supported"):
        RocmPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.ROCM_AITER_MLA_SPARSE,
            attn_selector_config=selector_config,
        )
