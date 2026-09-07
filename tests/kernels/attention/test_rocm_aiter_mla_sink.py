# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for ROCm AITER sparse MLA attention sinks."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

Q_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = Q_HEAD_DIM**-0.5


def _gamma_fp32(operations: int) -> float:
    u = torch.finfo(torch.float32).eps / 2
    return operations * u / (1 - operations * u)


def _sink_reference(q, keys, sinks, scale):
    # Reference the stored operands, so input quantization is not kernel error.
    q, keys, sinks = q.double(), keys.double(), sinks.double()
    scores = q @ keys.T * scale
    logits = torch.cat((scores, sinks[:, None]), dim=-1)
    probabilities = logits.softmax(dim=-1)[:, :-1]
    values = keys[:, :V_HEAD_DIM]
    expected = probabilities @ values
    value_scale = probabilities @ values.abs()
    # FP32 dot accumulation and conversion/multiplication of the scale.
    score_error = _gamma_fp32(q.shape[-1] + 2) * (q.abs() @ keys.abs().T) * abs(scale)
    score_error = score_error.amax(dim=-1) if keys.shape[0] else torch.zeros_like(sinks)
    return expected, logits.logsumexp(dim=-1), value_scale, score_error


def _assert_sink_output_close(
    actual, expected, value_scale, score_error, probability_dtype, native
):
    assert actual.shape == expected.shape
    # Budget the identifiable rounding stages against the absolute value
    # contributions: output-relative ULPs become unbounded when values cancel.
    # Rounding P before P@V costs u(P) * sum(P*abs(V)).
    # AITER rounds output before and after sink scaling; Triton rounds once.
    u_probability = torch.finfo(probability_dtype).eps / 2
    u_output = torch.finfo(actual.dtype).eps / 2
    output_rounds = 2 if native else 1
    amplification = (1 + u_output) ** output_rounds
    relative_error = (1 + u_probability) * torch.exp(2 * score_error) - 1
    allowance = (
        relative_error.unsqueeze(-1) * value_scale * amplification
        + (amplification - 1) * expected.abs()
        + output_rounds * torch.finfo(actual.dtype).tiny * u_output
    )
    actual = actual.to(device=expected.device, dtype=torch.float64)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[value_scale == 0]) == 0
    error = (actual - expected).abs()
    max_ratio = (error / allowance.clamp_min(1e-300)).max().item()
    assert max_ratio <= 1, f"exceeded rounding budget by {max_ratio:.3f}x"


def _require_aiter() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() not in (3, 4):
        pytest.skip("AITER MLA requires CDNA 3 or 4")

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
        (40, "bf16"),
        (48, "bf16"),
        (64, "bf16"),
        (80, "bf16"),
    ],
)
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

    set_random_seed(real_heads * 17 + (cache_kind == "fp8"))
    device = torch.device("cuda")
    seq_lens = [0, 1, 5, 23, 17]
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
        num_prefills=0,
        num_decodes=batch_size,
        num_decode_tokens=batch_size,
        max_query_len=1,
    )
    sinks = torch.linspace(-2.0, 6.0, real_heads, device=device)

    impl = object.__new__(ROCMAiterMLASparseImpl)
    impl.num_heads = real_heads
    impl.kv_lora_rank = V_HEAD_DIM
    impl.kv_cache_dtype = cache_kind
    impl.scale = SM_SCALE
    impl.sinks = sinks

    output, lse = impl._forward_mla(
        SimpleNamespace(_q_scale=q_scale, _k_scale=kv_scale), q, kv, metadata
    )
    kv_flat = kv_ref[:, 0]
    references = []
    start = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        rows = kv_flat[indices[start : start + seq_len].long()]
        references.append(_sink_reference(q_ref[batch_idx], rows, sinks, SM_SCALE))
        start += seq_len
    expected, expected_lse, value_scale, score_error = (
        torch.stack(values) for values in zip(*references)
    )

    assert output.dtype == torch.bfloat16
    if lse is not None:
        assert lse.dtype == torch.float32
        assert lse.shape == expected_lse.shape
        lse_error = (lse.double() - expected_lse).abs()
        rounded_lse = expected_lse.float()
        positive_inf = torch.full_like(rounded_lse, float("inf"))
        ulp = torch.maximum(
            torch.nextafter(rounded_lse, positive_inf) - rounded_lse,
            rounded_lse - torch.nextafter(rounded_lse, -positive_inf),
        ).double()
        lse_allowance = score_error + 2 * ulp
        assert torch.all(lse_error <= lse_allowance)
        torch.testing.assert_close(lse[0].double(), expected_lse[0], atol=0, rtol=0)
    else:
        from vllm.platforms.rocm import on_gfx942

        assert cache_kind == "bf16" and real_heads in (40, 48, 64) and on_gfx942()
    _assert_sink_output_close(
        output,
        expected,
        value_scale,
        score_error,
        q_real.dtype,
        native=lse is not None,
    )


@pytest.mark.parametrize(
    ("q_dtype", "kv_dtype"),
    [
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ],
)
def test_sparse_mla_sink_rejects_unsupported_aiter_dtypes(
    q_dtype: torch.dtype, kv_dtype: torch.dtype
) -> None:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    impl = object.__new__(ROCMAiterMLASparseImpl)
    impl.num_heads = 16
    impl.kv_lora_rank = V_HEAD_DIM
    impl.kv_cache_dtype = "auto"
    impl.scale = SM_SCALE
    impl.sinks = torch.zeros(16, dtype=torch.float32)
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=0,
        num_decodes=1,
        num_decode_tokens=1,
        max_query_len=1,
    )

    with pytest.raises(ValueError, match="both use BF16 or both use FP8"):
        impl._forward_mla(
            SimpleNamespace(_q_scale=None, _k_scale=None),
            torch.empty(1, 16, Q_HEAD_DIM, dtype=q_dtype),
            torch.empty(1, 1, Q_HEAD_DIM, dtype=kv_dtype),
            metadata,
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


def test_sparse_mla_backend_reports_sink_support_for_current_hardware() -> None:
    from vllm.platforms.rocm import get_cdna_version
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend.supports_sink() == (get_cdna_version() in (3, 4))


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


@pytest.mark.parametrize("num_heads,block_size", [(8, 16), (20, 32)])
@pytest.mark.parametrize("interleaved_pages", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_sparse_mla_sink_matches_dense_attention_with_empty_rows_and_paged_cache(
    num_heads: int,
    block_size: int,
    interleaved_pages: bool,
    dtype: torch.dtype,
) -> None:
    """Preserve head-specific sinks and latent values across ragged cache pages."""
    _require_aiter()
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

    head_dim, value_dim = 576, 512
    generator = torch.Generator().manual_seed(421)
    # The empty row exercises sink-only attention; the last row crosses both
    # page boundaries and the kernel's 16-key reduction tile.
    selected_rows = [
        [],
        [(2, 7)],
        [(3, 1), (0, block_size - 1), (1, 0), (2, 4)],
        [(i % 4, (i * 7) % block_size) for i in range(23)],
    ]
    q_cpu = torch.randn(
        len(selected_rows), num_heads, head_dim, generator=generator
    ).to(dtype)
    cache_cpu = torch.randn(4, block_size, head_dim, generator=generator).to(dtype)
    sinks_cpu = torch.linspace(-4.0, 6.0, num_heads)

    # Layers can share a backing allocation, leaving unused rows between pages.
    num_layers = 3 if interleaved_pages else 1
    backing = torch.full(
        (4, num_layers, block_size, head_dim),
        float("nan"),
        dtype=dtype,
        device="cuda",
    )
    cache = backing[:, num_layers - 1]
    cache.copy_(cache_cpu)
    indices = [
        page * num_layers * block_size + offset
        for rows in selected_rows
        for page, offset in rows
    ]
    lengths = torch.tensor([0, *(len(rows) for rows in selected_rows)])
    metadata = SimpleNamespace(
        block_size=block_size,
        num_prefills=0,
        num_decodes=len(selected_rows),
        num_decode_tokens=len(selected_rows),
        max_query_len=1,
        qo_indptr=torch.arange(
            len(selected_rows) + 1, dtype=torch.int32, device="cuda"
        ),
        paged_kv_last_page_len=torch.ones(
            len(selected_rows), dtype=torch.int32, device="cuda"
        ),
        work_meta_data=None,
        attn_out_dtype=dtype,
        paged_kv_indices=torch.tensor(indices, dtype=torch.int32, device="cuda"),
        paged_kv_indptr=lengths.cumsum(0).to(device="cuda", dtype=torch.int32),
    )
    impl = ROCMAiterMLASparseImpl.__new__(ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.head_size = head_dim
    impl.kv_lora_rank = value_dim
    impl.scale = head_dim**-0.5
    impl.kv_cache_dtype = "auto"
    impl.sinks = sinks_cpu.cuda()
    padded_q = AiterMLAHelper.get_mla_padded_q(num_heads, q_cpu.cuda())

    kv_rows, _ = flat_kv_row_view(cache, block_size)
    actual, _ = impl._forward_mla(
        SimpleNamespace(_q_scale=None, _k_scale=None),
        padded_q,
        kv_rows.unsqueeze(1),
        metadata,
    )

    references = []
    for query_idx, rows in enumerate(selected_rows):
        keys = (
            torch.stack([cache_cpu[page, offset] for page, offset in rows])
            if rows
            else cache_cpu.new_empty((0, head_dim))
        )
        references.append(
            _sink_reference(q_cpu[query_idx], keys, sinks_cpu, impl.scale)
        )
    expected, _, value_scale, score_error = (
        torch.stack(values) for values in zip(*references)
    )
    assert actual.shape == (len(selected_rows), num_heads, value_dim)
    assert actual.dtype == dtype
    _assert_sink_output_close(
        actual,
        expected,
        value_scale,
        score_error,
        dtype,
        native=dtype == torch.bfloat16,
    )


@pytest.mark.parametrize("layout", ["LBNHC", "LBHNC", "BLNHC"])
def test_sparse_mla_backend_resolves_only_contiguous_layer_layouts(monkeypatch, layout):
    from vllm.config import CacheConfig
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", layout)
    config = SimpleNamespace(cache_config=CacheConfig())
    supported = [
        [x.name for x in ROCMAiterMLASparseBackend.supported_kv_cache_layouts()]
    ]
    if layout == "BLNHC":
        with pytest.raises(ValueError, match="does not satisfy"):
            resolve_kv_cache_layout(config, supported)
    else:
        assert resolve_kv_cache_layout(config, supported).name == layout


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_sparse_mla_sink_forward_mqa_preserves_split_query(dtype):
    """The public sparse forward joins latent/RoPE queries in the model dtype."""
    _require_aiter()
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    set_random_seed(412)
    num_tokens, num_heads, block_size = 2, 8, 16
    q = torch.randn(num_tokens, num_heads, Q_HEAD_DIM, device="cuda").to(dtype)
    kv = torch.randn(2 * block_size, 1, Q_HEAD_DIM, device="cuda").to(dtype)
    selected = torch.tensor([[1, 7, 18], [0, 3, 20]], dtype=torch.int32, device="cuda")
    sinks = torch.linspace(-3.0, 5.0, num_heads, device="cuda")
    impl = object.__new__(ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = V_HEAD_DIM
    impl.kv_cache_dtype = "auto"
    impl.scale = SM_SCALE
    impl.sinks = sinks
    impl.topk_indices_buffer = torch.full(
        (num_tokens, 128), -1, dtype=torch.int32, device="cuda"
    )
    impl.topk_indices_buffer[:, : selected.shape[1]] = selected
    impl.q_concat_buffer = torch.empty_like(q)
    metadata = SimpleNamespace(
        attn_out_dtype=dtype,
        num_actual_tokens=num_tokens,
        num_prefills=0,
        num_decodes=num_tokens,
        num_decode_tokens=num_tokens,
        max_query_len=1,
        block_size=block_size,
        topk_tokens=impl.topk_indices_buffer.shape[1],
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device="cuda"),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
        qo_indptr=torch.arange(num_tokens + 1, dtype=torch.int32, device="cuda"),
        paged_kv_indptr=torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda"),
        paged_kv_indices=torch.empty(
            selected.numel(), dtype=torch.int32, device="cuda"
        ),
        paged_kv_last_page_len=torch.ones(num_tokens, dtype=torch.int32, device="cuda"),
        work_meta_data=None,
    )
    actual, _ = impl.forward_mqa(
        (q[..., :V_HEAD_DIM], q[..., V_HEAD_DIM:]),
        kv,
        metadata,
        SimpleNamespace(_q_scale=None, _k_scale=None),
    )
    references = [
        _sink_reference(q[i], kv[:, 0][selected[i].long()], sinks, SM_SCALE)
        for i in range(num_tokens)
    ]
    expected, _, value_scale, score_error = (
        torch.stack(values) for values in zip(*references)
    )
    assert actual.dtype == dtype
    _assert_sink_output_close(
        actual,
        expected,
        value_scale,
        score_error,
        dtype,
        native=dtype == torch.bfloat16,
    )
