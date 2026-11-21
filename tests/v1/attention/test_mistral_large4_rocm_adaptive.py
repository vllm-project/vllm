"""Adaptive verification for Mistral Large 4 DSpark on ROCm."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.mistral.amd.indexer import MistralIndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
    MistralLarge4IndexerBackend,
)
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseMetadataBuilder,
)
from vllm.v1.worker.gpu.attn_utils import (
    get_query_lens_mismatch_unsupported_backend,
    get_varlen_cudagraph_unsupported_backend,
)

K = 3
# Three verification requests with a draft budget of 6: the CPU spreads it as
# 2 drafts each, the device admits 3, 0 and 3. Trailing entries are padding.
CPU_QUERY_START_LOC = [0, 3, 6, 9, 9, 9]
DEVICE_QUERY_START_LOC = [0, 4, 5, 9, 9, 9]
NUM_REQS = 3
NUM_TOKENS = 9


def _make_ml4_config(
    *,
    adaptive: bool,
    num_speculative_tokens: int = K,
    model_type: str = "mistral_large_4",
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["PixtralForConditionalGeneration"],
            hf_text_config=SimpleNamespace(model_type=model_type),
        ),
        num_speculative_tokens=num_speculative_tokens,
        speculative_config=SimpleNamespace(
            enable_adaptive_verification=adaptive,
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=True,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        use_v2_model_runner=True,
    )


def _mock_rocm_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(indexer.current_platform, "is_rocm", lambda: True)


def _attn_group(backend, builder_cls, layer_name: str):
    return SimpleNamespace(
        layer_names=[layer_name],
        backend=backend,
        kv_cache_spec=None,
        get_metadata_builder=lambda ubatch_id=0: builder_cls,
    )


def _ml4_target_groups(indexer_backend=MistralLarge4IndexerBackend):
    return [
        [
            _attn_group(
                indexer_backend, DeepseekV32IndexerMetadataBuilder, "indexer.k_cache"
            ),
            _attn_group(
                ROCMAiterMLASparseBackend, ROCMAiterMLASparseMetadataBuilder, "attn"
            ),
        ]
    ]


def _make_sparse_mla_builder(*, query_lens_from_device: bool, topk_tokens: int = 64):
    builder = ROCMAiterMLASparseMetadataBuilder.__new__(
        ROCMAiterMLASparseMetadataBuilder
    )
    builder._query_lens_from_device = query_lens_from_device
    builder.req_id_per_token_buffer = torch.full((16,), -1, dtype=torch.int32)
    builder.qo_indptr = torch.arange(17, dtype=torch.int32)
    builder.topk_tokens = topk_tokens
    builder._num_attention_heads = 16
    return builder


def _make_common_metadata(device_query_start_loc: list[int], seq_lens: list[int]):
    return SimpleNamespace(
        query_start_loc=torch.tensor(device_query_start_loc, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(CPU_QUERY_START_LOC, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        max_query_len=K + 1,
    )


def _cpu_seg_lengths() -> np.ndarray:
    return np.diff(np.asarray(CPU_QUERY_START_LOC[: NUM_REQS + 1], dtype=np.int32))


@pytest.mark.cpu_test
def test_ml4_indexer_cache_selects_ml4_backend(monkeypatch: pytest.MonkeyPatch):
    cache = MistralIndexerCache.__new__(MistralIndexerCache)
    assert cache.get_attn_backend() is MistralLarge4IndexerBackend
    # The MLA layer keys its indexer handling on the V3.2 indexer name.
    assert MistralLarge4IndexerBackend.get_name() == "DEEPSEEK_V32_INDEXER"

    _mock_rocm_platform(monkeypatch)
    assert MistralLarge4IndexerBackend.supports_device_cpu_query_lens_mismatch()
    # Other DSA models keep the CPU-planned indexer path on ROCm.
    assert not DeepseekV32IndexerBackend.supports_device_cpu_query_lens_mismatch()


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_speculative_tokens", [1, K])
def test_ml4_adaptive_forces_flattened_indexer(
    monkeypatch: pytest.MonkeyPatch, num_speculative_tokens: int
):
    _mock_rocm_platform(monkeypatch)
    adaptive = _make_ml4_config(
        adaptive=True, num_speculative_tokens=num_speculative_tokens
    )
    # The per-token flattened rows take their lengths from the device.
    assert indexer._use_flattening(adaptive)
    assert (
        DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(adaptive, None)
        == AttentionCGSupport.ALWAYS
    )

    # ROCm runs next_n=2 natively, so only adaptive verification flattens it.
    assert not indexer._use_flattening(
        _make_ml4_config(adaptive=False, num_speculative_tokens=1)
    )
    assert not indexer._use_flattening(
        _make_ml4_config(
            adaptive=True, num_speculative_tokens=1, model_type="deepseek_v32"
        )
    )


@pytest.mark.cpu_test
def test_ml4_rocm_adaptive_gates(monkeypatch: pytest.MonkeyPatch):
    _mock_rocm_platform(monkeypatch)
    adaptive = _make_ml4_config(adaptive=True)

    assert get_query_lens_mismatch_unsupported_backend(_ml4_target_groups()) is None
    assert (
        get_varlen_cudagraph_unsupported_backend(_ml4_target_groups(), adaptive, K + 1)
        is None
    )

    v32_groups = _ml4_target_groups(indexer_backend=DeepseekV32IndexerBackend)
    assert (
        get_query_lens_mismatch_unsupported_backend(v32_groups)
        == "DeepseekV32IndexerBackend"
    )


@pytest.mark.cpu_test
def test_sparse_mla_varlen_cudagraph_bound():
    adaptive = _make_ml4_config(adaptive=True)
    fixed = _make_ml4_config(adaptive=False)
    builder = ROCMAiterMLASparseMetadataBuilder
    groups = [[_attn_group(ROCMAiterMLASparseBackend, builder, "attn")]]

    assert builder.get_cudagraph_support(adaptive, None) == (
        AttentionCGSupport.UNIFORM_BATCH
    )
    assert builder.get_varlen_cudagraph_max_query_len(adaptive, None) == K + 1
    # Without adaptive verification the metadata may follow the CPU split.
    assert builder.get_varlen_cudagraph_max_query_len(fixed, None) is None
    assert get_varlen_cudagraph_unsupported_backend(groups, fixed, K + 1) == (
        "ROCMAiterMLASparseBackend",
        None,
    )
    # Verification wider than the bound must still be rejected.
    assert get_varlen_cudagraph_unsupported_backend(groups, adaptive, K + 2) == (
        "ROCMAiterMLASparseBackend",
        K + 1,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "device_query_start_loc", [DEVICE_QUERY_START_LOC, CPU_QUERY_START_LOC]
)
def test_req_id_per_token_follows_device_query_lens(
    device_query_start_loc: list[int],
):
    builder = _make_sparse_mla_builder(query_lens_from_device=True)
    metadata = _make_common_metadata(device_query_start_loc, [20, 17, 30])

    builder._fill_req_id_per_token(metadata, NUM_REQS, NUM_TOKENS, _cpu_seg_lengths())

    device_seg_lengths = np.diff(device_query_start_loc[: NUM_REQS + 1])
    expected = np.repeat(np.arange(NUM_REQS, dtype=np.int32), device_seg_lengths)
    assert builder.req_id_per_token_buffer[:NUM_TOKENS].tolist() == expected.tolist()
    assert (builder.req_id_per_token_buffer[NUM_TOKENS:] == -1).all()


@pytest.mark.cpu_test
def test_req_id_per_token_differs_from_cpu_split_after_reallocation():
    builder = _make_sparse_mla_builder(query_lens_from_device=True)
    metadata = _make_common_metadata(DEVICE_QUERY_START_LOC, [20, 17, 30])

    builder._fill_req_id_per_token(metadata, NUM_REQS, NUM_TOKENS, _cpu_seg_lengths())

    cpu_split = np.repeat(np.arange(NUM_REQS, dtype=np.int32), _cpu_seg_lengths())
    assert cpu_split.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert builder.req_id_per_token_buffer[:NUM_TOKENS].tolist() == [
        0,
        0,
        0,
        0,
        1,
        2,
        2,
        2,
        2,
    ]


@pytest.mark.cpu_test
def test_persistent_metadata_key_tracks_device_query_lens():
    # Same CPU split and device seq_lens, different device query lengths:
    # the per-token KV lengths, and so the AITER work metadata, differ.
    step_a = _make_common_metadata(DEVICE_QUERY_START_LOC, [20, 17, 30])
    step_b = _make_common_metadata([0, 2, 5, 9, 9, 9], [20, 17, 30])
    cpu_seg_lengths = _cpu_seg_lengths()

    builder = _make_sparse_mla_builder(query_lens_from_device=True)
    key_a = builder._persistent_metadata_key(
        step_a, NUM_REQS, NUM_TOKENS, cpu_seg_lengths
    )
    key_b = builder._persistent_metadata_key(
        step_b, NUM_REQS, NUM_TOKENS, cpu_seg_lengths
    )
    assert key_a != key_b
    assert key_a == builder._persistent_metadata_key(
        step_a, NUM_REQS, NUM_TOKENS, cpu_seg_lengths
    )

    # Keyed on the CPU split, step_b would reuse step_a's stale metadata.
    cpu_keyed = _make_sparse_mla_builder(query_lens_from_device=False)
    assert cpu_keyed._persistent_metadata_key(
        step_a, NUM_REQS, NUM_TOKENS, cpu_seg_lengths
    ) == cpu_keyed._persistent_metadata_key(
        step_b, NUM_REQS, NUM_TOKENS, cpu_seg_lengths
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm indexer kernels")
def test_rocm_adaptive_indexer_rows_match_reference():
    """Each compacted verification row uses its true request and causal length."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as indexer_ops
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda")
    init_workspace_manager(device)
    torch.manual_seed(0)
    fp8_dtype = current_platform.fp8_dtype()
    num_heads, head_size, max_model_len = 64, 128, 256
    # Page size 1: AITER reads the cache unshuffled.
    block_size = 1
    # The last request is cudagraph padding with no query tokens.
    seq_lens = [180, 64, 130, 0]
    device_query_lens = [4, 1, 3, 0]
    cpu_query_lens = [3, 3, 2, 0]
    num_tokens = sum(device_query_lens)

    num_blocks = sum(seq_lens)
    block_tables = torch.zeros(
        (len(seq_lens), max(seq_lens)), dtype=torch.int32, device=device
    )
    offset = 0
    for req, seq_len in enumerate(seq_lens):
        block_tables[req, :seq_len] = torch.arange(offset, offset + seq_len)
        offset += seq_len
    kv_cache = torch.empty(
        num_blocks, block_size, 1, head_size + 4, dtype=torch.uint8, device=device
    )
    values = torch.randn(
        num_blocks, block_size, head_size, device=device, dtype=torch.bfloat16
    ).to(fp8_dtype)
    scales = torch.rand(num_blocks, block_size, device=device).clamp(min=1e-3)
    kv_flat = kv_cache.view(num_blocks, -1)
    kv_flat[:, : block_size * head_size] = values.view(num_blocks, -1).view(torch.uint8)
    kv_flat[:, block_size * head_size :] = scales.view(torch.uint8)

    builder = DeepseekV32IndexerMetadataBuilder.__new__(
        DeepseekV32IndexerMetadataBuilder
    )
    builder.vllm_config = _make_ml4_config(adaptive=True)
    builder.supports_varlen = False
    builder.decode_seq_lens_buffer = torch.zeros(
        num_tokens, dtype=torch.int32, device=device
    )
    builder.expanded_block_table_buffer = torch.zeros(
        (num_tokens, max(seq_lens)), dtype=torch.int32, device=device
    )
    builder.decode_lens_buffer = torch.zeros(
        num_tokens, dtype=torch.int32, device=device
    )
    builder.arange_buffer = torch.arange(num_tokens, dtype=torch.int32, device=device)

    query_start_loc = torch.tensor(
        [0, *np.cumsum(device_query_lens)], dtype=torch.int32, device=device
    )
    row_seq_lens, row_block_table, row_decode_lens, num_rows, requires_padding = (
        builder._prepare_decode_tensors(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            block_table=block_tables,
            decode_lens=query_start_loc.diff(),
            decode_lens_cpu=torch.tensor(cpu_query_lens, dtype=torch.int32),
            query_start_loc=query_start_loc[:-1],
            num_decodes=len(seq_lens),
            num_decode_tokens=num_tokens,
            use_native=False,
            next_n=K + 1,
            max_decode_len=max(cpu_query_lens),
        )
    )

    row_reqs = [r for r, n in enumerate(device_query_lens) for _ in range(n)]
    row_context_lens = [
        seq_lens[r] - device_query_lens[r] + j + 1
        for r, n in enumerate(device_query_lens)
        for j in range(n)
    ]
    assert (num_rows, requires_padding) == (num_tokens, False)
    assert row_decode_lens.tolist() == [1] * num_tokens
    assert row_seq_lens.tolist() == row_context_lens
    assert torch.equal(row_block_table, block_tables[row_reqs])

    q = torch.randn(
        num_tokens, 1, num_heads, head_size, device=device, dtype=torch.bfloat16
    ).to(fp8_dtype)
    weights = torch.rand(num_tokens, num_heads, device=device)
    logits = indexer_ops.rocm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        row_seq_lens,
        row_block_table,
        torch.empty(0, dtype=torch.int32, device=device),
        max_model_len,
    )
    reference = indexer_ops.fp8_paged_mqa_logits_torch(
        q,
        kv_cache,
        weights,
        torch.tensor(row_context_lens, dtype=torch.int32, device=device),
        block_tables[row_reqs],
        max_model_len,
    )
    for row, context_len in enumerate(row_context_lens):
        torch.testing.assert_close(
            logits[row, :context_len],
            reference[row, :context_len],
            atol=2e-3,
            rtol=2e-3,
        )
