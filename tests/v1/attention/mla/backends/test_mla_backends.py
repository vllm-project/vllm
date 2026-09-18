# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 MLA backends without GPUModelRunner dependency.

The parametrized backend-correctness matrix lives in ``../correctness/``, one
directory per MLA prefill backend, over the shared bodies in
``tests/v1/attention/_mla_backends.py``.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention._mla_backends import DEVICE_TYPE, MockMLAAttentionLayer
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.attention import mla_attention as mla_attention_module
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLACommonBaseImpl,
    _use_masked_mha,
    build_mla_chunked_context_metadata,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import flashmla as flashmla_module
from vllm.v1.attention.backends.mla import tokenspeed_mla as tokenspeed_mla_module
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    MLAAttentionSpec,
)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "query_len", "expected"),
    [
        (1, 8192, True),
        (1, 9216, False),
        (2, 20480, True),
        (2, 24576, False),
        (4, 48 * 1024, True),
        (4, 56 * 1024, False),
        (8, 112 * 1024, True),
        (8, 128 * 1024, False),
    ],
)
def test_glm5_masked_mha_pure_prefill_routing(
    tensor_parallel_size, query_len, expected
):
    assert (
        _use_masked_mha(
            backend_name="FLASHMLA_SPARSE",
            tensor_parallel_size=tensor_parallel_size,
            qk_head_dim=256,
            v_head_dim=256,
            query_len=query_len,
            seq_len=query_len,
            has_context=False,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("tensor_parallel_size", "query_len", "seq_len", "expected"),
    [
        (1, 8192, 8192, False),
        (2, 4096, 6144, True),
        (2, 4096, 8192, False),
        (2, 8192, 10240, True),
        (4, 16384, 24576, True),
        (4, 16384, 32768, False),
        (4, 32768, 34 * 1024, True),
        (4, 32768, 36 * 1024, False),
        (8, 32768, 49152, True),
        (8, 32768, 65536, False),
    ],
)
def test_glm5_masked_mha_context_routing(
    tensor_parallel_size, query_len, seq_len, expected
):
    assert (
        _use_masked_mha(
            backend_name="FLASHMLA_SPARSE",
            tensor_parallel_size=tensor_parallel_size,
            qk_head_dim=256,
            v_head_dim=256,
            query_len=query_len,
            seq_len=seq_len,
            has_context=True,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("tensor_parallel_size", "query_len", "seq_len", "has_context", "expected"),
    [
        (4, 36 * 1024, 36 * 1024, False, True),
        (4, 40 * 1024, 40 * 1024, False, False),
        (4, 4 * 1024, 12 * 1024, True, True),
        (4, 4 * 1024, 16 * 1024, True, False),
        (4, 24 * 1024, 28 * 1024, True, True),
        (4, 32 * 1024, 36 * 1024, True, False),
        (8, 64 * 1024, 64 * 1024, False, True),
        (8, 68 * 1024, 68 * 1024, False, False),
        (8, 4 * 1024, 20 * 1024, True, True),
        (8, 16 * 1024, 32 * 1024, True, True),
        (8, 48 * 1024, 64 * 1024, True, True),
        (8, 56 * 1024, 72 * 1024, True, True),
        (8, 63 * 1024, 79 * 1024, True, False),
    ],
)
def test_glm5_flashinfer_masked_mha_routing(
    tensor_parallel_size, query_len, seq_len, has_context, expected
):
    assert (
        _use_masked_mha(
            backend_name="FLASHINFER_MLA_SPARSE",
            tensor_parallel_size=tensor_parallel_size,
            qk_head_dim=256,
            v_head_dim=256,
            query_len=query_len,
            seq_len=seq_len,
            has_context=has_context,
        )
        is expected
    )


@pytest.mark.parametrize("qk_rope_head_dim", [64, 0], ids=["rope", "nope"])
def test_concat_k_nope_k_pe_matches_torch_cat(qk_rope_head_dim):
    """The K concat used by the MLA prefill context loop must equal torch.cat of
    k_nope with the broadcast k_pe; with no RoPE part it returns k_nope itself
    instead of allocating and copying."""
    torch.manual_seed(0)
    num_tokens, num_heads, qk_nope_head_dim = 5, 4, 256
    k_nope = torch.randn(num_tokens, num_heads, qk_nope_head_dim, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, 1, qk_rope_head_dim, dtype=torch.bfloat16)
    impl = SimpleNamespace(_use_flashinfer_concat_mla_k=False)

    k = MLACommonBaseImpl._concat_k_nope_k_pe(impl, k_nope, k_pe)

    expected = torch.cat([k_nope, k_pe.expand(-1, num_heads, -1)], dim=-1)
    assert k.shape == (num_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim)
    torch.testing.assert_close(k, expected, rtol=0, atol=0)
    assert (k.data_ptr() == k_nope.data_ptr()) == (qk_rope_head_dim == 0)


def test_masked_mha_routing_is_dimension_specific():
    assert _use_masked_mha(
        backend_name="FLASHMLA_SPARSE",
        tensor_parallel_size=1,
        qk_head_dim=192,
        v_head_dim=128,
        query_len=1536,
        seq_len=2048,
        has_context=True,
    )
    assert not _use_masked_mha(
        backend_name="FLASHMLA_SPARSE",
        tensor_parallel_size=1,
        qk_head_dim=128,
        v_head_dim=128,
        query_len=1536,
        seq_len=2048,
        has_context=True,
    )


@pytest.mark.parametrize(
    ("cache_dtype", "expected_quant_mode"),
    [
        ("auto", KVQuantMode.NONE),
        ("fp8_ds_mla", KVQuantMode.FP8_PER_TENSOR),
    ],
)
def test_mla_kv_cache_spec_uses_layer_cache_dtype(
    cache_dtype: str, expected_quant_mode: KVQuantMode
):
    layer = SimpleNamespace(
        kv_cache_dtype=cache_dtype,
        head_size=576,
        indexer=None,
        non_causal_multi_token_decode=False,
        sliding_window=None,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64), model_config=None
    )

    spec = MLAAttention.get_kv_cache_spec(layer, vllm_config)

    assert isinstance(spec, MLAAttentionSpec)
    assert spec.cache_dtype_str == cache_dtype
    assert spec.kv_quant_mode == expected_quant_mode
    if cache_dtype == "fp8_ds_mla":
        assert spec.page_size_bytes == 64 * 656


@pytest.mark.cpu_test
def test_dcp_chunked_context_accepts_non_virtual_block_aligned_prefix():
    context_lens_cpu = torch.tensor([65, 96], dtype=torch.int32)
    prefill_query_start_loc_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)

    metadata = build_mla_chunked_context_metadata(
        context_lens_cpu=context_lens_cpu,
        prefill_query_start_loc_cpu=prefill_query_start_loc_cpu,
        chunked_prefill_workspace=torch.empty(0),
        chunked_prefill_workspace_size=128,
        block_size=64,
        align_chunk_to_block=True,
        device=torch.device("cpu"),
        dcp_world_size=4,
        dcp_local_block_size=1,
        dcp_virtual_block_size=4,
    )

    assert metadata is not None
    assert metadata.context_lens.tolist() == [65, 96]
    assert [chunk.seq_lens.tolist() for chunk in metadata.chunks] == [[65], [96]]
    assert [chunk.starts.tolist() for chunk in metadata.chunks] == [[0], [0]]
    assert [chunk.local_context_lens_allranks for chunk in metadata.chunks] == [
        [[17, 16, 16, 16]],
        [[24, 24, 24, 24]],
    ]
    assert [chunk.padded_local_seq_lens for chunk in metadata.chunks] == [
        [17],
        [24],
    ]
    assert [chunk.padded_local_cu_seq_lens.tolist() for chunk in metadata.chunks] == [
        [0, 17],
        [0, 24],
    ]
    assert [chunk.cu_seq_lens.tolist() for chunk in metadata.chunks] == [
        [0, 65],
        [0, 96],
    ]
    assert [chunk.num_context_tokens for chunk in metadata.chunks] == [65, 96]
    assert [chunk.num_local_context_tokens for chunk in metadata.chunks] == [17, 24]


def test_mla_post_load_preserves_runtime_weight_addresses(monkeypatch):
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_lora_rank = 2
    layer.num_heads = 2
    layer.qk_nope_head_dim = 3
    layer.v_head_dim = 4
    layer.kv_b_proj = torch.nn.Module()
    layer.kv_b_proj.weight = torch.nn.Parameter(
        torch.arange(28.0, dtype=torch.float16).reshape(14, 2)
    )
    layer.kv_b_proj.quant_method = None
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.is_amx_bmm_enabled = False
    layer.dcp_q_replicate = False
    layer.quant_config = None
    layer.layer_name = "test"
    layer.impl = SimpleNamespace(process_weights_after_loading=lambda act_dtype: None)

    monkeypatch.setattr(
        mla_attention_module, "set_default_quant_scales", lambda *_, **__: None
    )

    with torch.no_grad():
        layer.process_weights_after_loading(torch.float32)
        assert isinstance(layer.W_UV, torch.nn.Parameter)
        assert isinstance(layer.W_UK_T, torch.nn.Parameter)
        w_uv_ptr = layer.W_UV.data_ptr()
        w_uk_t_ptr = layer.W_UK_T.data_ptr()
        old_w_uv = layer.W_UV.clone()
        old_w_uk_t = layer.W_UK_T.clone()

        layer.kv_b_proj.weight.add_(100)
        layer.process_weights_after_loading(torch.float32)

    assert layer.W_UV.data_ptr() == w_uv_ptr
    assert layer.W_UK_T.data_ptr() == w_uk_t_ptr
    torch.testing.assert_close(layer.W_UV, old_w_uv + 100)
    torch.testing.assert_close(layer.W_UK_T, old_w_uk_t + 100)


def test_mock_mla_dcp_fp8_decode_gathers_quantized_query(
    monkeypatch, default_vllm_config
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FP8 decode query quantization path.")

    device = torch.device(f"{DEVICE_TYPE}:0")
    num_tokens = 2
    num_heads = 2
    qk_nope_head_dim = 4
    qk_rope_head_dim = 2
    v_head_dim = 3
    kv_lora_rank = 5

    class _DummyKVProj:
        def __init__(self):
            # Shape expected by MockMLAAttentionLayer.__init__
            self.weight = torch.randn(
                num_heads * (qk_nope_head_dim + v_head_dim),
                kv_lora_rank,
                device=device,
                dtype=torch.float32,
            )

    class _FakeImpl:
        def __init__(self):
            self.kv_cache_dtype = "fp8"
            self.supports_quant_query_input = True
            self.dcp_world_size = 2
            self.forward_q = None

        def forward_mha(self, *args, **kwargs):
            return None

        def forward_mqa(self, q, kv_cache, attn_metadata, layer):
            self.forward_q = q
            assert isinstance(q, torch.Tensor)
            bsz, _, _ = q.shape
            return (
                torch.zeros(
                    bsz,
                    num_heads,
                    kv_lora_rank,
                    device=q.device,
                    dtype=torch.float32,
                ),
                None,
            )

    class _FakeDCPGroup:
        def __init__(self):
            self.calls = 0
            self.input_dtype = None
            self.input_shape = None

        def all_gather(self, x, dim=1):
            self.calls += 1
            self.input_dtype = x.dtype
            self.input_shape = tuple(x.shape)
            return torch.cat([x, x], dim=dim)

    fake_group = _FakeDCPGroup()
    monkeypatch.setattr(mla_attention_module, "get_dcp_group", lambda: fake_group)

    impl = _FakeImpl()
    with set_current_vllm_config(default_vllm_config):
        layer = MockMLAAttentionLayer(
            impl=impl,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            device=device,
            kv_b_proj=_DummyKVProj(),
            q_scale=1.0,
            k_scale=1.0,
        )

    q = torch.randn(
        num_tokens,
        num_heads,
        qk_nope_head_dim + qk_rope_head_dim,
        device=device,
        dtype=torch.float32,
    )
    kv_c = torch.randn(num_tokens, kv_lora_rank, device=device, dtype=torch.float32)
    k_pe = torch.randn(
        num_tokens, 1, qk_rope_head_dim, device=device, dtype=torch.float32
    )
    kv_cache = torch.empty(0, device=device, dtype=torch.float32)
    output = torch.empty(
        num_tokens, num_heads * v_head_dim, device=device, dtype=torch.float32
    )

    class _AttnMeta:
        num_decode_tokens = num_tokens
        num_decodes = 1
        num_prefills = 0
        slot_mapping = torch.empty(0, dtype=torch.long, device=device)

    layer.forward_impl(q, kv_c, k_pe, kv_cache, _AttnMeta(), output)

    assert fake_group.calls == 1
    assert fake_group.input_dtype == current_platform.fp8_dtype()
    assert fake_group.input_shape == (
        num_tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
    )
    assert isinstance(impl.forward_q, torch.Tensor)
    assert tuple(impl.forward_q.shape) == (
        num_tokens,
        num_heads * impl.dcp_world_size,
        kv_lora_rank + qk_rope_head_dim,
    )


def test_tokenspeed_mla_noncausal_capability():
    builder = tokenspeed_mla_module.TokenspeedMLAMetadataBuilder
    assert builder.supports_non_causal_multi_token_decode
    assert builder.supports_non_causal_multi_token_dcp
    assert tokenspeed_mla_module.TokenspeedMLABackend.supports_non_causal()


def test_flashinfer_mla_dspark_dcp_supports_target_and_draft(monkeypatch):
    flashinfer_mla_module = pytest.importorskip(
        "vllm.v1.attention.backends.mla.flashinfer_mla"
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        model_config=None,
    )
    monkeypatch.setattr(
        flashinfer_mla_module,
        "get_current_vllm_config",
        lambda: vllm_config,
    )

    backend = flashinfer_mla_module.FlashInferMLABackend
    builder = flashinfer_mla_module.FlashInferMLAMetadataBuilder
    reason = backend.supports_combination(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=SimpleNamespace(),
    )

    assert reason is None
    assert backend.supports_non_causal()
    assert builder.supports_non_causal_multi_token_decode
    assert backend.supports_non_causal_dcp()


@pytest.mark.parametrize(
    ("causal", "tokens_per_decode", "dcp_world_size", "dcp_rank"),
    [
        pytest.param(True, 1, 2, 1, id="causal-dcp"),
        pytest.param(False, 3, 1, 0, id="noncausal-multi-token"),
    ],
)
def test_tokenspeed_mla_decode_contract(
    monkeypatch, causal, tokens_per_decode, dcp_world_size, dcp_rank
):
    decode_call = None
    num_decodes = 2
    num_decode_tokens = num_decodes * tokens_per_decode
    num_heads = 128
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_size = kv_lora_rank + qk_rope_head_dim
    block_size = 64
    num_blocks = 4
    max_seq_len = 24

    def fake_decode(**kwargs):
        nonlocal decode_call
        decode_call = kwargs
        q = kwargs["query"]
        out = torch.empty(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            kv_lora_rank,
            dtype=torch.bfloat16,
        )
        lse = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32)
        return out, lse

    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_mla",
        SimpleNamespace(tokenspeed_mla_decode=fake_decode),
    )

    impl = object.__new__(tokenspeed_mla_module.TokenspeedMLAImpl)
    impl.dcp_world_size = dcp_world_size
    impl.dcp_rank = dcp_rank
    impl.cp_kv_cache_interleave_size = 1
    impl.need_to_return_lse_for_decode = True
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = qk_rope_head_dim
    impl.num_heads = num_heads
    impl.scale = 1.0
    impl.softmax_scale = 1.0
    impl.output_scale = 1.0
    impl._workspace_buffer = torch.empty(1, dtype=torch.int8)

    metadata = SimpleNamespace(
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        max_seq_len=max_seq_len,
        causal=causal,
        decode=SimpleNamespace(
            block_table=torch.empty((num_decodes, 1), dtype=torch.int32),
            seq_lens=torch.tensor([16, max_seq_len], dtype=torch.int32),
            dcp_tot_seq_lens=torch.tensor([16, max_seq_len], dtype=torch.int32),
        ),
    )
    q = torch.empty(num_decode_tokens, num_heads, head_size, dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty(
        num_blocks,
        block_size,
        head_size,
        dtype=torch.float8_e4m3fn,
    )

    out, lse = impl.forward_mqa(
        q,
        kv_cache,
        metadata,
        SimpleNamespace(_q_scale_float=2.0, _k_scale_float=3.0),
    )

    assert out.shape == (num_decode_tokens, num_heads, kv_lora_rank)
    assert lse is not None
    assert lse.shape == (num_decode_tokens, num_heads)

    assert decode_call is not None
    assert decode_call["query"].shape == (
        num_decodes,
        tokens_per_decode,
        num_heads,
        head_size,
    )
    torch.testing.assert_close(decode_call["seq_lens"], metadata.decode.seq_lens)
    torch.testing.assert_close(decode_call["block_tables"], metadata.decode.block_table)
    if dcp_world_size > 1:
        torch.testing.assert_close(
            decode_call["causal_seqs"], metadata.decode.dcp_tot_seq_lens
        )
    else:
        assert decode_call["causal_seqs"] is None
    assert decode_call["causal_mask"] is causal
    assert decode_call["return_lse"] is True
    assert decode_call["cp_world"] == dcp_world_size
    assert decode_call["cp_rank"] == dcp_rank


@pytest.mark.parametrize("is_fp8_kvcache", [False, True], ids=["bf16", "fp8"])
def test_flashmla_dcp_decode_metadata_uses_gathered_query_heads(
    monkeypatch, is_fp8_kvcache
):
    class _FakeSchedulerMetadata:
        tile_scheduler_metadata = None
        num_splits = None

    base_call: tuple[torch.Tensor, int, int, bool] | None = None
    fp8_call: tuple[torch.Tensor, int, int] | None = None

    def fake_get_mla_metadata(
        seq_lens_device,
        num_q_tokens_per_head_k,
        num_heads_k,
        is_fp8_kvcache=False,
    ):
        nonlocal base_call
        base_call = (
            seq_lens_device,
            num_q_tokens_per_head_k,
            num_heads_k,
            is_fp8_kvcache,
        )
        return _FakeSchedulerMetadata(), None

    def fake_get_mla_metadata_dense_fp8(
        seq_lens_device, num_q_tokens_per_head_k, num_heads_k
    ):
        nonlocal fp8_call
        fp8_call = (
            seq_lens_device,
            num_q_tokens_per_head_k,
            num_heads_k,
        )
        return (
            torch.empty((0, 8), dtype=torch.int32),
            torch.empty((0,), dtype=torch.int32),
        )

    monkeypatch.setattr(flashmla_module, "get_mla_metadata", fake_get_mla_metadata)
    monkeypatch.setattr(
        flashmla_module,
        "get_mla_metadata_dense_fp8",
        fake_get_mla_metadata_dense_fp8,
    )

    builder = object.__new__(flashmla_module.FlashMLAMetadataBuilder)
    builder.num_q_heads = 4
    builder.dcp_world_size = 2
    builder.is_fp8_kvcache = is_fp8_kvcache
    builder.compilation_config = type(
        "_CompilationConfig",
        (),
        {
            "cudagraph_mode": type(
                "_CudaGraphMode",
                (),
                {"has_full_cudagraphs": lambda self: False},
            )()
        },
    )()

    seq_lens = torch.tensor([16, 24], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)

    metadata = builder._build_decode(
        block_table_tensor=torch.empty((2, 1), dtype=torch.int32),
        seq_lens_device=seq_lens,
        max_seq_len=24,
        query_start_loc_cpu=query_start_loc,
        query_start_loc_device=query_start_loc,
        num_decode_tokens=2,
        dcp_tot_seq_lens_device=None,
    )

    assert base_call is not None
    assert base_call[0] is seq_lens
    assert base_call[1:] == (8, 1, is_fp8_kvcache)
    if is_fp8_kvcache:
        assert metadata.scheduler_metadata.tile_scheduler_metadata is not None
        assert metadata.scheduler_metadata.num_splits is not None
        assert fp8_call is not None
        assert fp8_call[0] is seq_lens
        assert fp8_call[1:] == (8, 1)
    else:
        assert metadata.scheduler_metadata.tile_scheduler_metadata is None
        assert metadata.scheduler_metadata.num_splits is None
        assert fp8_call is None
