# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.nvidia import triton_sparse as backend
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v41.nvidia.model import _select_dsv4_attn_cls
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

TritonAttention = backend.DeepseekV41TritonSparseAttention


@pytest.fixture(autouse=True)
def _sm90(monkeypatch):
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_attention_heads=64, head_dim=512, qk_rope_head_dim=64
            ),
            dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.TRITON_MLA_SPARSE_DSV41
        ),
    )


@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("kv_dtype", ["auto", "fp8", "fp8_ds_mla"])
def test_explicit_backend_keeps_native_head_buffers(tp_size, kv_dtype):
    config = _config()
    config.parallel_config.tensor_parallel_size = tp_size
    config.cache_config.cache_dtype = kv_dtype
    cls = _select_dsv4_attn_cls(config)
    assert cls is TritonAttention
    assert cls.get_padded_num_q_heads(64 // tp_size) == 64 // tp_size
    assert not issubclass(cls, DeepseekV4FlashMLAAttention)


@pytest.mark.parametrize(
    "selected",
    [
        None,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
        AttentionBackendEnum.FLASHMLA_SPARSE,
    ],
)
def test_existing_backend_choices_remain_on_flashmla(selected, monkeypatch):
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4MegaAttnAttention

    config = _config()
    config.attention_config.backend = selected
    monkeypatch.setattr(
        DeepseekV4MegaAttnAttention, "is_available_for", lambda _: False
    )
    assert _select_dsv4_attn_cls(config) is DeepseekV4FlashMLAAttention


@pytest.mark.parametrize("major,mega_available", [(9, False), (10, True), (12, False)])
def test_opt_in_backend_preserves_default_architecture_selection(
    major, mega_available, monkeypatch
):
    from vllm.models.deepseek_v41.nvidia import model

    config = _config()
    config.attention_config.backend = None
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major, 0)),
    )
    monkeypatch.setattr(
        model.DeepseekV4MegaAttnAttention, "is_available_for", lambda _: mega_available
    )
    expected = {
        9: DeepseekV4FlashMLAAttention,
        10: model.DeepseekV4MegaAttnAttention,
        12: model.DeepseekV4FlashInferSM120Attention,
    }
    assert _select_dsv4_attn_cls(config) is expected[major]


@pytest.mark.parametrize("major", [8, 10, 12])
def test_explicit_backend_rejects_unsupported_architecture(major, monkeypatch):
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major, 0)),
    )
    with pytest.raises(ValueError, match="requires SM90"):
        _select_dsv4_attn_cls(_config())


@pytest.mark.parametrize("heads,tp", [(64, 2), (63, 8), (0, 8), (64, 0)])
def test_explicit_backend_rejects_incompatible_head_topology(heads, tp):
    config = _config()
    config.model_config.hf_text_config.num_attention_heads = heads
    config.parallel_config.tensor_parallel_size = tp
    with pytest.raises(ValueError, match="8 or 16 local heads"):
        _select_dsv4_attn_cls(config)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("head_dim", 256, "512D heads"),
        ("qk_rope_head_dim", 128, "64D RoPE"),
    ],
)
def test_explicit_backend_rejects_incompatible_dimensions(field, value, error):
    config = _config()
    setattr(config.model_config.hf_text_config, field, value)
    with pytest.raises(ValueError, match=error):
        _select_dsv4_attn_cls(config)


@pytest.mark.parametrize("kv_dtype", ["nvfp4_ds_mla", "bfloat16", "fp8_e4m3"])
def test_explicit_backend_rejects_incompatible_cache_layout(kv_dtype):
    config = _config()
    config.cache_config.cache_dtype = kv_dtype
    with pytest.raises(ValueError, match="fp8_ds_mla KV cache"):
        _select_dsv4_attn_cls(config)


def test_explicit_backend_rejects_non_bf16_queries():
    config = _config()
    config.model_config.dtype = torch.float16
    with pytest.raises(ValueError, match="BF16 queries"):
        _select_dsv4_attn_cls(config)


def test_triton_metadata_does_not_create_flashmla_decode_plans(monkeypatch):
    from vllm.v1.attention.backends.mla import sparse_swa

    def fail():
        pytest.fail("Triton metadata must not create a FlashMLA plan")

    monkeypatch.setattr(sparse_swa, "get_mla_metadata", fail)
    cls = backend.DeepseekV41TritonSWAMetadataBuilder
    builder = cls.__new__(cls)
    plans = builder.build_tile_scheduler(64)
    assert len(plans) == 5 and all(value is None for value in plans.values())


@pytest.mark.parametrize(
    "attention_backend",
    [backend.DeepseekV41TritonSparseBackend, backend.DeepseekV41TritonSWABackend],
)
@pytest.mark.parametrize("drafts", [0, 5])
def test_graph_dispatch_keeps_prefill_out_of_full_capture(attention_backend, drafts):
    """CPU chunk plans are not replayable; retain variable-length decode graphs."""
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.gpu.attn_utils import (
        get_attn_cg_support,
        get_varlen_cudagraph_unsupported_backend,
    )
    from vllm.v1.worker.utils import AttentionGroup

    config = _config()
    config.use_v2_model_runner = True
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=drafts, parallel_drafting=False
    )
    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    group = AttentionGroup(attention_backend, ["target"], spec, 0)
    group.metadata_builders = [object.__new__(attention_backend.get_builder_cls())]
    groups = [[group]]
    assert get_attn_cg_support(groups, config).min_cg_support == (
        AttentionCGSupport.UNIFORM_BATCH
    )
    assert get_varlen_cudagraph_unsupported_backend(groups, config, drafts + 1) is None
    rejected = get_varlen_cudagraph_unsupported_backend(groups, config, drafts + 2)
    assert rejected is not None and rejected[1] == drafts + 1


def _layer(ratio):
    layer = TritonAttention.__new__(TritonAttention)
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = ratio
    layer.compressed_cache_prefix = "compressed" if ratio else None
    layer.n_local_heads = 8
    layer.scale = 512**-0.5
    layer.attn_sink = torch.zeros(8)
    layer.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=torch.empty((1, 64, 584), dtype=torch.uint8)
    )
    return layer


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_mixed_batch_writes_decode_and_prefill_to_their_original_slices(
    ratio, monkeypatch
):
    """Splitting native-width output must not swap or overwrite request rows."""
    layer = _layer(ratio)
    compressed_cache = torch.empty((1, 64, 584), dtype=torch.uint8)
    layer._compressed_kv_cache = lambda: compressed_cache
    swa = SimpleNamespace(num_decodes=2, num_prefills=1, num_decode_tokens=3)
    metadata = {"swa": swa, "compressed": object()}
    monkeypatch.setattr(
        backend, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    q = torch.arange(7 * 8 * 512).reshape(7, 8, 512).to(torch.bfloat16)
    positions = torch.arange(7)
    out = torch.full_like(q, -1)

    def prefill(**kwargs):
        torch.testing.assert_close(kwargs["positions"], positions[3:])
        torch.testing.assert_close(kwargs["q"], q[3:])
        assert kwargs["compressed_k_cache"] is (compressed_cache if ratio else None)
        kwargs["output"].fill_(2)

    def decode(**kwargs):
        torch.testing.assert_close(kwargs["q"], q[:3])
        assert kwargs["swa_only"] == (ratio == 0)
        kwargs["output"].fill_(1)

    monkeypatch.setattr(layer, "_forward_prefill", prefill)
    monkeypatch.setattr(layer, "_forward_decode", decode)
    layer.forward_mqa(q, torch.empty(0), positions, out)
    assert torch.all(out[:3] == 1) and torch.all(out[3:] == 2)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_decode_uses_only_decode_rows_and_correct_compressed_page_size(
    ratio, monkeypatch
):
    layer = _layer(ratio)
    layer.topk_indices_buffer = torch.arange(6 * 512).reshape(6, 512)
    kv = (
        torch.empty((1, 64 // max(1, ratio), 584), dtype=torch.uint8) if ratio else None
    )
    swa = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=3,
        is_valid_token=torch.ones(6, dtype=torch.bool),
        token_to_req_indices=torch.zeros(6, dtype=torch.int32),
        decode_swa_indices=torch.zeros((6, 1, 128), dtype=torch.int32),
        decode_swa_lens=torch.full((6,), 128, dtype=torch.int32),
    )
    compressed = SimpleNamespace(block_size=64, block_table=torch.zeros((4, 2)))
    indices = torch.ones((3, 512), dtype=torch.int32)
    lengths = torch.full((3,), 512, dtype=torch.int32)
    calls = []

    def global_indices(topk, token_reqs, block_table, page_size, valid):
        assert ratio > 0
        assert topk.shape == (3, 512) and valid.shape == (3,)
        assert block_table.shape == (2, 2) and page_size == 64 // ratio
        return indices, lengths

    def decode(**kwargs):
        assert kwargs["extra_cache"] is kv
        if ratio:
            torch.testing.assert_close(kwargs["extra_indices"].reshape(3, 512), indices)
            assert kwargs["extra_lens"] is lengths
        else:
            assert kwargs["extra_indices"] is kwargs["extra_lens"] is None
        assert kwargs["num_heads"] == 8
        kwargs["out"].fill_(3)
        calls.append(1)

    monkeypatch.setattr(backend, "compute_global_topk_indices_and_lens", global_indices)
    monkeypatch.setattr(backend, "_run_decode", decode)
    q = torch.zeros((3, 8, 512), dtype=torch.bfloat16)
    out = torch.empty_like(q)
    layer._forward_decode(q, kv, swa, compressed if ratio else None, ratio == 0, out)
    assert calls == [1] and torch.all(out == 3)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_chunked_prefill_excludes_decode_rows_from_topk_and_output(ratio, monkeypatch):
    layer = _layer(ratio)
    layer.window_size = 128
    layer.max_num_batched_tokens = 8
    layer.topk_indices_buffer = torch.arange(8 * 512).reshape(8, 512)
    starts = torch.tensor([0, 1, 3, 5, 8], dtype=torch.int32)
    swa = SimpleNamespace(
        num_prefill_tokens=5,
        num_decodes=2,
        num_decode_tokens=3,
        prefill_seq_lens=torch.tensor([32, 33]),
        prefill_gather_lens=torch.tensor([2, 3]),
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        block_table=torch.zeros((4, 2), dtype=torch.int32),
        block_size=64,
        get_prefill_chunk_plan=lambda **_: [(0, 1, 32, 162), (1, 2, 33, 164)],
    )
    compressed = SimpleNamespace(block_size=64, block_table=swa.block_table)
    manager = SimpleNamespace(
        get_simultaneous=lambda *specs: [
            torch.zeros(shape, dtype=dtype) for shape, dtype in specs
        ]
    )
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(backend, "dequantize_and_gather_k_cache", lambda *a, **kw: None)
    rows = []

    def combine(topk, query_starts, *args, out):
        rows.append(topk.clone())
        return out

    def prefill(q, kv, indices, scale, *, out, **kwargs):
        out.copy_(q)

    monkeypatch.setattr(backend, "combine_topk_swa_indices", combine)
    monkeypatch.setattr(backend, "run_fewhead_sparse_prefill", prefill)
    q = torch.arange(5 * 8 * 512).reshape(5, 8, 512).to(torch.bfloat16)
    storage = torch.full((7, 8, 512), -1, dtype=torch.bfloat16)
    layer._forward_prefill(
        q,
        torch.arange(5),
        torch.empty(0) if ratio else None,
        torch.empty(0),
        storage[1:6],
        compressed if ratio else None,
        swa,
    )
    torch.testing.assert_close(rows[0], layer.topk_indices_buffer[3:5])
    torch.testing.assert_close(rows[1], layer.topk_indices_buffer[5:8])
    torch.testing.assert_close(storage[1:6], q)
    assert torch.all(storage[0] == -1) and torch.all(storage[-1] == -1)


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("dspark", [False, True])
def test_decode_warmup_covers_native_launcher_specializations(
    monkeypatch, heads, dspark
):
    """Runtime dispatch must be covered across split and compressed-page boundaries."""
    import inspect

    from vllm.model_executor.warmup.jit_warmup_triton_helper import (
        TritonWarmupTensor,
        triton_scalar_specialization_rep,
    )
    from vllm.models.deepseek_v41.nvidia.ops import small_head_sparse_decode as decode
    from vllm.models.deepseek_v41.nvidia.ops.small_head_decode_warmup import (
        _decode_warmup_inputs,
        _merge_warmup_inputs,
    )

    config = _config()
    config.parallel_config.tensor_parallel_size = 64 // heads
    config.model_config.hf_text_config.sliding_window = 128
    config.model_config.hf_text_config.index_topk = 512
    config.model_config.hf_text_config.compress_ratios = [0, 1, 2]
    config.speculative_config = (
        SimpleNamespace(use_dspark=lambda: True, num_speculative_tokens=5)
        if dspark
        else None
    )

    def signature(inputs):
        result = []
        for name, value in sorted(inputs.items()):
            if name == "grid":
                continue
            if isinstance(value, torch.Tensor | TritonWarmupTensor):
                value = value.dtype
            elif type(value) is int and name.islower() and not name.startswith("num_"):
                value = triton_scalar_specialization_rep(value)
            result.append((name, value))
        return tuple(result)

    for name, provider in (
        ("_small_head_sparse_decode_kernel", _decode_warmup_inputs),
        ("_merge_splits_kernel", _merge_warmup_inputs),
    ):
        kernel = getattr(decode, name)
        names = tuple(inspect.signature(getattr(kernel, "fn", kernel)).parameters)
        expected = {signature(case) for case in provider(config, 64)}

        class CheckedLaunch:
            def __init__(self, names, expected):
                self.names, self.expected = names, expected

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    inputs = dict(zip(self.names, args)) | kwargs
                    assert signature(inputs) in self.expected

                return launch

        monkeypatch.setattr(decode, name, CheckedLaunch(names, expected))

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    for tokens in (1, 3, 4, 7, 8, 384):
        for width in (128, 192) if dspark else (128,):
            for ratio in (0, 1, 2):
                page = 64 // ratio if ratio else 64
                q = empty((tokens + 1, heads, 512), torch.bfloat16)[1:]
                decode.small_head_sparse_decode(
                    q,
                    empty((2, 64, 584), torch.uint8),
                    empty((tokens + 8, 1, width), torch.int32),
                    empty((tokens + 8,), torch.int32),
                    empty((2, page, 584), torch.uint8) if ratio else None,
                    empty((tokens + 8, 1, 512), torch.int32) if ratio else None,
                    empty((tokens + 8,), torch.int32) if ratio else None,
                    empty((heads,), torch.float32),
                    512**-0.5,
                    torch.empty_like(q),
                    heads,
                )


@pytest.mark.parametrize("heads", [8, 16])
def test_prefill_warmup_expands_native_and_padded_layouts(heads):
    """Warmup domains must resolve through the registry's AST expansion on CPU."""
    from vllm.models.deepseek_v41.nvidia.triton_fewhead_sparse_prefill import (
        _bf16_fewhead_sparse_fwd,
        _fewhead_warmup_inputs,
    )

    config = _config()
    config.parallel_config.tensor_parallel_size = 64 // heads
    config.model_config.hf_text_config.sliding_window = 128
    config.model_config.hf_text_config.index_topk = 512
    config.model_config.hf_text_config.compress_ratios = [0, 1, 2]
    cases = list(
        _bf16_fewhead_sparse_fwd._provider_cases(
            _fewhead_warmup_inputs, vllm_config=config
        )
    )
    for field in ("stride_q_s", "stride_o_s"):
        assert {case[field] for case in cases} == {heads * 512, 64 * 512}
    assert {case["h_q"] for case in cases} == {heads}
    assert {case["index_topk"] for case in cases} == {128, 640}


def test_registered_decode_warmup_expands_with_model_config(monkeypatch):
    """Registrations with block_size must retain config through startup warmup."""
    from vllm.model_executor.warmup import jit_warmup_triton_helper as helper
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v41.nvidia.ops.small_head_decode_warmup import (
        register_small_head_decode_warmup,
    )

    config = _config()
    config.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192, max_num_seqs=64
    )
    config.model_config.hf_text_config.sliding_window = 128
    config.model_config.hf_text_config.index_topk = 512
    config.model_config.hf_text_config.compress_ratios = [0, 1, 2]
    config.speculative_config = SimpleNamespace(
        use_dspark=lambda: True, num_speculative_tokens=5
    )
    expanded = []

    def collect_inputs(inputs):
        expanded.append(inputs)
        return ()

    monkeypatch.setattr(helper, "_triton_key_deriver", lambda _: collect_inputs)
    registry = JitWarmupRegistry(config)
    with registry.activate():
        register_small_head_decode_warmup(config, 64)
    registry.warmup()

    decode = [case for case in expanded if "num_heads" in case]
    assert {case["num_heads"] for case in decode} == {8}
    assert {case["swa_indices_stride"] for case in decode} == {128, 192}
    assert {case["extra_page_size"] for case in decode} == {32, 64}
    assert any("sink_ptr" in case for case in expanded)
    assert {case["B"] for case in expanded if "B" in case} == {32, 64, 128, 256, 512}


def test_query_union_state_is_recreated_for_each_forward():
    """Repeated buffers must not make a new backbone step reuse stale indices."""
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v41.nvidia.paired_decode import decode_step

    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL, additional_kwargs={}
    )
    key = "decode"
    first = decode_step(context, key, 2, 64)
    first.pairs_ready = True
    first.groups[(2, 2, 1)] = (1000,)
    assert decode_step(context, key, 3, 64) is first
    second = decode_step(context, key, 2, 64)
    assert not second.pairs_ready and not second.groups
    assert decode_step(context, key, 4, 96) is not second
    other_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL, additional_kwargs={}
    )
    assert decode_step(other_context, key, 2, 64) is not second


def test_piecewise_query_unions_have_a_producer_in_each_layer():
    """A captured segment cannot rely on Python state from an earlier segment."""
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v41.nvidia.paired_decode import decode_step

    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, additional_kwargs={}
    )
    first = decode_step(context, "decode", 2, 64)
    first.pairs_ready = True
    assert not decode_step(context, "decode", 3, 64).pairs_ready


@pytest.mark.parametrize(
    "heads,layer,ratio,enabled",
    [(8, 2, 1, True), (16, 2, 1, False), (8, 40, 1, False), (8, 0, 0, False)],
)
def test_paired_decode_keeps_draft_and_unsupported_layouts_on_single_query(
    heads, layer, ratio, enabled
):
    from vllm.models.deepseek_v41.nvidia.paired_decode import PairedDecode

    config = _config()
    config.model_config.hf_text_config.num_hidden_layers = 40
    config.model_config.hf_text_config.index_topk = 512
    config.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8192, max_num_seqs=64
    )
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=5, use_dspark=lambda: True
    )
    attention = SimpleNamespace(
        layer_id=layer,
        n_local_heads=heads,
        compress_ratio=ratio,
        kv_source_layer_id=2,
        index_source_layer_id=2,
        window_size=128,
    )
    decoder = PairedDecode(attention, config)
    assert decoder.enabled == enabled
    assert decoder.capacity == 384
    assert decoder.swa_width == 192
