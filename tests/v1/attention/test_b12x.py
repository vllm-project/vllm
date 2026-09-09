# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.test_attention_backends import (
    BATCH_SPECS,
    _test_backend_correctness,
)
from tests.v1.attention.utils import BatchSpec
from vllm.config import AttentionConfig, ModelConfig, set_current_vllm_config
from vllm.model_executor.kernels.attention import b12x_mla_query
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    _canonicalize_sparse_mla_kv_cache_dtype,
)
from vllm.models.deepseek_v32.attention import DeepseekV32Attention
from vllm.models.deepseek_v32.nvidia.b12x import (
    B12xDSAIndexer,
    DeepseekV32B12xAttention,
)
from vllm.models.deepseek_v32.nvidia.model import _get_attention_cls
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.b12x import get_b12x_paged_attention
from vllm.v1.attention.backends import b12x
from vllm.v1.attention.backends.b12x import (
    B12xPagedAttentionBackend,
    B12xPagedAttentionImpl,
    _kv_page_size,
    _max_page_table_width,
)
from vllm.v1.attention.backends.mla import (
    b12x_indexer,
    b12x_mla_sparse,
)
from vllm.v1.attention.backends.mla import (
    indexer as mla_indexer,
)
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseBackend,
    B12xMLASparseImpl,
    B12xMLASparseMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout


def _require_b12x_paged_attention() -> None:
    capability = current_platform.get_device_capability()
    if (
        not current_platform.is_cuda()
        or capability is None
        or not B12xPagedAttentionBackend.supports_compute_capability(capability)
    ):
        pytest.skip("b12x paged attention requires SM120 or SM121.")

    paged_attention = get_b12x_paged_attention()
    if paged_attention is None or not paged_attention.is_supported():
        pytest.skip("b12x paged attention is not available.")


class _Workspace:
    def get_simultaneous(self, *shapes_and_dtypes):
        return [torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes]


def test_b12x_bf16_mla_query_uses_public_run_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    module = SimpleNamespace(run=lambda *args: calls.append(args))
    monkeypatch.setattr(b12x_mla_query, "get_b12x_mla_query_projection", lambda: module)
    q_nope = torch.empty((8, 2, 192), dtype=torch.bfloat16)
    weight = torch.empty((8, 192, 512), dtype=torch.bfloat16)
    q_pe = torch.empty((2, 8, 64), dtype=torch.bfloat16)
    output = torch.empty((2, 8, 576), dtype=torch.bfloat16)

    b12x_mla_query._b12x_bf16_mla_query_impl(q_nope, weight, q_pe, output)

    assert calls == [(q_nope, weight, q_pe, output)]


def test_b12x_bf16_mla_query_uses_backend_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.q_pad_num_heads = None
    layer.kv_lora_rank = 512
    layer.qk_rope_head_dim = 64
    layer.W_UK_T = torch.nn.Parameter(
        torch.empty((8, 192, 512), dtype=torch.bfloat16), requires_grad=False
    )
    workspace = torch.empty((2, 8, 576), dtype=torch.bfloat16)
    layer.impl = SimpleNamespace(
        supports_fused_mla_query_output=lambda *args: True,
        get_fused_mla_query_output=lambda *args: workspace,
    )
    calls = []
    monkeypatch.setattr(
        mla_attention, "can_implement_bf16_mla_query", lambda **kwargs: True
    )

    def run_query(*args):
        calls.append(args)
        return args[-1]

    monkeypatch.setattr(
        mla_attention,
        "run_bf16_mla_query",
        run_query,
    )
    q_nope = torch.empty((8, 2, 192), dtype=torch.bfloat16)
    q_pe = torch.empty((2, 8, 64), dtype=torch.bfloat16)

    result = layer._try_fused_mla_query(q_nope, q_pe)

    assert result is workspace
    assert calls == [(q_nope, layer.W_UK_T, q_pe, workspace)]


def test_b12x_bf16_mla_query_requires_backend_capability() -> None:
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.q_pad_num_heads = None
    layer.impl = SimpleNamespace()

    q_nope = torch.empty((8, 2, 192), dtype=torch.bfloat16)
    q_pe = torch.empty((2, 8, 64), dtype=torch.bfloat16)

    assert layer._try_fused_mla_query(q_nope, q_pe) is None


def test_mla_preallocates_absorbed_weights_before_dequantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.impl = SimpleNamespace(
        process_weights_after_loading=lambda dtype: None,
        warmup=lambda token_counts: None,
    )
    layer.is_amx_bmm_enabled = False
    layer.kv_lora_rank = 2
    layer.num_heads = 2
    layer.qk_nope_head_dim = 3
    layer.v_head_dim = 4
    layer.kv_b_proj = torch.nn.Module()
    layer.kv_b_proj.qweight = torch.nn.Parameter(
        torch.ones((14, 2), dtype=torch.int8), requires_grad=False
    )
    layer.kv_b_proj.quant_method = object()
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.dcp_q_replicate = False
    layer.quant_config = None
    layer.layer_name = "test"
    dequantized = torch.arange(28.0, dtype=torch.float32).reshape(14, 2)
    events = []
    preallocated = []
    preallocate = mla_attention._preallocate_absorbed_mla_weights

    def track_preallocation(*args, **kwargs):
        events.append("preallocate")
        weights = preallocate(*args, **kwargs)
        preallocated.extend(weights)
        return weights

    def fake_dequant(*args, **kwargs):
        events.append("dequantize")
        return dequantized

    monkeypatch.setattr(
        mla_attention,
        "_preallocate_absorbed_mla_weights",
        track_preallocation,
    )
    monkeypatch.setattr(mla_attention, "get_and_maybe_dequant_weights", fake_dequant)
    monkeypatch.setattr(mla_attention, "set_default_quant_scales", lambda *a, **k: None)

    with torch.no_grad():
        layer.process_weights_after_loading(torch.float32)

    assert events == ["preallocate", "dequantize"]
    assert layer.W_UV.data_ptr() == preallocated[0].data_ptr()
    assert layer.W_UK_T.data_ptr() == preallocated[1].data_ptr()
    assert layer.b12x_warmup_provider is layer
    assert "b12x_warmup_provider" not in layer._modules


def test_b12x_selector_routes_glm_dsa_internally() -> None:
    config = SimpleNamespace(
        attention_config=AttentionConfig(backend="b12x"),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
        ),
    )

    assert config.attention_config.backend == AttentionBackendEnum.B12X
    assert AttentionBackendEnum.B12X.get_class() is B12xPagedAttentionBackend
    assert _get_attention_cls(config) is DeepseekV32B12xAttention
    assert DeepseekV32B12xAttention.indexer_cls is B12xDSAIndexer
    assert B12xMLASparseBackend.get_name() == "B12X"

    config.attention_config = AttentionConfig()
    assert _get_attention_cls(config) is DeepseekV32Attention


def test_b12x_sparse_mla_accepts_glm_dsa_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        b12x_mla_sparse,
        "get_b12x_sparse_mla",
        lambda: SimpleNamespace(is_supported=lambda: True),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="glm_moe_dsa",
                index_topk=2048,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            )
        )
    )

    with set_current_vllm_config(config):
        reason = B12xMLASparseBackend.supports_combination(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=DeviceCapability(12, 0),
        )

    assert reason is None
    assert B12xMLASparseBackend.get_preferred_block_size(16) == 64
    assert (
        _canonicalize_sparse_mla_kv_cache_dtype(B12xMLASparseBackend, "auto")
        == "fp8_ds_mla"
    )


def test_b12x_dsa_uses_rank_local_physical_slots_with_dcp() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_n_heads=16)),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
        ),
    )

    kwargs = B12xDSAIndexer.get_indexer_op_kwargs(config)

    assert kwargs["output_physical_slots"] is True


def test_b12x_sparse_mla_writes_rank_local_selected_lengths() -> None:
    global_lengths = torch.tensor([1, 2, 3, 4, 4097], dtype=torch.int32)
    output = torch.empty_like(global_lengths)

    b12x_mla_sparse._write_rank_local_selected_lengths(
        global_lengths,
        output,
        dcp_size=2,
        dcp_rank=0,
        interleave_size=1,
        topk=2048,
    )

    torch.testing.assert_close(
        output,
        torch.tensor([1, 1, 2, 2, 2048], dtype=torch.int32),
    )


def test_b12x_sparse_mla_replans_for_bound_page_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = []

    class FakeCaps(SimpleNamespace):
        pass

    class FakeModule:
        Caps = FakeCaps

        @staticmethod
        def plan(caps):
            planned.append(caps)
            return SimpleNamespace(caps=caps, layout=SimpleNamespace(nbytes=96))

    impl = object.__new__(B12xMLASparseImpl)
    impl._module = FakeModule
    impl._kernel_page_size = 0
    impl._input_num_heads = 8
    impl._max_tokens = 16
    impl._max_seqs = 4
    impl._topk_tokens = 2048
    impl._kv_dtype = torch.uint8
    impl._q_head_dim = 576
    impl.kv_lora_rank = 512
    impl.scale = 576**-0.5
    impl.need_to_return_lse_for_decode = False
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.impl = impl
    layer._vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(enable_jit_warmup=False)
    )
    layer.bind_kv_cache(torch.empty((2, 1, 128, 656), dtype=torch.uint8))

    assert impl._kernel_page_size == 128
    assert layer.kv_cache.shape == (2, 128, 656)
    assert [(caps.mode, caps.page_size) for caps in planned] == [
        ("decode", 128),
        ("extend", 128),
    ]
    assert B12xMLASparseBackend.supported_kv_cache_layouts() == (KVCacheLayout.BLHNC,)


def test_b12x_sparse_mla_dcp_uses_local_selection_and_returns_lse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}
    physical_indices = torch.zeros((2, 4), dtype=torch.int32)
    active_counts = torch.tensor([2, 3], dtype=torch.int32)
    expected_output = torch.empty((2, 4, 512))
    expected_lse = torch.empty((2, 4))

    def bind(plan, **kwargs):
        calls["bind"] = kwargs
        return object()

    impl = object.__new__(B12xMLASparseImpl)
    impl._decode_plan = object()
    impl._extend_plan = object()
    impl._scratch_nbytes = 64
    impl._q_spec = ((2, 4, 576), torch.bfloat16)
    impl._scratch_spec = ((64,), torch.uint8)
    impl._max_tokens = 2
    impl._input_num_heads = 4
    impl._q_head_dim = 576
    impl._topk_tokens = 4
    impl.topk_indices_buffer = physical_indices
    impl.dcp_world_size = 2
    impl._kernel_page_size = 64
    impl.kv_lora_rank = 512
    impl.scale = 576**-0.5
    impl.need_to_return_lse_for_decode = True
    impl._bind = bind
    impl._run = lambda binding: (expected_output, expected_lse)

    monkeypatch.setattr(
        b12x_mla_sparse, "current_workspace_manager", lambda: _Workspace()
    )
    metadata = SimpleNamespace(
        max_query_len=1,
        num_reqs=2,
        num_prefills=0,
        num_decode_tokens=2,
        seq_lens=torch.tensor([64, 64], dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        block_size=64,
        cp_kv_cache_interleave_size=1,
        cache_seq_lens_per_token=active_counts,
    )

    output, lse = impl.forward_mqa(
        (
            torch.empty((2, 4, 576), dtype=torch.bfloat16),
            torch.empty((2, 4, 0), dtype=torch.bfloat16),
        ),
        torch.empty((2, 64, 656), dtype=torch.uint8),
        metadata,
        SimpleNamespace(),
    )

    assert calls["bind"]["selected_indices"].data_ptr() == physical_indices.data_ptr()
    assert calls["bind"]["selected_lengths"].data_ptr() == active_counts.data_ptr()
    assert output is expected_output
    assert lse is expected_lse


def test_b12x_dsa_indexer_writes_to_caller_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    binding = object()

    def bind(plan, **kwargs):
        calls["plan"] = plan
        calls["bind"] = kwargs
        return binding

    plan = SimpleNamespace(shapes_and_dtypes=lambda: (((64,), torch.uint8),))

    def run(actual_binding):
        calls["run"] = actual_binding
        calls["bind"]["output_indices"].fill_(11)

    module = SimpleNamespace(
        PAGED_INDEX_PAGE_SIZE=64,
        bind=bind,
        run=run,
    )
    monkeypatch.setattr(b12x_indexer, "current_workspace_manager", _Workspace)

    output = torch.empty((3, 4), dtype=torch.int32)
    b12x_indexer._run_paged_topk(
        module=module,
        plan=plan,
        q=torch.empty((3, 16, 128), dtype=torch.float8_e4m3fn),
        weights=torch.empty((3, 16), dtype=torch.float32),
        kv_cache=torch.empty((4, 64, 132), dtype=torch.uint8),
        seq_lens=torch.full((3,), 128, dtype=torch.int32),
        block_table=torch.zeros((3, 2), dtype=torch.int32),
        active_width=torch.tensor([128], dtype=torch.int32),
        output=output,
    )

    assert calls["plan"] is plan
    assert calls["bind"]["output_indices"] is output
    assert calls["run"] is binding
    assert torch.count_nonzero(output != 11) == 0


def test_b12x_dsa_indexer_owns_prefill_width_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        plan=lambda caps: SimpleNamespace(caps=caps),
    )
    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: module)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_seqs=8,
        ),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8]),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    with set_current_vllm_config(config):
        indexer = b12x_indexer.B12xSparseIndexer(
            k_cache=object(),
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=4,
            head_dim=128,
            max_model_len=4096,
            max_total_seq_len=4096,
            topk_indices_buffer=torch.empty((8, 4), dtype=torch.int32),
            skip_k_cache_insert=True,
            num_q_heads=16,
            output_physical_slots=True,
        )

    torch.testing.assert_close(
        indexer.active_width_cap,
        torch.tensor([4096], dtype=torch.int32),
    )
    assert all(
        plan.caps.output_index_space == "physical"
        for plan in indexer._decode_plans.values()
    )
    assert indexer._decode_plans[8].caps.max_q_rows == 8
    assert max(indexer._prefill_plans) == 8
    assert indexer.b12x_warmup_provider is indexer
    assert "b12x_warmup_provider" not in indexer._modules


def test_b12x_metadata_builder_uses_backend_prefill_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = object.__new__(b12x_indexer.B12xIndexerMetadataBuilder)
    builder.compress_ratio = 1
    builder.use_pcp = False
    builder.max_prefill_buffer_size = 1024
    builder.dcp_rank = 0
    builder.dcp_world_size = 1
    builder.cp_kv_cache_interleave_size = 1
    builder.decode_threshold = 1
    builder.use_flattening = False
    builder.supports_varlen = False
    builder.active_width_buffer = torch.zeros((1,), dtype=torch.int32)
    builder.cache_seq_lens_per_token_buffer = torch.empty((2,), dtype=torch.int32)

    monkeypatch.setattr(
        mla_indexer,
        "split_decodes_and_prefills",
        lambda *args, **kwargs: (0, 2, 0, 2),
    )
    chunk_requests = []

    def build_chunk(start_idx, end_idx, *args, **kwargs):
        chunk_requests.append((start_idx, end_idx))
        return SimpleNamespace()

    monkeypatch.setattr(mla_indexer, "build_prefill_chunk_metadata", build_chunk)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([32, 64], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([32, 64], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        block_table_tensor=torch.zeros((2, 2), dtype=torch.int32),
        dcp_local_seq_lens=None,
        max_seq_len=64,
        positions=torch.tensor([0, 0], dtype=torch.int64),
    )

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert chunk_requests == [(0, 1), (1, 2)]
    assert metadata.prefill is not None
    assert len(metadata.prefill.chunks) == 2


def test_b12x_sparse_mla_prefill_binds_request_sequence_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    def bind(plan, **kwargs):
        calls["plan"] = plan
        calls["bind"] = kwargs
        return object()

    plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
    )
    impl = object.__new__(B12xMLASparseImpl)
    impl._decode_plan = object()
    impl._extend_plan = plan
    impl._scratch_nbytes = 64
    impl._q_spec = ((8, 2, 576), torch.bfloat16)
    impl._scratch_spec = ((64,), torch.uint8)
    impl._max_tokens = 8
    impl._input_num_heads = 2
    impl._q_head_dim = 576
    impl._topk_tokens = 4
    impl.topk_indices_buffer = torch.zeros((8, 4), dtype=torch.int32)
    impl.dcp_world_size = 1
    impl._kernel_page_size = 64
    impl.kv_lora_rank = 512
    impl.scale = 576**-0.5
    impl.need_to_return_lse_for_decode = False
    impl._bind = bind
    impl._run = lambda binding: torch.empty((8, 2, 512))

    monkeypatch.setattr(
        b12x_mla_sparse, "current_workspace_manager", lambda: _Workspace()
    )
    metadata = SimpleNamespace(
        max_query_len=8,
        num_reqs=1,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        req_id_per_token=torch.zeros(8, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=64,
        cp_kv_cache_interleave_size=1,
        cache_seq_lens_per_token=torch.arange(1, 9, dtype=torch.int32),
    )
    q = torch.empty((8, 2, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, 576), dtype=torch.uint8)

    impl.forward_mqa(q, kv_cache, metadata, SimpleNamespace())

    assert calls["plan"] is plan
    torch.testing.assert_close(calls["bind"]["cache_lengths"], metadata.seq_lens)
    assert calls["bind"]["cache_lengths"].shape == (1,)
    assert (
        calls["bind"]["selected_indices"].data_ptr()
        == impl.topk_indices_buffer.data_ptr()
    )
    assert (
        calls["bind"]["selected_lengths"].data_ptr()
        == metadata.cache_seq_lens_per_token.data_ptr()
    )


def test_b12x_sparse_mla_metadata_defers_per_token_lengths() -> None:
    metadata = B12xMLASparseMetadata(
        num_reqs=1,
        max_query_len=1,
        max_seq_len=1,
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=1,
    )

    metadata.cache_seq_lens_per_token = torch.tensor([1], dtype=torch.int32)

    torch.testing.assert_close(
        metadata.cache_seq_lens_per_token,
        torch.tensor([1], dtype=torch.int32),
    )


def _causal_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    context_len: int,
):
    return q_idx + context_len >= kv_idx


def _causal_sliding_window_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    context_len: int,
    sliding_window: int,
):
    causal_mask = q_idx + context_len >= kv_idx
    window_mask = q_idx + context_len - kv_idx < sliding_window
    return causal_mask & window_mask


@pytest.mark.parametrize(
    ("dtype", "kv_cache_dtype", "paged_attention", "expected_reason"),
    [
        pytest.param(
            torch.float16,
            "fp8_e4m3",
            None,
            "b12x currently requires bfloat16 queries",
            id="query-dtype",
        ),
        pytest.param(
            torch.bfloat16,
            "float16",
            None,
            "b12x does not support float16 KV cache",
            id="kv-cache-dtype",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            None,
            "Install the b12x backend with `pip install vllm[b12x]`",
            id="package-not-installed",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            SimpleNamespace(is_supported=lambda: False),
            "b12x paged attention is not supported on the current device",
            id="device-api",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            SimpleNamespace(is_supported=lambda: True),
            None,
            id="supported",
        ),
    ],
)
def test_b12x_attention_config_support(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    paged_attention,
    expected_reason: str | None,
) -> None:
    monkeypatch.setattr(
        b12x,
        "get_b12x_paged_attention",
        lambda: paged_attention,
    )

    reason = B12xPagedAttentionBackend.supports_combination(
        head_size=128,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=128,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=DeviceCapability(12, 0),
    )

    assert reason == expected_reason


def test_b12x_attention_uses_two_plane_nhd_cache() -> None:
    spec = B12xPagedAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        )
    )

    assert spec.num_head_slots == 2
    assert spec.state_content_bytes == 4 * 128 * 2
    assert B12xPagedAttentionBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.LBHNC,
        KVCacheLayout.BLHNC,
    )
    assert B12xPagedAttentionBackend.supports_block_size(128)
    assert not B12xPagedAttentionBackend.supports_block_size(32)


def test_b12x_attention_hybrid_cache_capacity_includes_expansion() -> None:
    assert _max_page_table_width(4096, 128, 4096, False) == 32
    assert _max_page_table_width(4096, 128, 4096, True) == 64


def test_b12x_attention_runtime_page_size_comes_from_cache() -> None:
    key_cache = torch.empty((3, 64, 4, 128), device="meta")
    value_cache = torch.empty_like(key_cache)

    assert _kv_page_size(key_cache, value_cache) == 64
    with pytest.raises(ValueError, match="matching K/V page sizes"):
        _kv_page_size(key_cache, torch.empty((3, 128, 4, 128), device="meta"))


def test_b12x_attention_lazily_prepares_decode_bucket(monkeypatch) -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    plan = SimpleNamespace(layout=SimpleNamespace(nbytes=96))
    created: list[tuple[int, int]] = []

    def create_plan(page_size: int, batch_size: int) -> SimpleNamespace:
        created.append((page_size, batch_size))
        return plan

    impl._decode_plans = {}
    impl._create_decode_plan = create_plan
    impl._scratch_nbytes = 128
    impl._extend_plans = {}
    impl._verify_q_per_req = 0
    metadata = SimpleNamespace(max_query_len=1)
    monkeypatch.setattr(b12x, "_capture_alloc_forbidden", lambda: False)

    assert impl._select_plan(metadata, 7, 7, 7, 64) is plan
    assert impl._select_plan(metadata, 7, 7, 7, 64) is plan
    assert created == [(64, 7)]


def test_b12x_attention_fp8_descales_follow_request_batch() -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    impl.kv_cache_dtype = "fp8_e4m3"
    layer = SimpleNamespace(
        _k_scale=torch.tensor(2.0),
        _v_scale=torch.tensor([3.0, 4.0, 5.0]),
    )

    k_descale, v_descale = impl._prepare_fp8_descales(
        layer, num_reqs=2, device=torch.device("cpu")
    )

    torch.testing.assert_close(k_descale, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(v_descale, torch.tensor([3.0, 4.0]))
    assert k_descale.stride() == (0,)
    assert v_descale.stride() == (1,)


def test_b12x_attention_sinks_refresh_in_place_after_reload() -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    source = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    impl._sinks_source = source
    impl.sinks = None

    impl.process_weights_after_loading(torch.bfloat16)
    assert impl.sinks is not None
    sinks_ptr = impl.sinks.data_ptr()
    source.copy_(torch.tensor([3.0, 4.0], dtype=torch.bfloat16))
    impl.process_weights_after_loading(torch.bfloat16)

    assert impl.sinks.data_ptr() == sinks_ptr
    torch.testing.assert_close(impl.sinks, source.float())


@pytest.mark.parametrize(
    "batch_spec_name",
    ["small_decode", "small_prefill", "mixed_small", "medium_decode"],
)
@pytest.mark.parametrize(
    ("kv_cache_dtype", "model_dtype"),
    [
        ("auto", None),
        ("bfloat16", torch.bfloat16),
        ("fp8_e4m3", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("block_size", [64, 128])
def test_b12x_causal_backend_correctness(
    default_vllm_config,
    workspace_init,
    batch_spec_name: str,
    kv_cache_dtype: str,
    model_dtype: torch.dtype | None,
    block_size: int,
) -> None:
    """b12x causal paged attention matches the shared SDPA reference."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS[batch_spec_name]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        model_dtype=model_dtype,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


@pytest.mark.parametrize(
    "batch_spec",
    [
        pytest.param(BatchSpec(seq_lens=[2080, 2200], query_lens=[1, 1]), id="decode"),
        pytest.param(BatchSpec(seq_lens=[2080, 2200], query_lens=[8, 8]), id="prefill"),
    ],
)
def test_b12x_causal_sliding_window(
    default_vllm_config,
    workspace_init,
    batch_spec: BatchSpec,
) -> None:
    """b12x causal sliding-window attention matches the shared reference."""
    _require_b12x_paged_attention()

    model = "microsoft/Phi-tiny-MoE-instruct"
    sliding_window = ModelConfig(
        model=model, max_model_len=max(batch_spec.seq_lens)
    ).get_sliding_window()
    assert sliding_window is not None
    mask = partial(_causal_sliding_window_mask, sliding_window=sliding_window)

    _test_backend_correctness(
        batch_spec,
        model,
        [AttentionBackendEnum.B12X],
        mask,
        block_size=64,
        atol=3e-2,
        rtol=3e-2,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_attention_sinks(
    default_vllm_config,
    workspace_init,
) -> None:
    """b12x attention sinks match the explicit sink reference."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS["small_prefill"]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=64,
        atol=3e-2,
        rtol=3e-2,
        use_sinks=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_decode_cuda_graph_replay(
    default_vllm_config,
    workspace_init,
) -> None:
    """b12x decode output remains correct after CUDA graph replay."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS["small_decode"]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=64,
        use_cuda_graph=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_speculative_verification_uses_cuda_graph_plan(
    default_vllm_config,
    workspace_init,
) -> None:
    """Exercise the verifier plan rather than the general extend plan."""
    _require_b12x_paged_attention()
    batch_spec = BatchSpec(seq_lens=[32, 40], query_lens=[4, 4])

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=128,
        num_speculative_tokens=3,
        use_cuda_graph=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )
