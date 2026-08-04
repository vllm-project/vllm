# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

_OFFLOAD_LAYER = "model.layers.0.self_attn"
_OFFLOAD_BLOCK_SIZE = 64
_OFFLOAD_BLOCKS = 4
_OFFLOAD_REQUESTS = 2
_OFFLOAD_TOPK = 128
_OFFLOAD_RESIDENT = 128
_OFFLOAD_QK_DIM = 576
_OFFLOAD_V_DIM = 512


def _make_sparse_mla_offload_case(num_heads: int, *, writer: bool):
    from types import MappingProxyType, SimpleNamespace

    from vllm.forward_context import ForwardContext
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    from vllm.v1.worker.gpu.sparse_mla_offload import SparseMLALayerView

    device = torch.device("cuda")
    torch.manual_seed(13026)
    host = torch.randn(
        _OFFLOAD_BLOCKS,
        _OFFLOAD_BLOCK_SIZE,
        _OFFLOAD_QK_DIM,
        dtype=torch.bfloat16,
        pin_memory=True,
    )
    host_uva = get_accelerator_view_from_cpu_tensor(host)
    request_block_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    request_num_tokens = torch.full(
        (_OFFLOAD_REQUESTS,), 128, dtype=torch.int32, device=device
    )
    generation = torch.full((_OFFLOAD_REQUESTS,), 7, dtype=torch.int64, device=device)
    resident_kv = torch.zeros(
        _OFFLOAD_REQUESTS,
        _OFFLOAD_RESIDENT,
        _OFFLOAD_QK_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    resident_ids = torch.full(
        (_OFFLOAD_REQUESTS, _OFFLOAD_RESIDENT),
        -1,
        dtype=torch.int64,
        device=device,
    )
    resident_generation = torch.full_like(resident_ids, -1)
    resident_access = torch.zeros_like(resident_ids)
    for request in range(_OFFLOAD_REQUESTS):
        physical = request_block_ids[request, 0].item()
        resident_kv[request, :32].copy_(host[physical, :32])
    resident_ids[:, :32] = torch.arange(32, dtype=torch.int64, device=device)
    resident_generation[:, :32] = 7
    newest_kv = torch.empty(
        _OFFLOAD_REQUESTS,
        1,
        _OFFLOAD_QK_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    for request in range(_OFFLOAD_REQUESTS):
        physical = request_block_ids[request, 1].item()
        newest_kv[request, 0].copy_(host[physical, 62])
    newest_ids = torch.full(
        (_OFFLOAD_REQUESTS, 1), 126, dtype=torch.int64, device=device
    )
    newest_generation = torch.full_like(newest_ids, 7)
    current = torch.randn(
        _OFFLOAD_REQUESTS,
        _OFFLOAD_QK_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        _OFFLOAD_REQUESTS,
        num_heads,
        _OFFLOAD_QK_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    topk = torch.arange(_OFFLOAD_TOPK, dtype=torch.int32, device=device)
    topk = topk.view(1, 1, -1).expand(_OFFLOAD_REQUESTS, 1, -1).contiguous()
    output = torch.empty(
        _OFFLOAD_REQUESTS,
        num_heads,
        _OFFLOAD_V_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    buffers = {
        "resident_main_kv": resident_kv,
        "resident_logical_ids": resident_ids,
        "resident_last_access": resident_access,
        "resident_generation": resident_generation,
        "newest_main_kv": newest_kv,
        "newest_logical_ids": newest_ids,
        "newest_generation": newest_generation,
        "request_block_ids": request_block_ids,
        "request_num_blocks": torch.full(
            (_OFFLOAD_REQUESTS,), 2, dtype=torch.int32, device=device
        ),
        "request_num_tokens": request_num_tokens,
        "request_generation": generation,
        "request_active": torch.ones(
            _OFFLOAD_REQUESTS, dtype=torch.bool, device=device
        ),
        "topk_logical_ids": torch.full_like(topk, -1),
        "topk_physical_ids": torch.zeros_like(topk),
        "topk_hit_mask": torch.zeros_like(topk, dtype=torch.bool),
        "miss_logical_ids": torch.full_like(topk, -1),
        "miss_victim_slots": torch.zeros_like(topk),
        "miss_counts": torch.zeros(
            _OFFLOAD_REQUESTS, 1, dtype=torch.int32, device=device
        ),
        "accepted_counts": torch.zeros(
            _OFFLOAD_REQUESTS, dtype=torch.int32, device=device
        ),
        "hit_output": torch.empty_like(output).unsqueeze(1),
        "hit_lse": torch.empty(
            _OFFLOAD_REQUESTS,
            1,
            num_heads,
            dtype=torch.float32,
            device=device,
        ),
        "miss_output": torch.empty_like(output).unsqueeze(1),
        "miss_lse": torch.empty(
            _OFFLOAD_REQUESTS,
            1,
            num_heads,
            dtype=torch.float32,
            device=device,
        ),
    }
    side_stream = torch.cuda.Stream(device=device)
    fork_event = torch.cuda.Event(enable_timing=False)
    ready_event = torch.cuda.Event(enable_timing=False)
    view = SparseMLALayerView(
        layer_name=_OFFLOAD_LAYER,
        layer_index=0,
        is_host_writer=writer,
        main_host_kv=host,
        main_host_kv_uva=host_uva,
        local_buffers=MappingProxyType(buffers),
        side_stream=side_stream,
        fork_ready_events=(fork_event,),
        miss_ready_events=(ready_event,),
    )
    req_ids = torch.arange(_OFFLOAD_REQUESTS, dtype=torch.int32, device=device)
    context = ForwardContext(
        no_compile_layers={_OFFLOAD_LAYER: view},
        attn_metadata={_OFFLOAD_LAYER: SimpleNamespace(req_id_per_token=req_ids)},
        slot_mapping={},
    )
    return SimpleNamespace(
        host=host,
        host_uva=host_uva,
        buffers=buffers,
        current=current,
        query=query,
        topk=topk,
        output=output,
        req_ids=req_ids,
        context=context,
        view=view,
    )


def _run_sparse_mla_offload(case) -> None:
    from vllm.forward_context import override_forward_context

    with override_forward_context(case.context):
        dependency = torch.ops.vllm.sparse_mla_cache_plan(
            case.current, case.topk, _OFFLOAD_LAYER
        )
        torch.ops.vllm.sparse_mla_offload_attention(
            case.query,
            case.current,
            case.output,
            _OFFLOAD_LAYER,
            dependency,
        )


def _full_sparse_mla_reference(case, active: int) -> torch.Tensor:
    import vllm.v1.attention.ops.flashmla as fm

    resident_ids = case.buffers["resident_logical_ids"].cpu()
    mapped = torch.empty_like(case.topk[:active])
    for request in range(active):
        slots = {
            logical: slot for slot, logical in enumerate(resident_ids[request].tolist())
        }
        mapped[request, 0].copy_(
            torch.tensor(
                [
                    request * _OFFLOAD_RESIDENT + slots[logical]
                    for logical in range(_OFFLOAD_TOPK)
                ],
                dtype=torch.int32,
                device=mapped.device,
            )
        )
    query = case.query[:active]
    if query.shape[1] < 64:
        query = torch.nn.functional.pad(query, (0, 0, 0, 64 - query.shape[1]))
    lengths = torch.full(
        (active,),
        _OFFLOAD_TOPK,
        dtype=torch.int32,
        device=query.device,
    )
    output, _, _ = fm.flash_mla_sparse_fwd(
        query,
        case.buffers["resident_main_kv"].view(-1, 1, _OFFLOAD_QK_DIM),
        mapped,
        _OFFLOAD_QK_DIM**-0.5,
        _OFFLOAD_V_DIM,
        topk_length=lengths,
    )
    return output[:, : case.query.shape[1]]


def _snapshot_sparse_mla_case(case):
    state = {
        name: tensor.clone()
        for name, tensor in case.buffers.items()
        if name
        in {
            "resident_main_kv",
            "resident_logical_ids",
            "resident_last_access",
            "resident_generation",
            "newest_main_kv",
            "newest_logical_ids",
            "newest_generation",
        }
    }
    return state, case.host.clone()


def _restore_sparse_mla_case(case, snapshot, *, active: int) -> None:
    state, host = snapshot
    for name, tensor in state.items():
        case.buffers[name].copy_(tensor)
    case.host.copy_(host)
    case.req_ids.fill_(-1)
    case.req_ids[:active].copy_(
        torch.arange(active, dtype=torch.int32, device=case.req_ids.device)
    )
    case.output.zero_()


def test_deepseek_v4_c128a_dynamic_topk_packed_buffers():
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    device = torch.device("cuda")
    capacity_width = 256
    active_width = 128
    global_decode_buffer = torch.empty(
        (2, capacity_width), dtype=torch.int32, device=device
    )
    decode_lens_buffer = torch.empty(2, dtype=torch.int32, device=device)
    prefill_buffer = torch.empty((2, capacity_width), dtype=torch.int32, device=device)

    global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
        positions=torch.tensor([255, 511], dtype=torch.int64, device=device),
        compress_ratio=128,
        num_decode_tokens=1,
        token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32, device=device),
        block_table=torch.tensor([[3]], dtype=torch.int32, device=device),
        block_size=capacity_width,
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64, device=device),
        global_decode_buffer=global_decode_buffer,
        decode_lens_buffer=decode_lens_buffer,
        prefill_buffer=prefill_buffer,
        max_compressed_tokens=active_width,
    )

    assert global_decode.shape == (1, active_width)
    assert prefill_local.shape == (1, active_width)
    assert global_decode.stride() == (active_width, 1)
    assert prefill_local.stride() == (active_width, 1)
    assert global_decode[0, :2].cpu().tolist() == [768, 769]
    assert decode_lens.cpu().tolist() == [2]
    assert prefill_local[0, :4].cpu().tolist() == list(range(4))
    assert torch.all(global_decode[0, 2:] == -1)
    assert torch.all(prefill_local[0, 4:] == -1)


def test_sparse_flashmla_metadata_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 128
    num_heads_k = 1
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    topk = 128

    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )
    assert isinstance(tile_md, fm.FlashMLASchedMeta)
    assert tile_md.tile_scheduler_metadata is None
    assert tile_md.num_splits is None
    assert num_splits is None


def test_sparse_flashmla_decode_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 64
    head_dim_k = 576
    head_dim_v = 512
    num_heads_k = 1
    page_block_size = 64
    bytes_per_token = 656
    topk = 128

    # Metadata
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    # q_heads_per_hk = num_heads_q // num_heads_k
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )

    # Inputs
    q = torch.zeros(
        (batch_size, seqlen_q, num_heads_q, head_dim_k),
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = torch.zeros(
        (1, page_block_size, num_heads_k, bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    indices = torch.zeros(
        (batch_size, seqlen_q, topk), dtype=torch.int32, device=device
    )

    block_table = torch.zeros((batch_size, 128), dtype=torch.int32, device=device)
    out, lse = fm.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_md,
        num_splits,
        indices=indices,
        is_fp8_kvcache=True,
    )
    assert out.shape[0] == batch_size
    assert out.shape[-1] == head_dim_v
    assert lse.shape[0] == batch_size


@pytest.mark.parametrize("h_q", [64, 128])
def test_sparse_flashmla_prefill_smoke(h_q: int):
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    torch.manual_seed(0)
    s_q = 1
    s_kv = 8
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128
    q = torch.randn((s_q, h_q, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=torch.bfloat16, device=device)
    indices = torch.randint(s_kv, (s_q, h_kv, topk), dtype=torch.int32, device=device)
    reference_indices = indices.clone()
    reference_indices[..., 1:] = -1
    kwargs = {"topk_length": torch.ones(1, dtype=torch.int32, device=device)}
    reference = fm.flash_mla_sparse_fwd(q, kv, reference_indices, 1.0, d_v, **kwargs)
    actual = fm.flash_mla_sparse_fwd(q, kv, indices, 1.0, d_v, **kwargs)

    for actual_tensor, reference_tensor in zip(actual, reference):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)
    assert actual[0].shape == (s_q, h_q, d_v)


def test_deepseek_v4_prefill_chunk_planning_expands_for_short_sequences():
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        block_size=64,
        num_prefills=5,
        prefill_seq_lens_cpu=torch.tensor([80, 96, 112, 128, 144], dtype=torch.int32),
        prefill_query_lens_cpu=torch.tensor([4, 4, 4, 4, 4], dtype=torch.int32),
        prefill_window_size=64,
        prefill_max_model_len=1024,
        prefill_max_num_batched_tokens=128,
    )

    chunk_plan = metadata.get_prefill_chunk_plan(compress_ratio=4, prefill_chunk_size=4)

    # the adaptive plan keeps all 5 in one chunk
    assert chunk_plan == [(0, 5, 36, 103)]


def test_flashinfer_sparse_indices_cache(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as flashinfer_mod
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    builder_calls = 0

    def fake_build(*args, **kwargs):
        nonlocal builder_calls
        builder_calls += 1
        return (
            torch.tensor([[builder_calls]], dtype=torch.int32),
            torch.tensor([builder_calls], dtype=torch.int32),
        )

    monkeypatch.setattr(
        flashinfer_mod, "build_flashinfer_mixed_sparse_indices", fake_build
    )

    def make_attn(compress_ratio: int, topk_width: int):
        attn = object.__new__(flashinfer_mod.DeepseekV4FlashInferMLAAttention)
        attn.compress_ratio = compress_ratio
        attn.window_size = 4
        attn.topk_indices_buffer = torch.tensor(
            [[0, 1], [2, 3], [4, 5]], dtype=torch.int32
        )[:, :topk_width]
        return attn

    def make_swa_metadata():
        return DeepseekSparseSWAMetadata(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
            block_size=64,
            seq_lens=torch.tensor([8, 10], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1, 1], dtype=torch.int32),
            decode_swa_indices=torch.tensor([[5, 6, -1, -1]], dtype=torch.int32),
            decode_swa_lens=torch.tensor([2], dtype=torch.int32),
            is_valid_token=torch.tensor([True], dtype=torch.bool),
            num_decodes=1,
            num_prefills=1,
            num_decode_tokens=1,
            num_prefill_tokens=2,
        )

    def make_flashmla_metadata():
        return DeepseekV4FlashMLAMetadata(
            num_reqs=2,
            max_query_len=2,
            max_seq_len=10,
            num_actual_tokens=3,
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
            block_size=256,
            topk_tokens=2,
            c128a_global_decode_topk_indices=torch.tensor(
                [[[9, 10]]], dtype=torch.int32
            ),
            c128a_decode_topk_lens=torch.tensor([2], dtype=torch.int32),
            c128a_prefill_topk_indices=torch.tensor(
                [[0, 1], [1, 2]], dtype=torch.int32
            ),
        )

    swa_attn = make_attn(1, 0)
    swa_metadata = make_swa_metadata()
    _, _, sparse_indices_first, sparse_lens_first = (
        swa_attn._build_sparse_index_metadata(
            kv_cache=None,
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
        )
    )
    _, _, sparse_indices_second, sparse_lens_second = (
        swa_attn._build_sparse_index_metadata(
            kv_cache=None,
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
        )
    )
    assert builder_calls == 1
    assert sparse_indices_first is sparse_indices_second
    assert sparse_lens_first is sparse_lens_second

    c128a_attn = make_attn(128, 2)
    c128a_metadata = make_swa_metadata()
    c128a_flashmla_md = make_flashmla_metadata()
    _, _, sparse_indices_first, sparse_lens_first = (
        c128a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c128a_metadata,
            attn_metadata=c128a_flashmla_md,
            swa_only=False,
        )
    )
    _, _, sparse_indices_second, sparse_lens_second = (
        c128a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c128a_metadata,
            attn_metadata=c128a_flashmla_md,
            swa_only=False,
        )
    )

    assert builder_calls == 2
    assert sparse_indices_first is sparse_indices_second
    assert sparse_lens_first is sparse_lens_second

    c4a_attn = make_attn(4, 2)
    c4a_metadata = make_swa_metadata()
    c4a_flashmla_md = make_flashmla_metadata()
    c4a_flashmla_md.c128a_global_decode_topk_indices = None
    c4a_flashmla_md.c128a_decode_topk_lens = None
    c4a_flashmla_md.c128a_prefill_topk_indices = None
    _, _, sparse_indices_third, sparse_lens_third = (
        c4a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c4a_metadata,
            attn_metadata=c4a_flashmla_md,
            swa_only=False,
        )
    )
    _, _, sparse_indices_fourth, sparse_lens_fourth = (
        c4a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c4a_metadata,
            attn_metadata=c4a_flashmla_md,
            swa_only=False,
        )
    )

    assert builder_calls == 4
    assert sparse_indices_third is not sparse_indices_fourth
    assert sparse_lens_third is not sparse_lens_fourth


def test_sparse_mla_offload_operator_path_schema_and_fake():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    import vllm.v1.attention.ops.flashmla  # noqa: F401
    from vllm.config.compilation import CompilationConfig

    plan_schema = torch._C._dispatch_find_schema_or_throw(
        "_C::sparse_mla_cache_plan", ""
    ).schema()
    transfer_schema = torch._C._dispatch_find_schema_or_throw(
        "_C::sparse_mla_offload_transfer", ""
    ).schema()
    assert "int num_host_blocks" in str(plan_schema)
    assert "num_host_blocks" not in str(transfer_schema)

    with FakeTensorMode():
        current = torch.empty(2, _OFFLOAD_QK_DIM, dtype=torch.bfloat16)
        topk = torch.empty(2, 1, _OFFLOAD_TOPK, dtype=torch.int32)
        query = torch.empty(2, 64, _OFFLOAD_QK_DIM, dtype=torch.bfloat16)
        output = torch.empty(2, 64, _OFFLOAD_V_DIM, dtype=torch.bfloat16)
        dependency = torch.ops.vllm.sparse_mla_cache_plan(current, topk, _OFFLOAD_LAYER)
        torch.ops.vllm.sparse_mla_offload_attention(
            query, current, output, _OFFLOAD_LAYER, dependency
        )
        assert dependency.shape == (0,)
        assert dependency.dtype == current.dtype
        assert dependency.device == current.device

    def offload_path(current, topk, query, output):
        dependency = torch.ops.vllm.sparse_mla_cache_plan(current, topk, _OFFLOAD_LAYER)
        torch.ops.vllm.sparse_mla_offload_attention(
            query, current, output, _OFFLOAD_LAYER, dependency
        )
        return output

    graph = make_fx(offload_path, tracing_mode="fake")(
        torch.empty(2, _OFFLOAD_QK_DIM, dtype=torch.bfloat16),
        torch.empty(2, 1, _OFFLOAD_TOPK, dtype=torch.int32),
        torch.empty(2, 64, _OFFLOAD_QK_DIM, dtype=torch.bfloat16),
        torch.empty(2, 64, _OFFLOAD_V_DIM, dtype=torch.bfloat16),
    )
    offload_nodes = [
        str(node.target).removesuffix(".default")
        for node in graph.graph.nodes
        if "sparse_mla_" in str(node.target)
    ]
    assert offload_nodes == [
        "vllm.sparse_mla_cache_plan",
        "vllm.sparse_mla_offload_attention",
    ]
    assert "vllm::sparse_mla_offload_attention" in CompilationConfig._attention_ops
    assert "vllm::sparse_mla_cache_plan" not in CompilationConfig._attention_ops


def test_sparse_mla_offload_operator_path_eager_writer_and_follower():
    import vllm.v1.attention.ops.flashmla as fm
    from vllm.forward_context import override_forward_context

    supported, reason = fm.is_flashmla_sparse_supported()
    if not supported:
        pytest.skip(reason)

    follower = _make_sparse_mla_offload_case(64, writer=False)
    follower_host = follower.host.clone()
    _run_sparse_mla_offload(follower)
    torch.accelerator.synchronize()
    follower_reference = _full_sparse_mla_reference(follower, 2)
    assert torch.equal(follower.host, follower_host)
    assert torch.equal(
        follower.buffers["accepted_counts"],
        torch.full_like(follower.buffers["accepted_counts"], 34),
    )
    assert torch.equal(
        follower.buffers["miss_counts"],
        torch.full_like(follower.buffers["miss_counts"], 94),
    )
    expected_mask = torch.zeros(_OFFLOAD_TOPK, dtype=torch.bool, device="cuda")
    expected_mask[:32] = True
    expected_mask[126:] = True
    assert torch.equal(follower.buffers["topk_hit_mask"][0, 0], expected_mask)
    assert torch.equal(
        follower.buffers["miss_victim_slots"][0, 0, :94],
        torch.arange(34, 128, dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        follower.output, follower_reference, rtol=2e-2, atol=2e-2
    )

    writer = _make_sparse_mla_offload_case(64, writer=True)
    _run_sparse_mla_offload(writer)
    torch.accelerator.synchronize()
    writer_reference = _full_sparse_mla_reference(writer, 2)
    for request, physical_block in enumerate((1, 3)):
        assert torch.equal(writer.host[physical_block, 63], writer.current[request])
    for name in (
        "resident_main_kv",
        "resident_logical_ids",
        "resident_last_access",
        "resident_generation",
        "newest_main_kv",
        "newest_logical_ids",
        "newest_generation",
    ):
        assert torch.equal(writer.buffers[name], follower.buffers[name])
    torch.testing.assert_close(writer.output, writer_reference, rtol=2e-2, atol=2e-2)

    malformed = _make_sparse_mla_offload_case(64, writer=True)
    malformed.buffers["request_block_ids"][0, 0] = _OFFLOAD_BLOCKS
    state_before, host_before = _snapshot_sparse_mla_case(malformed)
    with override_forward_context(malformed.context):
        torch.ops.vllm.sparse_mla_cache_plan(
            malformed.current, malformed.topk, _OFFLOAD_LAYER
        )
    torch.accelerator.synchronize()
    assert malformed.buffers["accepted_counts"][0].item() == 0
    assert malformed.buffers["miss_counts"][0, 0].item() == 0
    for name, before in state_before.items():
        assert torch.equal(malformed.buffers[name][0], before[0])
    assert torch.equal(malformed.host, host_before)


def test_sparse_mla_offload_operator_path_full_graph_overlap():
    import vllm.v1.attention.ops.flashmla as fm
    from vllm.forward_context import override_forward_context

    supported, reason = fm.is_flashmla_sparse_supported()
    if not supported:
        pytest.skip(reason)

    for num_heads in (64, 32):
        case = _make_sparse_mla_offload_case(num_heads, writer=True)
        snapshot = _snapshot_sparse_mla_case(case)
        for _ in range(3):
            _restore_sparse_mla_case(case, snapshot, active=2)
            _run_sparse_mla_offload(case)
        torch.accelerator.synchronize()

        _restore_sparse_mla_case(case, snapshot, active=2)
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with override_forward_context(case.context), torch.cuda.stream(capture_stream):
            graph.capture_begin()
            _run_sparse_mla_offload(case)
            graph.capture_end()
        torch.accelerator.synchronize()

        pointers = [
            case.host_uva.data_ptr(),
            case.output.data_ptr(),
            *(tensor.data_ptr() for tensor in case.buffers.values()),
        ]
        runtime_ids = [
            id(case.view.side_stream),
            id(case.view.fork_ready_events[0]),
            id(case.view.miss_ready_events[0]),
        ]
        for replay, active in enumerate((2, 1) * 5):
            _restore_sparse_mla_case(case, snapshot, active=active)
            allocated = torch.accelerator.memory_allocated()
            torch.cuda.nvtx.range_push(f"sparse_mla_offload_replay_{replay}")
            graph.replay()
            torch.cuda.nvtx.range_pop()
            torch.accelerator.synchronize()
            assert torch.accelerator.memory_allocated() == allocated
            reference = _full_sparse_mla_reference(case, active)
            torch.testing.assert_close(
                case.output[:active], reference, rtol=2e-2, atol=2e-2
            )
            if active < _OFFLOAD_REQUESTS:
                assert torch.count_nonzero(case.output[active:]).item() == 0
            assert pointers == [
                case.host_uva.data_ptr(),
                case.output.data_ptr(),
                *(tensor.data_ptr() for tensor in case.buffers.values()),
            ]
            assert runtime_ids == [
                id(case.view.side_stream),
                id(case.view.fork_ready_events[0]),
                id(case.view.miss_ready_events[0]),
            ]
