# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HiSparse Stage 1 top-k-filtered sparse prefill staging."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import TypeAdapter

from vllm.config.attention import AttentionConfig, HiSparseConfig
from vllm.config.utils import hash_config
from vllm.model_executor.layers.attention import (
    sparse_mla_attention as sparse_mla_module,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla import index_group as index_group_module
from vllm.v1.attention.backends.mla.index_group import (
    HiSparseMLAIndexGroup,
    SparseMLAIndexGroupBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import SparseMLAPrefillRoute
from vllm.v1.hisparse import prefill as prefill_module
from vllm.v1.hisparse import runtime as runtime_module
from vllm.v1.hisparse.backend import HiSparseBackend
from vllm.v1.hisparse.prefill import (
    filter_prefill_gather_inputs,
    gather_sparse_prefill_cache,
    hisparse_build_selected_mask,
    hisparse_gather_resident_rows,
)
from vllm.v1.hisparse.runtime import (
    HiSparsePrefillStagingPlan,
    HiSparseRuntime,
    ResolvedHiSparseConfig,
    build_hisparse_prefill_staging_plan,
)

REQUIRES_CUDA = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="requires CUDA"
)
DEVICES = ["cpu", pytest.param("cuda", marks=REQUIRES_CUDA)]
GATHER_RESIDENT_CASES = [
    (torch.uint8, torch.uint8, 656, 0),
    (torch.uint8, torch.uint8, 528, 1),
    (torch.uint8, torch.uint8, 525, 0),
    (torch.float32, torch.uint8, 4, 1),
]


def _i32(data: Any, dev: Any = None) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.int32, device=dev)


def _nonzero_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.nonzero(as_tuple=False).squeeze(-1)


def _reference_mask(
    req_ids: torch.Tensor, bt: torch.Tensor, topk: torch.Tensor, bs: int, n_staged: int
) -> torch.Tensor:
    mask = torch.zeros((1, n_staged), dtype=torch.int32)
    bt_cpu, topk_cpu = bt.cpu().tolist(), topk.cpu().tolist()
    for t_idx, req in enumerate(req_ids.cpu().tolist()):
        if 0 <= req < bt.shape[0]:
            for tok in topk_cpu[t_idx]:
                blk = tok // bs
                if tok >= 0 and blk < bt.shape[1]:
                    row = bt_cpu[req][blk] * bs + (tok % bs)
                    if 0 <= row < n_staged:
                        mask[0, row] = 1
    return mask


def _assert_no_cuda_sync(fn: Any, *args: Any, **kwargs: Any) -> None:
    torch.accelerator.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        fn(*args, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode(prev)


def _spy_selected_mask(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    built_masks: list[torch.Tensor] = []
    orig_build = prefill_module.hisparse_build_selected_mask

    def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        built_masks.append(args[2].clone())
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(prefill_module, "hisparse_build_selected_mask", spy)
    return built_masks


def _make_fp8_plan(
    dev: Any, gpu_rows: tuple[int, int] = (11, 12)
) -> tuple[HiSparsePrefillStagingPlan, int]:
    bt, sl = _i32([[1, 2], [3, 4]], dev), _i32([8, 8], dev)
    plan = build_hisparse_prefill_staging_plan(bt, sl, 4, 4)
    base = int(plan.block_table[0, 0].item()) * 4
    plan.gpu_row_ids = torch.full(
        (1, plan.row_ids.shape[1]), -1, dtype=torch.int32, device=dev
    )
    plan.gpu_row_ids[0, base : base + 2] = _i32(list(gpu_rows), dev)
    return plan, base


def _setup_fp8_env(
    monkeypatch: pytest.MonkeyPatch,
    dev: Any,
    resident_blocks: int = 4,
    recorded_objs: list[torch.Tensor] | None = None,
) -> tuple[Any, ...]:
    plan, base = _make_fp8_plan(dev)
    res_cache = torch.empty((resident_blocks, 4, 656), dtype=torch.uint8, device=dev)
    src_kv = torch.empty((8, 4, 656), dtype=torch.uint8, device=dev)
    dst_kv = torch.empty((plan.row_ids.shape[1], 576), dtype=torch.bfloat16, device=dev)
    buffers = (src_kv, dst_kv, _i32([0], dev))
    recorded_ids: list[torch.Tensor] = []
    built_masks = _spy_selected_mask(monkeypatch)

    def stub_upconvert(*args: Any, device_row_ids: torch.Tensor, **kwargs: Any) -> None:
        if recorded_objs is not None:
            recorded_objs.append(device_row_ids)
        recorded_ids.append(device_row_ids.clone())

    monkeypatch.setattr(
        index_group_module.ops, "cp_gather_and_upconvert_fp8_kv_cache", stub_upconvert
    )
    if dev.type != "cuda":

        def _make_mock(*args: Any, **kwargs: Any) -> MagicMock:
            return MagicMock()

        for attr in ("Stream", "Event"):
            monkeypatch.setattr(index_group_module.torch, attr, _make_mock)
        monkeypatch.setattr(torch.accelerator, "current_stream", object)
    stream = torch.Stream(device=dev) if dev.type == "cuda" else MagicMock()
    return plan, base, res_cache, buffers, recorded_ids, built_masks, stream


def _make_mock_cache(
    res_cache: Any = None,
    gather_masks: list[torch.Tensor] | None = None,
    fallback_calls: list[tuple[int, Any, int]] | None = None,
    layer_idx: int = 0,
) -> Any:
    def _stage_fallback(kv: Any, plan: Any, n_decode: int) -> torch.Tensor:
        assert fallback_calls is not None
        fallback_calls.append((layer_idx, plan, n_decode))
        return torch.empty((1, 4, 8), device=kv.device)

    def _gather_plan(*args: Any) -> int:
        if gather_masks is not None:
            gather_masks.append(args[4].clone())
        return 0

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    stage_fn = (
        _stage_fallback
        if fallback_calls is not None
        else MagicMock(side_effect=AssertionError("fallback"))
    )
    cache = SimpleNamespace(
        view=SimpleNamespace(cache=res_cache) if res_cache is not None else None,
        block_table=None,
        index_group_caches=None,
        stage_prefill_rows=stage_fn,
        prepare_group_for_batch=_noop,
        ensure_prefill_gpu_sources=_noop,
        runtime=SimpleNamespace(
            cache_ops=SimpleNamespace(hisparse_gather_plan=_gather_plan)
        ),
    )
    return cast(Any, cache)


def _make_mock_group(
    topk: Any,
    res_cache: Any = None,
    gather_masks: list[torch.Tensor] | None = None,
    fallback_calls: list[tuple[int, Any, int]] | None = None,
    prefill_stream: Any = None,
) -> HiSparseMLAIndexGroup:
    group = object.__new__(HiSparseMLAIndexGroup)
    group.caches = [
        _make_mock_cache(res_cache, gather_masks, fallback_calls, i) for i in range(2)
    ]
    group.num_layers, group.has_indexer, group.sparse_prefill_staging = 2, True, True
    group.logical_topk_indices, group._prefill_mask_cache, group._topk_owner = (
        topk,
        None,
        None,
    )
    group.prefill_stream = prefill_stream if prefill_stream is not None else MagicMock()
    group.prefill_ready_events = [MagicMock(), MagicMock()]
    return group


def _make_vllm_cfg(
    hisparse_cfg: Any = None,
    dcp_size: int = 1,
    pcp_size: int = 1,
    use_ubatching: bool = False,
    static_ctx: Any = None,
) -> Any:
    parallel_cfg = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        prefill_context_parallel_size=pcp_size,
        use_ubatching=use_ubatching,
    )
    attn_cfg = AttentionConfig(
        hisparse_config=hisparse_cfg or HiSparseConfig(sparse_prefill_staging=True)
    )
    comp_cfg = SimpleNamespace(static_forward_context=static_ctx)
    return cast(
        Any,
        SimpleNamespace(
            attention_config=attn_cfg,
            parallel_config=parallel_cfg,
            compilation_config=comp_cfg,
        ),
    )


def _make_meta(
    plan: Any,
    reqs: Any,
    n_dec: int = 2,
    n_act: int = 4,
    route: SparseMLAPrefillRoute = SparseMLAPrefillRoute.SPARSE_MQA,
) -> Any:
    meta = SimpleNamespace(
        num_decodes=1, num_decode_tokens=n_dec, num_actual_tokens=n_act, topk_start=0
    )
    meta.req_id_per_token = reqs
    meta.prefill = SimpleNamespace(host_staging_plan=plan, route=route)
    return meta


def _run_fp8_chunk(
    group: HiSparseMLAIndexGroup,
    layer_idx: int,
    req_start: int,
    plan: Any,
    meta: Any,
    dev: Any,
    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    src_kv, dst_kv, workspace = buffers
    bt_slice = plan.block_table[req_start : req_start + 1]
    group.gather_fp8_prefill(
        layer_idx, src_kv, dst_kv, bt_slice, workspace, 1, meta, req_start
    )
    if dev.type == "cuda":
        assert group.prefill_stream is not None
        group.prefill_stream.synchronize()


@pytest.mark.parametrize("dev_str", DEVICES)
@pytest.mark.parametrize(
    "bs,num_topk,non_contig,truncated",
    [(64, 256, False, True), (16, 2063, True, False), (64, 4224, False, False)],
)
def test_build_selected_mask(
    dev_str: str, bs: int, num_topk: int, non_contig: bool, truncated: bool
) -> None:
    dev = torch.device(dev_str)
    seq_lens, reqs = _i32([44, 60, 14], dev), _i32([0, 1, 1, 2, -1, 3], dev)
    req_bt = _i32([[5, 2, 7, -1], [5, 2, 9, 3], [11, -1, -1, -1]], dev)
    plan = build_hisparse_prefill_staging_plan(req_bt, seq_lens, bs, 10)
    topk = torch.full((6, num_topk), -1, dtype=torch.int32, device=dev)
    gen = torch.Generator("cpu").manual_seed(17)
    for tok_idx, req in enumerate(reqs.tolist()):
        seq_lim = int(seq_lens[req].item()) if 0 <= req < 3 else 64
        act = min(num_topk // 2 + 7, num_topk)
        topk[tok_idx, :act] = torch.randint(
            0, seq_lim, (act,), generator=gen, dtype=torch.int32
        )
        topk[tok_idx, 1], topk[tok_idx, 5], topk[tok_idx, act - 1] = -1, bs * 4 + 9, -3
    bt, n_staged = plan.block_table.clone(), plan.row_ids.shape[1]
    if truncated:
        n_staged -= bs
        bt[2, 0], bt[0, 3], bt[1, 3] = -1, -(2**31), 2**30
        topk[0, 0], topk[1, 0] = 3 * bs + 5, 3 * bs + 7
    if non_contig:
        sentinel = bt.shape[1] * bs - 1
        assert int(bt[1, sentinel // bs].item()) >= 0
        strided_buf = torch.empty((6, num_topk + 1), dtype=torch.int32, device=dev)
        strided_buf[:, :-1], strided_buf[:, -1] = topk, sentinel
        topk = strided_buf[:, :-1]
        assert not topk.is_contiguous()
    mask = hisparse_build_selected_mask(
        reqs, bt, topk, block_size=bs, num_staged_rows=n_staged
    )
    assert mask.shape == (1, n_staged) and mask.dtype == torch.int32
    assert torch.equal(mask.cpu(), _reference_mask(reqs, bt, topk, bs, n_staged))
    if dev.type == "cuda":
        _assert_no_cuda_sync(
            hisparse_build_selected_mask,
            reqs,
            bt,
            topk,
            block_size=bs,
            num_staged_rows=n_staged,
        )
    if truncated:

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.int32, device=dev)

        empty_dims = [(0, 9, 64, 1, 2), (3, 0, 64, 1, 2), (3, 9, 0, 1, 2)]
        for n_tok, n_cols, n_rows, n_reqs, n_blks in empty_dims + [
            (3, 9, 64, 0, 2),
            (3, 9, 64, 1, 0),
        ]:
            empty_mask = hisparse_build_selected_mask(
                zeros(n_tok),
                zeros(n_reqs, n_blks),
                zeros(n_tok, n_cols),
                block_size=64,
                num_staged_rows=n_rows,
            )
            assert empty_mask.shape == (1, n_rows) and (empty_mask == 0).all()


@pytest.mark.parametrize("dev_str", DEVICES)
@pytest.mark.parametrize("src_dtype,dst_dtype,row_elems,strided", GATHER_RESIDENT_CASES)
def test_gather_resident_rows(
    dev_str: str, src_dtype: Any, dst_dtype: Any, row_elems: int, strided: int
) -> None:
    dev, n_blks, bs, n_dst = torch.device(dev_str), 6, 4, 140
    raw_shape = (n_blks, bs * (2 if strided else 1), row_elems)
    gen = torch.Generator("cpu").manual_seed(31)
    raw = torch.randint(1, 100, raw_shape, generator=gen, dtype=torch.uint8).to(
        dtype=src_dtype, device=dev
    )
    src = raw[:, :bs, :] if strided else raw
    row_ids = torch.full((n_dst,), -1, dtype=torch.int32, device=dev)
    row_ids[[0, 5, 63, 128, 130, 139]] = _i32([3, 0, n_blks * bs - 1, 7, 24, 10], dev)
    dst_cols = row_elems * src_dtype.itemsize // dst_dtype.itemsize
    dst = torch.full((n_dst, dst_cols), 77, dtype=dst_dtype, device=dev)
    expected, src_bytes = (
        dst.clone().view(torch.uint8),
        src.reshape(-1, row_elems).view(torch.uint8),
    )
    for dst_row, src_row in enumerate(row_ids.tolist()):
        if 0 <= src_row < n_blks * bs:
            expected[dst_row].copy_(src_bytes[src_row])
    hisparse_gather_resident_rows(src, dst, row_ids)
    assert torch.equal(dst.view(torch.uint8), expected)
    empty_dst = torch.full((8, 16), 99, dtype=torch.uint8, device=dev)
    empty_src = torch.empty((0, 4, 16), dtype=torch.uint8, device=dev)
    hisparse_gather_resident_rows(empty_src, empty_dst, _i32(range(8), dev))
    assert (empty_dst == 99).all()


@pytest.mark.parametrize("dev_str", DEVICES)
def test_gather_sparse_prefill_cache(
    monkeypatch: pytest.MonkeyPatch, dev_str: str
) -> None:
    dev, bs, res_bs, rw = torch.device(dev_str), 4, 2, 8
    miss_mask = _i32([[1, 0, 1, 1]], dev)
    gpu_ids = _i32([[-1, 12, -1, 15]], dev)
    sel_filter = _i32([[1, 0, 0, 1]], dev)
    assert filter_prefill_gather_inputs(miss_mask, gpu_ids, None) == (
        miss_mask,
        gpu_ids,
    )
    assert filter_prefill_gather_inputs(miss_mask, None, sel_filter)[1] is None
    filt_miss, filt_gpu = filter_prefill_gather_inputs(miss_mask, gpu_ids, sel_filter)
    assert torch.equal(filt_miss, sel_filter) and filt_gpu is not None
    assert torch.equal(filt_gpu, _i32([[-1, -1, -1, 15]], dev))
    with pytest.raises(ValueError, match="selected_mask shape"):
        filter_prefill_gather_inputs(
            miss_mask, gpu_ids, torch.zeros((1, 5), dtype=torch.int32, device=dev)
        )

    bt = _i32([[5, 2, 0], [9, 3, 0]], dev)
    plan = build_hisparse_prefill_staging_plan(bt, _i32([5, 8], dev), bs, 4)
    src_gpu_ids = _i32([[11, 12, 0, 13, 0, 0], [21, 0, 22, 23, 0, 0]], dev)
    plan.ensure_gpu_sources(src_gpu_ids, _i32([0, 1], dev), 2)
    n_staged, sel_idx = plan.row_ids.shape[1], [0, 1, 4, 9, 13]
    n_blks = n_staged // bs
    assert n_blks < cdiv(runtime_module._MIN_STAGED_PREFILL_KV_ROWS, bs)
    sel_mask = torch.zeros((1, n_staged), dtype=torch.int32, device=dev)
    sel_mask[0, sel_idx] = 1

    def _select_rows(t: torch.Tensor, width: int, idx: torch.Tensor) -> torch.Tensor:
        return t.view(-1, width).index_select(0, idx)

    def stub_gather(kv: Any, staged: Any, rids: Any, drows: Any, mask: Any) -> None:
        act = _nonzero_rows((mask[0] != 0) & (rids[0] >= 0))
        staged.view(-1, rw).index_copy_(
            0, drows[0, act].long(), kv.index_select(0, rids[0, act].long())
        )

    stub_rt = object.__new__(HiSparseRuntime)
    stub_rt.backend = HiSparseBackend()
    stub_rt.cache_ops = SimpleNamespace(hisparse_gather_plan=stub_gather)
    host = torch.arange(10 * bs * rw, dtype=torch.float32, device=dev).view(10, bs, rw)
    res = torch.arange(24 * res_bs * rw, dtype=torch.float32, device=dev).view(
        24, res_bs, rw
    )
    bad_host = torch.empty((10, bs + 1, rw), device=dev)
    with pytest.raises(ValueError, match="block size"):
        gather_sparse_prefill_cache(
            stub_rt.cache_ops,
            bad_host,
            plan,
            resident_cache=res,
            selected_mask=sel_mask,
        )
    full_cache = stub_rt.gather_prefill_cache(host, plan, resident_cache=res)
    sparse_cache = gather_sparse_prefill_cache(
        stub_rt.cache_ops, host, plan, resident_cache=res, selected_mask=sel_mask
    )
    sel_t = _i32(sel_idx, dev).long()
    assert sparse_cache.shape == (n_blks, bs, rw)
    assert torch.equal(
        _select_rows(sparse_cache, rw, sel_t), _select_rows(full_cache, rw, sel_t)
    )

    if dev.type == "cuda" and runtime_module._has_hisparse_ops():
        gen = torch.Generator("cpu").manual_seed(41)
        topk_cu = _i32([[0, 4, -1, -1], [1, 6, -1, -1], [-1, -1, -1, -1]], dev)
        reqs_cu = _i32([0, 1, 1], dev)
        sel_cu = hisparse_build_selected_mask(
            reqs_cu, plan.block_table, topk_cu, block_size=bs, num_staged_rows=n_staged
        )
        sel_rows, unsel_rows = (
            _nonzero_rows(sel_cu[0] != 0),
            _nonzero_rows(sel_cu[0] == 0),
        )
        assert sel_rows.numel() > 0 and unsel_rows.numel() > 0
        orig_empty, resolved_cfg = torch.empty, ResolvedHiSparseConfig(4, 8, 8)

        def poison_empty(*size: Any, **kwargs: Any) -> torch.Tensor:
            out = orig_empty(*size, **kwargs)
            return out.fill_(0xFF) if out.ndim == 3 and out.shape[1] == bs else out

        for width in (528, 656, 1152):
            host_pin = torch.randint(
                0, 128, (10, bs, width), generator=gen, dtype=torch.uint8
            ).pin_memory()
            res_gpu = torch.randint(
                128, 255, (24, res_bs, width), generator=gen, dtype=torch.uint8
            ).to(dev)
            rt = HiSparseRuntime(resolved_cfg, 2, width, torch.uint8, "cuda")
            rt.bind_source_cache(host_pin)
            full_gpu = rt.gather_prefill_cache(host_pin, plan, resident_cache=res_gpu)
            with monkeypatch.context() as local_mp:
                local_mp.setattr(prefill_module.torch, "empty", poison_empty)
                sparse_gpu = gather_sparse_prefill_cache(
                    rt.cache_ops,
                    host_pin,
                    plan,
                    resident_cache=res_gpu,
                    selected_mask=sel_cu,
                )
            assert torch.equal(
                _select_rows(sparse_gpu, width, sel_rows),
                _select_rows(full_gpu, width, sel_rows),
            )
            assert (_select_rows(sparse_gpu, width, unsel_rows) == 0xFF).all()
            _assert_no_cuda_sync(
                gather_sparse_prefill_cache,
                rt.cache_ops,
                host_pin,
                plan,
                resident_cache=res_gpu,
                selected_mask=sel_cu,
            )


def test_config_and_builder_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    assert HiSparseConfig().sparse_prefill_staging is False
    adapter = TypeAdapter(AttentionConfig)
    for json_val, expected in [("", False), ("true", True), ("false", False)]:
        payload = f'{{"sparse_prefill_staging": {json_val}}}' if json_val else "{}"
        parsed = adapter.validate_json(
            f'{{"hisparse_config": {payload}}}'
        ).hisparse_config
        assert parsed is not None and parsed.sparse_prefill_staging is expected

    hc_true = HiSparseConfig(sparse_prefill_staging=True)
    hc_false = HiSparseConfig(sparse_prefill_staging=False)
    assert hash_config(AttentionConfig(hisparse_config=hc_true)) != hash_config(
        AttentionConfig(hisparse_config=hc_false)
    )

    def _create_handle(*args: Any, **kwargs: Any) -> Any:
        return _make_mock_cache()

    monkeypatch.setattr(
        index_group_module, "create_hisparse_cache_handle", _create_handle
    )
    _setup_fp8_env(monkeypatch, torch.device("cpu"))
    layer_kw = {"row_width": 16, "kv_cache_dtype": "fp8_ds_mla"}
    zero_topk = torch.zeros((4, 4), dtype=torch.int32)
    cases = [
        (HiSparseConfig(), True, 1, 1, False),
        (hc_true, True, 1, 1, False),
        (hc_true, False, 1, 1, False),
        (hc_true, True, 2, 1, False),
        (hc_true, True, 1, 2, False),
        (hc_true, True, 1, 1, True),
    ]
    for cfg, is_leader, dcp, pcp, use_ub in cases:
        builder = SparseMLAIndexGroupBuilder(zero_topk, 4)
        vllm_cfg = _make_vllm_cfg(cfg, dcp, pcp, use_ub)
        group, idx = builder.register_layer(is_leader, vllm_cfg, **layer_kw)
        exp = cfg is hc_true and is_leader and dcp == 1 and pcp == 1 and not use_ub
        assert isinstance(group, HiSparseMLAIndexGroup) and idx == 0
        assert group.sparse_prefill_staging is exp
        if exp:
            follower_grp, follower_idx = builder.register_layer(
                False, vllm_cfg, **layer_kw
            )
            assert follower_grp is group and follower_idx == 1

    sentinel_cfg = cast(
        Any,
        SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=object()),
            model_config=None,
        ),
    )
    sentinel_grp, _ = SparseMLAIndexGroupBuilder(zero_topk, 4).register_layer(
        True, sentinel_cfg, **layer_kw
    )
    assert isinstance(sentinel_grp, HiSparseMLAIndexGroup)
    assert sentinel_grp.sparse_prefill_staging is False
    bare = object.__new__(HiSparseMLAIndexGroup)
    assert bare.sparse_prefill_staging is False and bare._prefill_mask_cache is None


@pytest.mark.parametrize("dev_str", DEVICES)
def test_index_group_stage_prefill_rows_and_microbatch_isolation(
    monkeypatch: pytest.MonkeyPatch, dev_str: str
) -> None:
    dev, bs, rw = torch.device(dev_str), 4, 8
    req_bt, seq_lens = _i32([[1, 2], [3, 4], [5, 0]], dev), _i32([8, 8, 4], dev)
    plan_lead = build_hisparse_prefill_staging_plan(req_bt, seq_lens, bs, 6)
    plan_follow = build_hisparse_prefill_staging_plan(req_bt, seq_lens, bs, 6)
    gather_masks: list[torch.Tensor] = []
    fb_calls: list[tuple[int, Any, int]] = []
    topk = torch.full((8, 4), 7, dtype=torch.int32, device=dev)
    topk[2:4].copy_(_i32([[0, 4, -1, -1], [2, 3, -1, -1]], dev))
    group = _make_mock_group(topk, gather_masks=gather_masks, fallback_calls=fb_calls)
    kv = torch.empty((2, bs, rw), device=dev)
    reqs, built = _i32([0, 0, 1, 2], dev), _spy_selected_mask(monkeypatch)
    meta_lead, meta_follow = _make_meta(plan_lead, reqs), _make_meta(plan_follow, reqs)
    stage_fn = group.stage_prefill_rows

    staged0, bt0, rows0 = stage_fn(0, kv, meta_lead)
    assert len(built) == 1 and torch.equal(built[-1], topk[2:4])
    assert bt0 is plan_lead.block_table and rows0.tolist() == [0, 1]
    assert staged0.shape == (plan_lead.row_ids.shape[1] // bs, bs, rw)
    assert isinstance(group._prefill_mask_cache, dict)
    cached_mask = next(iter(group._prefill_mask_cache.values()))[1]

    staged1, bt1, _ = stage_fn(1, kv, meta_follow)
    assert len(built) == 1 and bt1 is plan_follow.block_table
    assert staged1.shape == staged0.shape and group._prefill_mask_cache is None
    assert torch.equal(
        gather_masks[0], ((plan_lead.miss_mask != 0) & (cached_mask != 0)).int()
    )
    assert torch.equal(
        gather_masks[1], ((plan_follow.miss_mask != 0) & (cached_mask != 0)).int()
    )

    group.num_layers, group.caches = 1, group.caches[:1]
    stage_fn(0, kv, meta_lead)
    assert group._prefill_mask_cache is None

    group.num_layers = 2
    group.caches = _make_mock_group(
        topk, gather_masks=gather_masks, fallback_calls=fb_calls
    ).caches
    stage_fn(0, kv, meta_lead)
    assert isinstance(group._prefill_mask_cache, dict)
    prev_mask = next(iter(group._prefill_mask_cache.values()))[1].clone()
    topk_ptr = topk.data_ptr()
    topk[2:4].copy_(_i32([[1, 5, -1, -1], [0, 7, -1, -1]], dev))
    stage_fn(0, kv, meta_lead)
    assert isinstance(group._prefill_mask_cache, dict)
    assert group.logical_topk_indices.data_ptr() == topk_ptr
    assert not torch.equal(next(iter(group._prefill_mask_cache.values()))[1], prev_mask)

    group.prepare_for_batch(0, None)
    assert group._prefill_mask_cache is None

    stale = torch.ones((1, plan_lead.row_ids.shape[1]), dtype=torch.int32, device=dev)
    mq, dm = SparseMLAPrefillRoute.SPARSE_MQA, SparseMLAPrefillRoute.DENSE_MHA
    fb_cases = [
        (False, True, mq, 4, reqs),
        (True, False, mq, 4, reqs),
        (True, True, dm, 4, reqs),
        (True, True, mq, 2, reqs[:2]),
        (True, True, mq, 10, reqs),
    ]
    for staging_on, has_idx, route, n_act, req_slice in fb_cases:
        group.sparse_prefill_staging, group.has_indexer = staging_on, has_idx
        group._prefill_mask_cache = cast(Any, ("stale", stale))
        prev_built = len(built)
        _, fb_bt, _ = stage_fn(
            0, kv, _make_meta(plan_lead, req_slice, n_act=n_act, route=route)
        )
        assert len(built) == prev_built and group._prefill_mask_cache is None
        assert fb_bt is plan_lead.block_table and fb_calls[-1] == (0, plan_lead, 1)

    row_ids = torch.arange(12, dtype=torch.int32, device=dev).view(1, 12)
    ones = torch.ones_like(row_ids)

    def _unit_plan(bt_rows: list[list[int]]) -> HiSparsePrefillStagingPlan:
        return HiSparsePrefillStagingPlan(
            _i32(bt_rows, dev), row_ids, row_ids.clone(), ones, bs
        )

    def _last_nz_rows() -> list[int]:
        return _nonzero_rows(gather_masks[-1][0] != 0).tolist()

    plan_a, plan_b, plan_c = (
        _unit_plan([[2, 1]]),
        _unit_plan([[1, 2]]),
        _unit_plan([[0, 1]]),
    )
    group.sparse_prefill_staging, group.has_indexer = True, True
    reqs_3 = _i32([0, 0, 1], dev)
    topk.fill_(-1)
    topk[2, 0] = 0
    built.clear()
    meta_a, meta_b, meta_c = (
        _make_meta(plan_a, reqs_3, 2, 3),
        _make_meta(plan_b, reqs_3, 2, 3),
        _make_meta(plan_c, reqs_3, 2, 3, dm),
    )
    seq = [
        (0, meta_a, [8]),
        (0, meta_b, [4]),
        (0, meta_c, None),
        (1, meta_a, [8]),
        (1, meta_b, [4]),
    ]
    for layer_idx, meta, exp_rows in seq:
        stage_fn(layer_idx, kv, meta)
        if exp_rows is None:
            assert fb_calls[-1] == (0, plan_c, 1)
        else:
            assert _last_nz_rows() == exp_rows
        if layer_idx == 1 and exp_rows == [8]:
            assert isinstance(group._prefill_mask_cache, dict)
    assert group._prefill_mask_cache is None and len(built) == 2


@pytest.mark.parametrize("dev_str", DEVICES)
def test_index_group_gather_fp8_prefill(
    monkeypatch: pytest.MonkeyPatch, dev_str: str
) -> None:
    dev = torch.device(dev_str)
    objs: list[torch.Tensor] = []
    plan_lead, base, res_cache, buffers, rec_ids, built, stream = _setup_fp8_env(
        monkeypatch, dev, 4, objs
    )
    plan_follow, n_staged = _make_fp8_plan(dev, (21, 22))[0], plan_lead.row_ids.shape[1]
    topk = torch.full((8, 4), 7, dtype=torch.int32, device=dev)
    topk[2:3].copy_(_i32([[0, 2, -1, -1]], dev))
    group = _make_mock_group(topk, res_cache=res_cache, prefill_stream=stream)
    req_access_cnt, reqs = 0, _i32([0, 0, 1], dev)

    class _Fp8Meta:
        num_decodes, num_decode_tokens = 1, 2
        fp8_extra_metadata = SimpleNamespace(num_prefill_tokens=1)

        def __init__(
            self,
            plan: HiSparsePrefillStagingPlan,
            n_act: int | None = None,
            route: SparseMLAPrefillRoute = SparseMLAPrefillRoute.SPARSE_MQA,
        ) -> None:
            self.prefill = SimpleNamespace(host_staging_plan=plan, route=route)
            if n_act is not None:
                self.num_actual_tokens = n_act

        @property
        def req_id_per_token(self) -> torch.Tensor:
            nonlocal req_access_cnt
            assert not hasattr(self, "num_actual_tokens")
            req_access_cnt += 1
            return reqs

    meta_lead, meta_follow = _Fp8Meta(plan_lead), _Fp8Meta(plan_follow)
    for layer_idx, req_start in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        _run_fp8_chunk(
            group,
            layer_idx,
            req_start,
            plan_lead if layer_idx == 0 else plan_follow,
            meta_lead if layer_idx == 0 else meta_follow,
            dev,
            buffers,
        )
        assert len(built) == 1 and req_access_cnt == 1
        assert (group._prefill_mask_cache is None) is bool(layer_idx & req_start)

    assert objs[0] is objs[1] and objs[2] is objs[3] and objs[2] is not objs[0]
    assert rec_ids[0][0, base : base + 4].tolist() == [11, 0, -1, 0]
    assert rec_ids[2][0, base : base + 4].tolist() == [21, 0, -1, 0]

    stale = torch.ones((1, n_staged), dtype=torch.int32, device=dev)
    mq, dm = SparseMLAPrefillRoute.SPARSE_MQA, SparseMLAPrefillRoute.DENSE_MHA
    fb_cases = [
        (False, True, mq, 3, res_cache),
        (True, False, mq, 3, res_cache),
        (True, True, dm, 3, res_cache),
        (True, True, mq, 2, res_cache),
        (True, True, mq, 10, res_cache),
        (True, True, mq, 3, res_cache[:1, :0]),
    ]
    for staging_on, has_idx, route, n_act, cache_view in fb_cases:
        group.sparse_prefill_staging, group.has_indexer = staging_on, has_idx
        cast(Any, group.caches[0]).view.cache = cache_view
        group._prefill_mask_cache = cast(Any, ("stale", stale))
        _run_fp8_chunk(
            group, 0, 0, plan_lead, _Fp8Meta(plan_lead, n_act, route), dev, buffers
        )
        assert len(built) == 1 and group._prefill_mask_cache is None
        assert objs[-1] is plan_lead.gpu_row_ids


@pytest.mark.parametrize("dev_str", DEVICES)
def test_mtp_repointed_topk_buffer(
    monkeypatch: pytest.MonkeyPatch, dev_str: str
) -> None:
    dev, bs, rw, n_blks, n_dec, n_act = torch.device(dev_str), 4, 656, 8, 2, 4
    plan, _, res_cache, fp8_bufs, rec_ids, built, _ = _setup_fp8_env(
        monkeypatch, dev, 12
    )
    n_staged = plan.row_ids.shape[1]
    blk_bases = [
        int(plan.block_table[r, c].item()) * bs for r in (0, 1) for c in (0, 1)
    ]
    assert plan.gpu_row_ids is not None
    for base_row, start_val in zip(blk_bases, (10, 20, 30, 40)):
        plan.gpu_row_ids[0, base_row : base_row + bs] = _i32(
            range(start_val, start_val + bs), dev
        )

    gather_masks: list[torch.Tensor] = []

    def _create_handle(*args: Any, **kwargs: Any) -> Any:
        return _make_mock_cache(res_cache, gather_masks)

    monkeypatch.setattr(
        index_group_module, "create_hisparse_cache_handle", _create_handle
    )
    default_buf, topk_buf1, topk_buf2, extra_buf = (
        torch.full((n_blks, 4), -1, dtype=torch.int32, device=dev) for _ in range(4)
    )
    for buf, vals in [
        (default_buf, [7, 7]),
        (topk_buf1, [0, 2]),
        (topk_buf2, [1, 3]),
        (extra_buf, [4, 5]),
    ]:
        buf[n_dec:n_act, 0] = _i32(vals, dev)

    static_ctx: dict[str, Any] = {}
    layer_kw = {"row_width": rw, "kv_cache_dtype": "fp8_ds_mla"}
    vllm_cfg = _make_vllm_cfg(static_ctx=static_ctx)
    builder = SparseMLAIndexGroupBuilder(default_buf, n_blks)
    group, idx0 = builder.register_layer(True, vllm_cfg, **layer_kw)
    follower_grp, idx1 = builder.register_layer(False, vllm_cfg, **layer_kw)
    assert (
        isinstance(group, HiSparseMLAIndexGroup)
        and follower_grp is group
        and (idx0, idx1) == (0, 1)
    )

    def _make_layer(
        target_grp: HiSparseMLAIndexGroup,
        grp_idx: int,
        indexer: Any,
        indices_buf: Any,
        **extra_attrs: Any,
    ) -> SimpleNamespace:
        impl = object.__new__(sparse_mla_module.SparseMLACommonImpl)
        impl.init_topk_indices_buffer(indexer, indices_buf)
        impl.index_group, impl.index_group_index = target_grp, grp_idx
        return SimpleNamespace(
            hisparse_cache=target_grp.caches[grp_idx],
            index_group=target_grp,
            impl=impl,
            **extra_attrs,
        )

    indexer = SimpleNamespace(topk_indices_buffer=default_buf)
    static_ctx["skip_topk_follower"] = _make_layer(
        group, 1, None, default_buf, topk_indices_buffer=default_buf
    )
    lead_layer = _make_layer(group, 0, indexer, None)
    static_ctx["leader_layer"] = lead_layer
    indexer.topk_indices_buffer = topk_buf1

    reqs, prefill_reqs = _i32([0, 0, 1, 2], dev), _i32([0, 1], dev)
    meta = _make_meta(plan, reqs, n_dec, n_act)
    kv = torch.empty((2, bs, rw), dtype=torch.uint8, device=dev)

    def expected_mask(topk_slice: torch.Tensor) -> torch.Tensor:
        sel = hisparse_build_selected_mask(
            prefill_reqs,
            plan.block_table,
            topk_slice,
            block_size=bs,
            num_staged_rows=n_staged,
        )
        return ((plan.miss_mask != 0) & (sel != 0)).int()

    group.stage_prefill_rows(0, kv, meta)
    assert group._topk_owner is lead_layer and torch.equal(
        built[-1], topk_buf1[n_dec:n_act]
    )
    group.stage_prefill_rows(1, kv, meta)
    assert len(built) == 1 and group._prefill_mask_cache is None
    assert torch.equal(gather_masks[0], expected_mask(topk_buf1[n_dec:n_act]))
    assert torch.equal(gather_masks[1], expected_mask(topk_buf1[n_dec:n_act]))

    for layer_idx, req_start in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        _run_fp8_chunk(group, layer_idx, req_start, plan, meta, dev, fp8_bufs)
        assert len(built) == 2 and (group._prefill_mask_cache is None) is bool(
            layer_idx & req_start
        )

    for row_ids in rec_ids:
        assert row_ids[0, blk_bases[0] : blk_bases[0] + 4].tolist() == [10, 0, 0, 0]
        assert row_ids[0, blk_bases[2] : blk_bases[2] + 4].tolist() == [0, 0, 32, 0]
        assert row_ids[0, blk_bases[1] + 3] == 0 and row_ids[0, blk_bases[3] + 3] == 0

    for exp_calls, (owner_obj, repointed_buf) in enumerate(
        [(indexer, topk_buf2), (lead_layer.impl, extra_buf)], start=3
    ):
        owner_obj.topk_indices_buffer = repointed_buf
        group.prepare_for_batch(0, None)
        group.stage_prefill_rows(0, kv, meta)
        assert len(built) == exp_calls and torch.equal(
            built[-1], repointed_buf[n_dec:n_act]
        )
        assert torch.equal(gather_masks[-1], expected_mask(repointed_buf[n_dec:n_act]))

    qsa_topk = torch.full((n_blks, 5), -1, dtype=torch.int32, device=dev)
    qsa_default = torch.full_like(qsa_topk, 7)
    qsa_topk[n_dec:n_act, 0], qsa_topk[:, -1] = _i32([0, 2], dev), 6
    qsa_ctx: dict[str, Any] = {}
    qsa_grp, _ = SparseMLAIndexGroupBuilder(qsa_default[:, :-1], n_blks).register_layer(
        True, _make_vllm_cfg(static_ctx=qsa_ctx), **layer_kw
    )
    assert isinstance(qsa_grp, HiSparseMLAIndexGroup)
    qsa_layer = SimpleNamespace(
        hisparse_cache=qsa_grp.caches[0], impl=SimpleNamespace(index_group=qsa_grp)
    )
    qsa_ctx["qsa_layer_0"] = qsa_layer
    qsa_layer.topk_indices_buffer = qsa_topk
    qsa_grp.stage_prefill_rows(0, kv, meta)
    assert qsa_grp._topk_owner is qsa_layer
    assert torch.equal(built[-1], qsa_topk[n_dec:n_act, :-1])
    assert torch.equal(gather_masks[-1], expected_mask(qsa_topk[n_dec:n_act, :-1]))
