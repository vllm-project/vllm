# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    TritonSSUBackend,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    selective_state_update,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
)
from vllm.v1.worker.replayssm_utils import (
    ReplaySSMModelContext,
    _compact_replayssm_requests_kernel,
    _postprocess_replayssm_kernel,
)
from vllm.v1.worker.utils import allocate_kv_cache

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


@pytest.fixture(autouse=True)
def restore_backend_state():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old_backend = mod._mamba_ssu_backend
    old_replayssm_kernel = mod._flashinfer_replayssm_kernel
    yield
    mod._mamba_ssu_backend = old_backend
    mod._flashinfer_replayssm_kernel = old_replayssm_kernel


def _kv_cache_config_with_ssu(
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2,
) -> KVCacheConfig:
    spec = MambaSpec(
        block_size=16,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_type=mamba_type,
    )
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
    )


def _cuda_ints(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _replayssm_mixer(kv_cache, ring_start, committed, A):
    return SimpleNamespace(
        kv_cache=kv_cache,
        _replayssm_ring_start=ring_start,
        _replayssm_prev_num_accepted=committed,
        replayssm_buffer_len=16,
        A=A,
        mamba_config=MambaConfig(backend=MambaBackendEnum.FLASHINFER),
    )


@pytest.mark.parametrize("checkpointed", [False, True], ids=["replay", "checkpoint"])
@pytest.mark.parametrize(
    ("post_step", "layout", "ssm_dtype"),
    [
        pytest.param(False, KVCacheLayout.LBNHC, torch.float16, id="v1-fp16-lbnhc"),
        pytest.param(True, KVCacheLayout.BLHNC, torch.float32, id="v2-fp32-blhnc"),
    ],
)
def test_replayssm_materializes_only_accepted_boundary_with_compacted_rows(
    post_step, layout, ssm_dtype, checkpointed
):
    """Publish a boundary without modifying its live or cached-prefix sources."""
    try:
        from flashinfer.mamba.replayssm_materialize import (
            replayssm_materialize,  # noqa: F401
        )

        has_materializer = True
    except ImportError:
        has_materializer = False
    if not (torch.cuda.is_available() and has_materializer):
        reason = "requires CUDA and FlashInfer ReplaySSM materialization"
        if os.environ.get("VLLM_TEST_REQUIRE_REPLAYSSM") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    set_random_seed(42)
    slots, heads, dim, dstate, ring_len = 19, 8, 64, 128, 20
    spec = MambaSpec(
        block_size=16,
        shapes=(
            (heads, dim),
            (heads, dim, dstate),
            (heads, ring_len, dim),
            (heads, ring_len),
            (1, ring_len, dstate),
        ),
        dtypes=(
            torch.bfloat16,
            ssm_dtype,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
        ),
        use_flashinfer_replayssm=True,
        mamba_cache_mode="align",
        state_alignment=16,
        block_stride_alignment=16,
    )
    spec = replace(spec, page_size_padded=spec.page_size_bytes + 128)
    vllm_config = SimpleNamespace(
        cache_config=CacheConfig(),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    vllm_config.cache_config.kv_cache_layout = layout.name
    layer_names = [
        ["mixer", "neighbor"],
        ["other_group", "other_neighbor"],
        ["last_group"],
    ]
    config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec(names, spec) for names in layer_names],
        slots * (2 * spec.page_size_bytes + 24),
    )
    pages = allocate_kv_cache(config, torch.device("cuda"), layout)
    caches = {}
    for name, page in pages.items():
        layer = SimpleNamespace(
            get_state_shape=lambda: spec.shapes,
            get_state_dtype=lambda: spec.dtypes,
            state_alignment=16,
        )
        MambaBase.bind_kv_cache(layer, page)
        caches[name] = layer.kv_cache
    rings = {name: states[2:] for name, states in caches.items()}
    for name, states in caches.items():
        states[1].normal_(std=0.1)
        x, dt, B = rings[name]
        x.normal_()
        dt.fill_(0.01)
        B.normal_()
        assert (
            not x.is_contiguous() and not dt.is_contiguous() and not B.is_contiguous()
        )
    assert rings["mixer"][0].data_ptr() == rings["other_group"][0].data_ptr()
    assert rings["mixer"][0].data_ptr() == rings["last_group"][0].data_ptr()

    # Groups share backing pools, but own disjoint physical slots.
    # Each group's cached prefix remains immutable after copying to its live slot.
    for names, prefix, live in zip(layer_names, [1, 7, 13], [3, 9, 15], strict=True):
        for name in names:
            caches[name][1][live].copy_(caches[name][1][prefix])
    original = {name: states[1].clone() for name, states in caches.items()}
    ring_before = {
        name: [tensor.clone() for tensor in tensors] for name, tensors in rings.items()
    }
    A = -torch.ones(heads, device="cuda")
    null = NULL_BLOCK_ID
    block_tables = [
        [[2, 3], [null, 5], [4, 4], [null, null], [null, null]],
        [[8, 9], [null, 11], [10, 10], [null, null], [null, null]],
        [[14, 15], [null, 17], [16, 16], [null, null], [null, null]],
    ]
    histories = [
        {
            null: (11, 9),
            offset + 3: (7, 14 if checkpointed else 2),
            offset + 5: (2, 1),
            offset + 4: (6, 2),
        }
        for offset in (0, 6, 12)
    ]
    grouped = []
    for names, table, history in zip(layer_names, block_tables, histories, strict=True):
        starts, committed = [0] * slots, [0] * slots
        for slot, (start, count) in history.items():
            starts[slot], committed[slot] = start, count
        starts, committed = _cuda_ints(starts), _cuda_ints(committed)
        mixers = [
            _replayssm_mixer(caches[name], starts, committed, A) for name in names
        ]
        grouped.append((mixers, spec, _cuda_ints(table)))
    context = ReplaySSMModelContext.create(grouped, max_num_reqs=6)
    # Batch row 0 crosses token 16; row 1 rejects all drafts and stops at 15.
    # Row 2 uses in-place replay; row 3 has null slots in every group.
    # Row 4 is a padded request, and row 5 is unused buffer capacity.
    mapping = _cuda_ints([2, 0, 1, 3, -1])
    # Groups use different slots but preserve equal logical replay histories.
    context.reset_new_slots(
        idx_mapping=mapping,
        src_cols=_cuda_ints([0, 1, 1, 1]),
        dst_cols=_cuda_ints([1, 0, 0, 0]),
        num_reqs=5,
    )
    reset_histories = [
        {**history, offset + 2: history[offset + 3], offset + 5: (0, 0)}
        for offset, history in zip((0, 6, 12), histories, strict=True)
    ]
    for (mixers, _, _), expected in zip(grouped, reset_histories, strict=True):
        assert mixers[0]._replayssm_ring_start.tolist() == [
            expected.get(slot, (0, 0))[0] for slot in range(slots)
        ]
        assert mixers[0]._replayssm_prev_num_accepted.tolist() == [
            expected.get(slot, (0, 0))[1] for slot in range(slots)
        ]
    metadata = dict(
        idx_mapping=mapping,
        query_metadata=_cuda_ints([0, 4, 8, 12, 16, 20])
        if post_step
        else _cuda_ints([4] * 5),
        query_metadata_is_cumulative=post_step,
        num_computed_tokens=_cuda_ints([15, 17, 17, 17])
        if post_step
        else _cuda_ints([14] * 4),
        num_computed_is_post_step=post_step,
        num_accepted_tokens=_cuda_ints([1, 3, 3, 3]),
        is_prefilling=torch.zeros(5, dtype=torch.bool, device="cuda"),
        live_cols=_cuda_ints([1] * 4),
        num_reqs=5,
    )
    context.postprocess(**metadata)
    # After a checkpoint, the source state is canonical at the old ring end.
    # Only the two accepted tokens through the boundary remain to replay.
    plans = {0: (1, 2) if checkpointed else (7, 4), 2: (6, 4)}
    assert context.active_request_indices.tolist() == [0, 2, -1, -1, -1, -1]
    assert context.plan_ring_start.tolist() == [
        plans.get(row, (0, -1))[0] for row in range(6)
    ]
    assert context.plan_flush_count.tolist() == [
        plans.get(row, (0, -1))[1] for row in range(6)
    ]
    destinations: dict[int, set[int]] = {}
    for names, (mixers, _, _), table in zip(
        layer_names, grouped, block_tables, strict=True
    ):
        starts = mixers[0]._replayssm_ring_start
        committed = mixers[0]._replayssm_prev_num_accepted
        for row, (start, flush) in plans.items():
            dst, src = table[row]
            for name in names:
                destinations.setdefault(caches[name][1].data_ptr(), set()).add(dst)
                x, dt, B = rings[name]
                expected = original[name][src].float().clone()
                for offset in range(flush):
                    pos = (start + offset) % ring_len
                    delta = dt[src, :, pos]
                    expected *= torch.exp(delta * A)[:, None, None]
                    expected += (
                        delta[:, None, None]
                        * x[src, :, pos].float()[:, :, None]
                        * B[src, 0, pos].float()[None, None, :]
                    )
                torch.testing.assert_close(
                    caches[name][1][dst].float(), expected, atol=2e-3, rtol=2e-3
                )
            assert starts[dst].item() == 0
            assert committed[dst].item() == 0
            if src != dst:
                assert starts[src].item() == start
                assert committed[src].item() == flush + 1
    # Check all other slots, including immutable prefixes and each live tail.
    for name, states in caches.items():
        for slot in set(range(slots)) - destinations[states[1].data_ptr()]:
            torch.testing.assert_close(
                states[1][slot], original[name][slot], atol=0, rtol=0
            )
        for tensor, before in zip(rings[name], ring_before[name], strict=True):
            torch.testing.assert_close(tensor, before, atol=0, rtol=0)

    # Old plans beyond the new batch must not replay a second time.
    before_shrink = {name: states[1].clone() for name, states in caches.items()}
    mapping[:2].fill_(-1)
    metadata["num_reqs"] = 2
    context.postprocess(**metadata)
    assert context.active_request_indices.tolist() == [-1] * 6
    for name, states in caches.items():
        torch.testing.assert_close(states[1], before_shrink[name], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("post_step", [False, True], ids=["v1", "v2"])
def test_replayssm_skips_canonical_self_copy_but_resets_prefill_trackers(post_step):
    """Skip prefill self-copies, reset trackers, and preserve in-place MTP replay."""
    mixer = _replayssm_mixer(
        kv_cache=(
            torch.empty(0, device="cuda"),
            torch.empty(6, 1, 1, 1, device="cuda"),
            torch.empty(6, 1, 20, 1, device="cuda"),
            torch.empty(6, 1, 20, device="cuda"),
            torch.empty(6, 1, 20, 1, device="cuda"),
        ),
        ring_start=_cuda_ints([3] * 6),
        committed=_cuda_ints([2] * 6),
        A=torch.empty(1, device="cuda"),
    )
    spec = MambaSpec(
        block_size=16,
        shapes=tuple(tensor.shape[1:] for tensor in mixer.kv_cache),
        dtypes=tuple(tensor.dtype for tensor in mixer.kv_cache),
        mamba_cache_mode="align",
        use_flashinfer_replayssm=True,
    )
    table = _cuda_ints(
        [
            [0, 1, 2],
            [3, 4, 5],
            [NULL_BLOCK_ID] * 3,
            [NULL_BLOCK_ID] * 3,
        ]
    )
    context = ReplaySSMModelContext.create([([mixer], spec, table)], max_num_reqs=4)
    # Reordered rows: canonical prefill in slot 2; in-place MTP replay in slot 4;
    # padded slot; padded request.
    _postprocess_replayssm_kernel[(4, 1)](
        _cuda_ints([1, 2, 0, -1]),
        _cuda_ints([0, 48, 52, 68, 68]) if post_step else _cuda_ints([48, 4, 16, 0]),
        _cuda_ints([16, 48, 32]) if post_step else _cuda_ints([0, 0, 28]),
        _cuda_ints([1, 1, 4]),
        torch.tensor([True, False, True, True], device="cuda"),
        _cuda_ints([0, 2, 1]),
        context.block_table_ptrs,
        context.ring_start_ptrs,
        context.num_committed_ptrs,
        context.src_slots_ptrs,
        context.dst_slots_ptrs,
        context.plan_ring_start,
        context.plan_flush_count,
        context.block_table_strides,
        context.num_layers,
        MAX_NUM_REQS=4,
        MAMBA_BLOCK_SIZE=16,
        LOGICAL_WINDOW=16,
        RING_BUFFER_LEN=20,
        PAD_SLOT_ID=NULL_BLOCK_ID,
        QUERY_METADATA_IS_CUMULATIVE=post_step,
        NUM_COMPUTED_IS_POST_STEP=post_step,
        HAS_IDX_MAPPING=True,
        MATERIALIZE_PREFIXES=True,
        LIVE_COL_IS_ZERO=False,
    )
    _compact_replayssm_requests_kernel[(1,)](
        context.plan_flush_count,
        context.active_request_indices,
        4,
        MAX_NUM_REQS=4,
        BLOCK_SIZE=4,
    )

    assert context.plan_flush_count.tolist() == [-1, 6, -1, -1]
    assert context.active_request_indices.tolist() == [1, -1, -1, -1]
    assert context.src_slots[0, 1].item() == 4
    assert context.dst_slots[0, 1].item() == 4
    assert mixer._replayssm_ring_start.tolist() == [3, 0, 0, 3, 0, 3]
    assert mixer._replayssm_prev_num_accepted.tolist() == [2, 0, 0, 2, 0, 2]


def test_default_backend_is_triton():
    initialize_mamba_ssu_backend(MambaConfig(), _kv_cache_config_with_ssu())
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)
    assert backend.name == "triton"


def test_explicit_triton_backend():
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.TRITON), _kv_cache_config_with_ssu()
    )
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
def test_flashinfer_backend_init():
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.FLASHINFER), _kv_cache_config_with_ssu()
    )
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, FlashInferSSUBackend)
    assert backend.name == "flashinfer"


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("simple", "simple"),
        ("vertical", "vertical"),
        ("horizontal", "horizontal"),
    ],
)
def test_flashinfer_forwards_ssu_algorithm(
    algorithm: MambaSSUAlgorithm | None,
    expected: MambaSSUAlgorithm,
    monkeypatch,
):
    import flashinfer.mamba

    kernel = Mock()
    monkeypatch.setattr(flashinfer.mamba, "selective_state_update", kernel)
    backend = FlashInferSSUBackend(
        MambaConfig(
            backend=MambaBackendEnum.FLASHINFER,
            ssu_algorithm=algorithm,
        )
    )

    tensor = torch.empty(1)
    backend(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
    )

    assert kernel.call_args.kwargs["algorithm"] == expected


def test_uninitialized_backend_raises():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    # restore_backend_state (autouse) puts the global back afterwards.
    mod._mamba_ssu_backend = None
    with pytest.raises(RuntimeError, match="not been initialized"):
        get_mamba_ssu_backend()


@pytest.mark.parametrize(
    "mamba_type",
    [
        MambaAttentionBackendEnum.LINEAR,
        MambaAttentionBackendEnum.GDN_ATTN,
        MambaAttentionBackendEnum.SHORT_CONV,
    ],
)
def test_init_is_noop_for_non_ssu_mamba_type(mamba_type):
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    try:
        initialize_mamba_ssu_backend(
            MambaConfig(), _kv_cache_config_with_ssu(mamba_type)
        )
        assert mod._mamba_ssu_backend is None
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_mamba_ssu_backend()
    finally:
        mod._mamba_ssu_backend = old


@pytest.mark.skipif(HAS_FLASHINFER, reason="flashinfer is installed")
def test_flashinfer_import_error():
    with pytest.raises(ImportError, match="FlashInfer is required"):
        FlashInferSSUBackend(MambaConfig())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_basic_call():
    set_random_seed(0)
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.TRITON), _kv_cache_config_with_ssu()
    )
    device = "cuda"
    batch_size = 2
    dim = 64
    dstate = 16

    state = torch.randn(batch_size, dim, dstate, device=device)
    x = torch.randn(batch_size, dim, device=device)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device)
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
    )
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    ("backend", "num_speculative_tokens", "expected_ring_len"),
    [
        (MambaBackendEnum.TRITON, 0, 16),
        (MambaBackendEnum.FLASHINFER, 0, 17),
        (MambaBackendEnum.FLASHINFER, 3, 20),
    ],
)
def test_replayssm_physical_ring_shape(
    backend, num_speculative_tokens, expected_ring_len
):
    base_shapes = ((8, 4), (8, 4, 16))
    shapes = MambaStateShapeCalculator.append_replayssm_ring(
        base_shapes,
        n_groups=4,
        tp_world_size=2,
        logical_window=16,
        backend=backend,
        num_speculative_tokens=num_speculative_tokens,
    )

    assert shapes == (
        *base_shapes,
        (8, expected_ring_len, 4),
        (8, expected_ring_len),
        (2, expected_ring_len, 16),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("post_step", [False, True])
def test_replayssm_postprocess_commits_staged_transition(post_step):
    """Keep group histories equal across checkpoints without touching other slots."""
    ring_start = [_cuda_ints([11, 3, 7, 13]), _cuda_ints([11, 7, 3, 13])]
    committed = [_cuda_ints([9, 2, 9, 17]), _cuda_ints([9, 9, 2, 17])]
    cache = (
        torch.empty(4, 1, device="cuda"),
        torch.empty(4, 1, 1, 1, device="cuda"),
        torch.empty(4, 1, 20, 1, device="cuda"),
        torch.empty(4, 1, 20, device="cuda"),
        torch.empty(4, 1, 20, 1, device="cuda"),
    )
    spec = MambaSpec(
        block_size=256,
        shapes=tuple(tensor.shape[1:] for tensor in cache),
        dtypes=tuple(tensor.dtype for tensor in cache),
        mamba_cache_mode="none",
        use_flashinfer_replayssm=True,
    )
    # Neighboring request rows provide valid canary IDs, making erroneous
    # column reads deterministically corrupt another slot rather than crash.
    block_tables = [
        torch.full((256, 1), 3, dtype=torch.int32, device="cuda") for _ in range(2)
    ]
    grouped = []
    for group_idx, table in enumerate(block_tables):
        table[0, 0] = group_idx + 1
        mixer = _replayssm_mixer(
            cache,
            ring_start[group_idx],
            committed[group_idx],
            torch.empty(1, device="cuda"),
        )
        grouped.append(([mixer], spec, table))
    context = ReplaySSMModelContext.create(grouped, max_num_reqs=1)
    computed = 50_000
    history_tokens = checkpoint_tokens = 0
    # Reset a later prefill chunk, then reject, partially accept, and fully accept.
    for step, accepted in enumerate([1] + [4, 1, 3, 2] * 5):
        prefilling = step == 0
        query_len = 4
        if not prefilling:
            if history_tokens - checkpoint_tokens + query_len > 16:
                checkpoint_tokens = history_tokens
            history_tokens += accepted
        computed_after = computed + (query_len if prefilling else accepted)
        context.postprocess(
            idx_mapping=_cuda_ints([0]) if post_step else None,
            query_metadata=_cuda_ints([0, query_len])
            if post_step
            else _cuda_ints([query_len]),
            query_metadata_is_cumulative=post_step,
            num_computed_tokens=_cuda_ints([computed_after if post_step else computed]),
            num_computed_is_post_step=post_step,
            num_accepted_tokens=_cuda_ints([accepted]),
            is_prefilling=torch.tensor([prefilling], device="cuda"),
            live_cols=None,
            num_reqs=1,
        )
        expected_committed = history_tokens - checkpoint_tokens
        expected_start = checkpoint_tokens % 20
        for group_idx in range(2):
            expected_starts, expected_counts = [11, 7, 7, 13], [9, 9, 9, 17]
            expected_starts[group_idx + 1] = expected_start
            expected_counts[group_idx + 1] = expected_committed
            assert ring_start[group_idx].tolist() == expected_starts, step
            assert committed[group_idx].tolist() == expected_counts, step
        assert ring_start[0][1].item() == ring_start[1][2].item()
        assert committed[0][1].item() == committed[1][2].item()
        assert 0 <= expected_committed <= 16
        assert expected_committed + query_len <= 20
        assert context.plan_ring_start.item() == 0
        assert context.plan_flush_count.item() == -1
        computed = computed_after
    assert checkpoint_tokens >= 40  # Exercise more than one physical ring wrap.


@pytest.mark.parametrize(
    ("query_start_loc", "expected_x_shape", "expected_B_shape", "expected_max_seqlen"),
    [
        pytest.param([0, 4, 8], (2, 4, 2, 4), (2, 4, 1, 8), None, id="dense"),
        pytest.param([0, 4, 6], (1, 6, 2, 4), (1, 6, 1, 8), 4, id="packed"),
    ],
)
def test_replayssm_mixer_selects_mtp_layout(
    monkeypatch,
    query_start_loc,
    expected_x_shape,
    expected_B_shape,
    expected_max_seqlen,
):
    import vllm.model_executor.layers.mamba.mamba_mixer2 as mod
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as dispatch

    mixer = mod.MambaMixer2.__new__(mod.MambaMixer2)
    torch.nn.Module.__init__(mixer)
    mixer.prefix = "mixer"
    mixer.tped_intermediate_size = 0
    mixer.tped_conv_size = 1
    mixer.tped_dt_size = 2
    mixer.num_heads = 2
    mixer.head_dim = 4
    mixer.n_groups = mixer.tp_size = 1
    mixer.ssm_state_size = 8
    mixer.num_spec = 3
    mixer.use_replayssm = True
    mixer.use_flashinfer_replayssm = True
    mixer.replayssm_buffer_len = 16
    mixer.mamba_config = MambaConfig(backend=MambaBackendEnum.FLASHINFER)
    mixer.cache_config = SimpleNamespace(mamba_block_size=16, mamba_cache_mode="none")
    mixer.conv_weights = torch.empty(0)
    mixer.conv1d = SimpleNamespace(bias=None)
    mixer.activation = "silu"
    mixer.A = torch.empty(2)
    mixer.dt_bias = torch.empty(2)
    mixer.D = torch.empty(2)
    mixer._replayssm_ring_start = torch.zeros(3, dtype=torch.int32)
    mixer._replayssm_prev_num_accepted = torch.zeros(3, dtype=torch.int32)
    mixer.kv_cache = (
        torch.empty(3, 1),
        torch.empty(3, 2, 4, 8),
        torch.empty(3, 2, 20, 4),
        torch.empty(3, 2, 20),
        torch.empty(3, 1, 20, 8),
    )

    num_decode_tokens = query_start_loc[-1]
    query_start_loc_d = torch.tensor(query_start_loc, dtype=torch.int32)
    metadata = Mamba2AttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=num_decode_tokens,
        num_reqs=2,
        has_initial_states_p=None,
        query_start_loc_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=torch.tensor([[1], [2]], dtype=torch.int32),
        query_start_loc_d=query_start_loc_d,
        num_accepted_tokens=torch.tensor([4, 2], dtype=torch.int32),
        seq_lens=torch.tensor([104, 102], dtype=torch.int32),
        replayssm_scratch=(torch.empty(0), torch.empty(0), torch.empty(0)),
        replayssm_state_indices_d=torch.tensor([1, 2], dtype=torch.int32),
    )

    def split_hidden_states_B_C(values):
        tokens = values.size(0)
        return (
            torch.empty(tokens, 8),
            torch.empty(tokens, 8),
            torch.empty(tokens, 8),
        )

    mixer.split_hidden_states_B_C_fn = split_hidden_states_B_C
    kernel = Mock()
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={mixer.prefix: metadata}),
    )
    monkeypatch.setattr(
        mod, "causal_conv1d_update", lambda values, *args, **kwargs: values
    )
    monkeypatch.setattr(dispatch, "_flashinfer_replayssm_kernel", kernel)

    mixer.conv_ssm_forward(
        torch.empty(num_decode_tokens, 3), torch.empty(num_decode_tokens, 8)
    )

    args = kernel.call_args.args
    assert args[6].shape == expected_x_shape
    assert args[7].shape == expected_x_shape
    assert args[9].shape == expected_B_shape
    assert args[10].shape == expected_B_shape
    assert args[11].shape == expected_x_shape
    if expected_max_seqlen is None:
        assert kernel.call_args.kwargs["cu_seqlens"] is None
    else:
        assert kernel.call_args.kwargs["cu_seqlens"] is query_start_loc_d
    assert kernel.call_args.kwargs["max_seqlen"] == expected_max_seqlen


def test_replayssm_ring_shape_extends_groups_for_head_shards():
    """B_cache retains one replicated group on each head shard."""
    base_shapes = ((2, 64), (2, 64, 128))
    shapes = MambaStateShapeCalculator.append_replayssm_ring(
        base_shapes,
        n_groups=1,
        tp_world_size=2,
        logical_window=16,
        backend=MambaBackendEnum.TRITON,
    )

    assert shapes == (*base_shapes, (2, 16, 64), (2, 16), (1, 16, 128))
