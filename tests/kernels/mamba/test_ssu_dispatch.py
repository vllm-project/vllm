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
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
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
from vllm.v1.worker.mamba_utils import _validate_replayssm_read_state_spec
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
        [[null, 3], [null, 5], [4, 4], [null, null], [null, null]],
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
    # Row 0 crosses token 16, but only later groups have a destination.
    # Row 1 rejects all drafts and stops at 15.
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
        {**histories[0], 5: (0, 0)},
        {**histories[1], 8: histories[1][9], 11: (0, 0)},
        {**histories[2], 14: histories[2][15], 17: (0, 0)},
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
            if dst == null:
                assert starts[src].item() == start
                assert committed[src].item() == flush + 1
                assert (starts[null].item(), committed[null].item()) == (11, 9)
                continue
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
    mixer.replayssm_state_dtype = "auto"
    mixer.replayssm_checkpoint_mode = "shared"
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


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("requires the NVIDIA CUDA platform")


def _bytes(value):
    return value.contiguous().view(torch.uint8).cpu()


def _encode_rows(value, dtype):
    value = value.float().cpu()
    qmax = 127 if dtype == torch.int8 else 448
    amax = value.abs().amax(-1)
    multiplier = torch.where(amax == 0, 1.0, qmax / amax)
    scaled = value * multiplier[..., None]
    if dtype == torch.int8:
        scaled = scaled.round()
    return scaled.to(dtype), multiplier.reciprocal()


def _assert_rows(value, payload, scale):
    expected_payload, expected_scale = _encode_rows(value, payload.dtype)
    assert torch.equal(_bytes(payload), _bytes(expected_payload))
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)


def _source(dtype):
    qmax = 127 if dtype == torch.int8 else 448
    value = torch.ones(3, 16, 64, 128)
    value[..., -1] = qmax
    factors = torch.exp2(torch.arange(16 * 64).reshape(16, 64) % 3 - 2)
    return value * factors[None, ..., None], factors


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_shared_prefill_selected_slots_and_reuse(dtype):
    """Pack, gather, and replace prefill states without touching padding."""
    _require_cuda()
    from vllm.model_executor.layers.mamba.ops.replayssm_quantization import (
        dequantize_replayssm_state,
        quantize_replayssm_state,
    )

    source, _ = _source(dtype)
    source[0, 0, 0].zero_()
    source[2] *= 2
    source = source.cuda()
    # A larger outer stride reproduces padded cache pages.
    payload_pages = torch.full((6, 16, 64, 128), 7, dtype=dtype, device="cuda")
    scale_pages = torch.full((6, 16, 64), 7.0, device="cuda")
    payload, scale = payload_pages[::2], scale_pages[::2]
    ids = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")
    quantize_replayssm_state(
        source,
        payload,
        scale,
        ids,
        source_indices=torch.tensor([2, -1, 0], device="cuda"),
    )
    _assert_rows(source[2], payload[2], scale[2])
    _assert_rows(source[0], payload[1], scale[1])
    assert torch.all(payload[0].float() == 7)
    assert torch.all(scale[0] == 7)
    decoded = dequantize_replayssm_state(
        payload,
        scale,
        torch.tensor([2, 1, 1, -1], dtype=torch.int32, device="cuda"),
        torch.tensor([True, True, False, True], device="cuda"),
    )
    assert decoded.dtype == torch.float32
    expected = torch.zeros_like(decoded)
    expected[0] = payload[2].float() * scale[2, ..., None]
    expected[1] = payload[1].float() * scale[1, ..., None]
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    torch.testing.assert_close(decoded[:2], source[[2, 0]], rtol=0, atol=0)
    preserved_payload, preserved_scale = _bytes(payload[1]), scale[1].clone()

    source.mul_(4)
    ids.copy_(torch.tensor([2, -1, -1], dtype=torch.int32, device="cuda"))
    quantize_replayssm_state(source, payload, scale, ids)
    _assert_rows(source[0], payload[2], scale[2])
    assert torch.equal(_bytes(payload[1]), preserved_payload)
    assert torch.equal(scale[1], preserved_scale)
    assert torch.all(payload_pages[1::2].float() == 7)
    assert torch.all(scale_pages[1::2] == 7)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_shared_graph_mixed_flush_checkpoint_bytes(dtype):
    """Two graph replays update only the flushing slot and preserve its scales."""
    _require_cuda()
    from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu

    from vllm.model_executor.layers.mamba.ops.replayssm_quantization import (
        quantize_replayssm_state,
    )

    source, factors = _source(dtype)
    source[0].zero_()
    payload = torch.zeros_like(source, dtype=dtype, device="cuda")
    scale = torch.ones(3, 16, 64, device="cuda")
    quantize_replayssm_state(
        source.cuda(), payload, scale, torch.tensor([-1, 1, 2], device="cuda")
    )
    initial_payload, initial_scale = payload.clone(), scale.clone()
    x = (
        (factors / 8)
        .cuda()
        .to(torch.bfloat16)[None, None]
        .expand(3, 1, -1, -1)
        .contiguous()
    )
    b = torch.zeros(3, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    b[..., 0] = 1
    x_cache = x[:, 0, :, None].expand(-1, -1, 17, -1).contiguous()
    b_cache = b[:, 0, :, None].expand(-1, -1, 17, -1).contiguous()
    dt_cache = torch.ones(3, 16, 17, device="cuda")
    out = torch.empty_like(x)
    args = dict(
        state=payload,
        state_scale=scale,
        x_cache=x_cache,
        B_cache=b_cache,
        dt_cache=dt_cache,
        ring_start=torch.zeros(3, device="cuda", dtype=torch.int32),
        prev_num_accepted_tokens=torch.tensor(
            [0, 0, 16], device="cuda", dtype=torch.int32
        ),
        x=x,
        dt=torch.ones(3, 1, 16, 1, device="cuda", dtype=torch.bfloat16).expand(
            -1, -1, -1, 64
        ),
        A=torch.zeros(16, 1, 1, device="cuda").expand(-1, 64, 128),
        B=b,
        C=b,
        out=out,
        state_batch_indices=torch.tensor([2, 1, -1], device="cuda"),
        algorithm="monolith",
        d_split=1,
    )
    checkpointing_ssu(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        checkpointing_ssu(**args)
    payload.copy_(initial_payload)
    scale.copy_(initial_scale)
    for iteration in range(1, 3):
        graph.replay()
        expected = source.clone()
        expected[2, ..., 0] += iteration * 2 * factors
        _assert_rows(expected, payload, scale)
        expected_out = torch.stack(
            ((1 + iteration * 2 + 0.125) * factors, 1.125 * factors)
        ).to(torch.bfloat16)
        assert torch.equal(out[:2, 0].cpu(), expected_out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.int8])
def test_shared_bolt_t8_acceptance_and_ring_wrap(dtype):
    """Rejected candidates must not enter the next Bolt checkpoint or output."""
    _require_cuda()
    from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu

    from vllm.model_executor.layers.mamba.ops.replayssm_quantization import (
        quantize_replayssm_state,
    )

    heads, groups, tokens, ring_len = 128, 8, 8, 24
    source = torch.zeros(4, heads, 64, 128)
    source[..., -1] = 127 if dtype == torch.int8 else 448
    source[0].zero_()
    state = source.to(device="cuda", dtype=dtype)
    scales = None
    if dtype != torch.bfloat16:
        scales = torch.ones(4, heads, 64, device="cuda")
        quantize_replayssm_state(
            source.cuda(), state, scales, torch.tensor([-1, 1, 2, 3], device="cuda")
        )
    x_ring = torch.full(
        (4, heads, ring_len, 64), 1 / 8, device="cuda", dtype=torch.bfloat16
    )
    b_ring = torch.zeros(4, groups, ring_len, 128, device="cuda", dtype=torch.bfloat16)
    b_ring[..., 0] = 1
    dt_ring = torch.ones(4, heads, ring_len, device="cuda")
    starts = torch.tensor([0, 21, 19, 23], device="cuda", dtype=torch.int32)
    history = torch.tensor([0, 8, 9, 16], device="cuda", dtype=torch.int32)
    slots = torch.tensor([3, 1, 2, -1], device="cuda", dtype=torch.int32)
    increments = torch.arange(1, tokens + 1).float() / 8
    x = (
        increments[None, :, None, None]
        .expand(4, tokens, heads, 64)
        .cuda()
        .bfloat16()
        .contiguous()
    )
    b = torch.zeros(4, tokens, groups, 128, device="cuda", dtype=torch.bfloat16)
    b[..., 0] = 1
    out = torch.empty_like(x)
    args = dict(
        state=state,
        state_scale=scales,
        x_cache=x_ring,
        B_cache=b_ring,
        dt_cache=dt_ring,
        ring_start=starts,
        prev_num_accepted_tokens=history,
        x=x,
        dt=torch.ones(4, tokens, heads, 1, device="cuda", dtype=torch.bfloat16).expand(
            -1, -1, -1, 64
        ),
        A=torch.zeros(heads, 1, 1, device="cuda").expand(-1, 64, 128),
        B=b,
        C=b,
        out=out,
        state_batch_indices=slots,
        algorithm="monolith",
        d_split=1,
    )
    saved = [value.clone() for value in (state, x_ring, b_ring, dt_ring)]
    saved_scale = None if scales is None else scales.clone()
    checkpointing_ssu(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        checkpointing_ssu(**args)
    for value, initial in zip((state, x_ring, b_ring, dt_ring), saved, strict=True):
        value.copy_(initial)
    if scales is not None:
        scales.copy_(saved_scale)

    expected = source.clone()
    expected_ring = x_ring.cpu().clone()
    starts_cpu, history_cpu = starts.cpu(), history.cpu()
    for accepted in ([1, 7, 8], [8, 1, 7], [7, 8, 1]):
        expected_output = []
        for row, slot in enumerate((3, 1, 2)):
            start, count = int(starts_cpu[slot]), int(history_cpu[slot])
            positions = (start + torch.arange(count)) % ring_len
            old_sum = expected_ring[slot, 0, positions, 0].float().sum()
            current = expected[slot, 0, 0, 0] + old_sum
            expected_output.append((current + increments.cumsum(0)).bfloat16())
            if count + tokens > 16:
                expected[slot, ..., 0] += old_sum
                if scales is None:
                    expected[slot] = expected[slot].bfloat16().float()
                else:
                    packed, row_scale = _encode_rows(expected[slot], dtype)
                    expected[slot] = packed.float() * row_scale[..., None]
            written = (start + count + torch.arange(tokens)) % ring_len
            expected_ring[slot, :, written, :] = x[row].cpu().permute(1, 0, 2)
        graph.replay()
        for row, slot in enumerate((3, 1, 2)):
            if scales is None:
                torch.testing.assert_close(
                    state[slot].cpu(), expected[slot].bfloat16(), rtol=0, atol=0
                )
            else:
                _assert_rows(expected[slot], state[slot], scales[slot])
            torch.testing.assert_close(
                out[row].cpu(),
                expected_output[row][:, None, None].expand(-1, heads, 64),
                rtol=0,
                atol=0,
            )
            count = int(history_cpu[slot])
            flushed = count + tokens > 16
            if flushed:
                starts_cpu[slot] = (starts_cpu[slot] + count) % ring_len
            history_cpu[slot] = (0 if flushed else count) + accepted[row]
        torch.testing.assert_close(x_ring.cpu(), expected_ring, rtol=0, atol=0)
        assert torch.equal(_bytes(state[0]), _bytes(saved[0][0]))
        starts.copy_(starts_cpu)
        history.copy_(history_cpu)


@pytest.mark.parametrize("state_dtype", ["fp8", "int8"])
def test_replayssm_quantized_cache_copies_payload_and_scale(state_dtype):
    """A checkpoint record must copy its payload and FP32 scale together."""
    state_shape = (16, 64, 128)
    spec = MambaSpec(
        block_size=16,
        shapes=(
            (3, 64),
            state_shape,
            (16, 17, 64),
            (16, 17),
            (1, 17, 128),
            *MambaStateShapeCalculator.replayssm_read_shapes(state_shape, state_dtype),
        ),
        dtypes=(
            torch.bfloat16,
            MambaStateDtypeCalculator.replayssm_checkpoint_dtype(state_dtype),
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            *MambaStateDtypeCalculator.replayssm_read_dtypes(state_dtype),
        ),
    )
    _validate_replayssm_read_state_spec(spec)
    with pytest.raises(ValueError, match="requires an FP32 scale page"):
        _validate_replayssm_read_state_spec(
            replace(spec, shapes=spec.shapes[:5], dtypes=spec.dtypes[:5])
        )
    with pytest.raises(ValueError, match="Invalid quantized ReplaySSM"):
        _validate_replayssm_read_state_spec(
            replace(spec, shapes=(*spec.shapes[:5], (16, 65)))
        )
    with pytest.raises(ValueError, match="Invalid quantized ReplaySSM"):
        _validate_replayssm_read_state_spec(
            replace(spec, dtypes=(*spec.dtypes[:5], torch.float16))
        )


@pytest.mark.parametrize(
    ("read_dtype", "mode", "expected_shapes", "expected_dtypes", "expected_bytes"),
    [
        (
            "fp8",
            "separate",
            ((16, 64, 128), (16, 64), (1,)),
            (torch.float8_e4m3fn, torch.float32, torch.int64),
            135176,
        ),
        (
            "nvfp4",
            "separate",
            ((8, 128, 64), (8, 4, 256), (3, 1), (1,), (1,)),
            (torch.uint8, torch.uint8, torch.float32, torch.int32, torch.int64),
            73752,
        ),
    ],
)
def test_replayssm_read_state_storage(
    read_dtype, mode, expected_shapes, expected_dtypes, expected_bytes
):
    """Keep payload packing and scale storage in the allocated cache record."""
    shapes = MambaStateShapeCalculator.replayssm_read_shapes(
        (16, 64, 128), read_dtype, mode, n_groups=1
    )
    dtypes = MambaStateDtypeCalculator.replayssm_read_dtypes(read_dtype, mode)
    assert shapes == expected_shapes
    assert dtypes == expected_dtypes
    assert (
        sum(
            torch.empty(shape, dtype=dtype).nbytes
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )
        == expected_bytes
    )


@pytest.mark.parametrize("read_dtype", ["fp8", "nvfp4"])
def test_replayssm_read_state_rejects_unsupported_dimensions(read_dtype):
    with pytest.raises(ValueError, match="head dimension 64 and state size 128"):
        MambaStateShapeCalculator.replayssm_read_shapes(
            (2, 32, 128), read_dtype, "separate"
        )


@pytest.mark.parametrize("read_dtype", ["fp8", "nvfp4"])
@pytest.mark.parametrize("draft_tokens", [0, 7])
def test_replayssm_pending_bolt_storage(read_dtype, draft_tokens):
    """Reserve accepted history and the complete unconfirmed verification window."""
    ring_len = 17 + draft_tokens
    rings = MambaStateShapeCalculator.append_replayssm_ring(
        base_shapes=((3, 64), (128, 64, 128)),
        n_groups=8,
        tp_world_size=1,
        logical_window=16,
        backend=MambaBackendEnum.FLASHINFER,
        num_speculative_tokens=draft_tokens,
        checkpoint_mode="separate",
        pending_state=True,
    )[2:]
    assert rings == ((128, ring_len, 64), (128, ring_len), (8, ring_len, 128))
    extras = MambaStateShapeCalculator.replayssm_read_shapes(
        (128, 64, 128), read_dtype, "separate", n_groups=8, pending_state=True
    )
    assert extras[-1] == (2,)
    spec = MambaSpec(
        block_size=16,
        shapes=((3, 64), (128, 64, 128), *rings, *extras),
        dtypes=(
            torch.bfloat16,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            *MambaStateDtypeCalculator.replayssm_read_dtypes(read_dtype, "separate"),
        ),
    )
    _validate_replayssm_read_state_spec(spec)
    for index in (2, 3, 4):
        wrong_shapes = list(spec.shapes)
        wrong_shapes[index] = (
            wrong_shapes[index][0],
            ring_len - 1,
            *wrong_shapes[index][2:],
        )
        with pytest.raises(ValueError, match="Invalid quantized ReplaySSM"):
            _validate_replayssm_read_state_spec(
                replace(spec, shapes=tuple(wrong_shapes))
            )


@pytest.mark.parametrize(
    ("read_dtype", "mode"),
    [("fp8", "separate"), ("nvfp4", "separate")],
)
def test_replayssm_read_state_copy_spec_validates_payload_and_scales(read_dtype, mode):
    """Reject malformed cache records before copying payloads or scales."""
    spec = MambaSpec(
        block_size=16,
        shapes=(
            (3, 64),
            (16, 64, 128),
            (16, 16, 64),
            (16, 16),
            (1, 16, 128),
            *MambaStateShapeCalculator.replayssm_read_shapes(
                (16, 64, 128), read_dtype, mode, n_groups=1
            ),
        ),
        dtypes=(
            torch.bfloat16,
            MambaStateDtypeCalculator.replayssm_checkpoint_dtype(read_dtype, mode),
            torch.bfloat16,
            torch.float32,
            torch.bfloat16,
            *MambaStateDtypeCalculator.replayssm_read_dtypes(read_dtype, mode),
        ),
    )
    _validate_replayssm_read_state_spec(spec)
    for index in range(5, len(spec.shapes)):
        wrong_shapes = list(spec.shapes)
        wrong_shapes[index] = (*wrong_shapes[index][:-1], wrong_shapes[index][-1] + 1)
        with pytest.raises(ValueError, match="Invalid quantized ReplaySSM"):
            _validate_replayssm_read_state_spec(
                replace(spec, shapes=tuple(wrong_shapes))
            )
        wrong_dtypes = list(spec.dtypes)
        wrong_dtypes[index] = torch.float16
        with pytest.raises(ValueError, match="Invalid quantized ReplaySSM"):
            _validate_replayssm_read_state_spec(
                replace(spec, dtypes=tuple(wrong_dtypes))
            )
