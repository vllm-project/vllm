# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.layers.mamba.ops.ssu_dispatch as ssu_dispatch
from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    ReplaySSMModelContext,
    TritonSSUBackend,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    selective_state_update,
    selective_state_update_replayssm_flashinfer,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

try:
    from flashinfer.mamba.checkpointing_ssu import CheckpointingSSURunner
    from flashinfer.mamba.checkpointing_ssu import (
        checkpointing_ssu as checkpointing_ssu_kernel,
    )

    HAS_FLASHINFER_CHECKPOINTING_SSU = callable(CheckpointingSSURunner)
except ImportError:
    HAS_FLASHINFER_CHECKPOINTING_SSU = False


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


def test_replayssm_flashinfer_call_forwards_packed_mtp(monkeypatch):
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    kernel = Mock(return_value=torch.empty(1, 6, 2, 4))
    monkeypatch.setattr(mod, "_flashinfer_replayssm_kernel", kernel)

    tokens, nheads, dim, dstate, ngroups = 6, 2, 4, 8, 1
    state = torch.empty(2, nheads, dim, dstate)
    x = torch.empty(tokens, nheads, dim)
    dt = torch.empty_like(x)
    A = torch.empty(nheads, dim, dstate)
    B = torch.empty(tokens, ngroups, dstate)
    C = torch.empty_like(B)
    out = torch.empty_like(x)
    x_cache = torch.empty(2, nheads, 20, dim)
    dt_cache = torch.empty(2, nheads, 20)
    B_cache = torch.empty(2, ngroups, 20, dstate)
    ring_start = torch.zeros(2, dtype=torch.int32)
    prev_num_accepted = torch.zeros(2, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)

    selective_state_update_replayssm_flashinfer(
        state,
        x,
        dt,
        A,
        B,
        C,
        out,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_num_accepted,
        state_batch_indices=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
    )

    args = kernel.call_args.args
    kwargs = kernel.call_args.kwargs
    assert args[6].shape == (1, tokens, nheads, dim)
    assert args[7].shape == (1, tokens, nheads, dim)
    assert args[9].shape == (1, tokens, ngroups, dstate)
    assert args[10].shape == (1, tokens, ngroups, dstate)
    assert args[11].shape == (1, tokens, nheads, dim)
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert kwargs["max_seqlen"] == 4


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="compatible flashinfer checkpointing_ssu not available",
)
def test_replayssm_flashinfer_backend_init():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.FLASHINFER),
        _kv_cache_config_with_ssu(),
        use_replayssm=True,
    )
    assert isinstance(get_mamba_ssu_backend(), FlashInferSSUBackend)
    assert mod._flashinfer_replayssm_kernel is checkpointing_ssu_kernel


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
    base_shapes = ((64, 3), (8, 4, 16))

    shapes = MambaStateShapeCalculator.append_replayssm_ring(
        base_shapes,
        n_groups=4,
        tp_world_size=2,
        logical_window=16,
        backend=backend,
        num_speculative_tokens=num_speculative_tokens,
    )

    assert shapes[2:] == (
        (8, expected_ring_len, 4),
        (8, expected_ring_len),
        (2, expected_ring_len, 16),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        ((1 << 63) - 1, (1 << 63) - 1),
        (1 << 63, -(1 << 63)),
        ((1 << 64) - 1, -1),
    ],
)
def test_reinterpret_u64_as_i64(value: int, expected: int):
    assert ssu_dispatch._reinterpret_u64_as_i64(value) == expected


def _materialize_mixer(device: str = "cpu") -> Mock:
    mixer = Mock()
    mixer.kv_cache = [
        torch.empty(0, device=device),
        torch.empty(8, 4, 3, 5, device=device),
        torch.empty(8, 4, 20, 3, device=device),
        torch.empty(8, 4, 20, device=device),
        torch.empty(8, 2, 20, 5, device=device),
    ]
    mixer.A = torch.empty(4, 3, 5, device=device)
    mixer._replayssm_ring_start = torch.arange(8, dtype=torch.int32, device=device)
    mixer._replayssm_prev_num_accepted = torch.zeros(
        8, dtype=torch.int32, device=device
    )
    mixer.replayssm_buffer_len = 16
    mixer.mamba_config = SimpleNamespace(
        backend=MambaBackendEnum.FLASHINFER,
        enable_stochastic_rounding=True,
        stochastic_rounding_philox_rounds=6,
    )
    return mixer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_validate_replayssm_cache_rejects_incomplete_cache():
    mixer = _materialize_mixer(device="cuda")

    mixer.kv_cache[1] = torch.empty(0, device="cuda")
    with pytest.raises(RuntimeError, match="cache tensors"):
        ssu_dispatch._validate_replayssm_cache([mixer])

    mixer = _materialize_mixer(device="cuda")
    mixer.kv_cache[2] = torch.empty(0, device="cuda")
    with pytest.raises(RuntimeError, match="cache tensors"):
        ssu_dispatch._validate_replayssm_cache([mixer])

    mixer = _materialize_mixer(device="cuda")
    mixer._replayssm_ring_start = torch.empty(0, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="ring trackers"):
        ssu_dispatch._validate_replayssm_cache([mixer])


def test_validate_replayssm_cache_requires_cuda_state():
    mixer = _materialize_mixer(device="cpu")

    with pytest.raises(RuntimeError, match="requires CUDA cache tensors"):
        ssu_dispatch._validate_replayssm_cache([mixer])


def _modelwide_replayssm_fixture(cache_mode: str = "align"):
    groups = [
        [_materialize_mixer(device="cuda"), _materialize_mixer(device="cuda")],
        [_materialize_mixer(device="cuda"), _materialize_mixer(device="cuda")],
    ]
    layer_names: list[list[str]] = []
    forward_context = {}
    for group_idx, mixers in enumerate(groups):
        # Layers in one cache group share the physical tracker namespace.
        for mixer in mixers[1:]:
            mixer._replayssm_ring_start = mixers[0]._replayssm_ring_start
            mixer._replayssm_prev_num_accepted = mixers[0]._replayssm_prev_num_accepted
        names = [f"group{group_idx}.layer{layer_idx}" for layer_idx in range(2)]
        layer_names.append(names)
        for name, mixer in zip(names, mixers):
            mixer.use_replayssm = True
            forward_context[name] = mixer

    config = Mock()
    specs = [
        MambaSpec(
            block_size=1024 if cache_mode == "none" else 4,
            shapes=((4, 3, 5),),
            dtypes=(torch.float32,),
            mamba_cache_mode=cache_mode,
        )
        for _ in layer_names
    ]
    config.kv_cache_groups = [
        Mock(layer_names=names, kv_cache_spec=spec)
        for names, spec in zip(layer_names, specs)
    ]
    block_tables = [
        torch.tensor([[1, 2, 3], [0, 0, 0]], dtype=torch.int32, device="cuda"),
        torch.tensor([[4, 5, 6], [0, 0, 0]], dtype=torch.int32, device="cuda"),
    ]
    return groups, config, forward_context, block_tables


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_modelwide_replayssm_none_commits_trackers_without_materialization(
    monkeypatch,
):
    groups, config, forward_context, block_tables = _modelwide_replayssm_fixture(
        cache_mode="none"
    )
    materializer = Mock()
    monkeypatch.setattr(
        ssu_dispatch, "_load_replayssm_materialize", lambda: materializer
    )
    ctx = ReplaySSMModelContext.create(
        config,
        [0, 1],
        forward_context,
        block_tables,
        max_num_reqs=2,
    )
    assert ctx is not None
    assert not ctx.materialize_prefixes

    query_len = torch.zeros(2, dtype=torch.int32, device="cuda")
    num_computed = torch.zeros(2, dtype=torch.int32, device="cuda")
    accepted = torch.ones(2, dtype=torch.int32, device="cuda")
    is_prefilling = torch.zeros(2, dtype=torch.bool, device="cuda")
    live_cols = torch.zeros(2, dtype=torch.int32, device="cuda")
    no_materialize = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    materialize_counts = torch.zeros(2, dtype=torch.int32, device="cuda")

    def step(*, scheduled: int, num_accepted: int, prefilling: bool = False) -> None:
        query_len[0] = scheduled
        accepted[0] = num_accepted
        is_prefilling[0] = prefilling
        ctx.postprocess(
            idx_mapping=None,
            query_metadata=query_len,
            query_metadata_is_cumulative=False,
            num_computed_tokens=num_computed,
            num_computed_is_post_step=False,
            num_accepted_tokens=accepted,
            is_prefilling=is_prefilling,
            live_cols=live_cols,
            materialize_src_cols=no_materialize,
            materialize_dst_cols=no_materialize,
            materialize_token_counts=materialize_counts,
            mamba_block_size=1024,
            num_reqs=1,
        )
        num_computed[0] += num_accepted

    # Prefill canonicalizes the one live slot in mode none.
    for mixers, slot in zip(groups, (1, 4)):
        mixers[0]._replayssm_ring_start[slot] = 17
        mixers[0]._replayssm_prev_num_accepted[slot] = 9
    step(scheduled=8, num_accepted=1, prefilling=True)

    # Mix STP and MTP-shaped updates. Several scheduled lengths exceed the
    # accepted count; repeated crossings eventually wrap the 20-row ring.
    for scheduled, num_accepted in (
        (4, 2),
        (4, 1),
        (14, 3),
        (18, 2),
        (16, 16),
        (1, 1),
    ):
        step(scheduled=scheduled, num_accepted=num_accepted)
    torch.accelerator.synchronize()

    assert materializer.call_count == 0
    assert ctx.plan_flush_count.tolist() == [-1, -1]
    for mixers, live_slot in zip(groups, (1, 4)):
        # Both layers share this group tracker. The expected single transition
        # per step would differ if either layer committed it independently.
        assert mixers[0]._replayssm_ring_start[live_slot].item() == 4
        assert mixers[0]._replayssm_prev_num_accepted[live_slot].item() == 1

    # A later prefill resets the exact same live physical slot again.
    step(scheduled=8, num_accepted=1, prefilling=True)
    torch.accelerator.synchronize()
    for mixers, live_slot in zip(groups, (1, 4)):
        assert mixers[0]._replayssm_ring_start[live_slot].item() == 0
        assert mixers[0]._replayssm_prev_num_accepted[live_slot].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_modelwide_replayssm_postprocess_launches_materializer_once(monkeypatch):
    groups, config, forward_context, block_tables = _modelwide_replayssm_fixture()
    for mixers, source_slot in zip(groups, (1, 4)):
        mixers[0]._replayssm_ring_start[source_slot] = 2
        mixers[0]._replayssm_prev_num_accepted[source_slot] = 4
    kernel = Mock()
    monkeypatch.setattr(ssu_dispatch, "_load_replayssm_materialize", lambda: kernel)

    ctx = ReplaySSMModelContext.create(
        config,
        [0, 1],
        forward_context,
        block_tables,
        max_num_reqs=2,
    )
    assert ctx is not None
    ctx.postprocess(
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
        query_metadata=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        query_metadata_is_cumulative=True,
        num_computed_tokens=torch.tensor([4, 0], dtype=torch.int32, device="cuda"),
        num_computed_is_post_step=True,
        num_accepted_tokens=torch.tensor([2, 1], dtype=torch.int32, device="cuda"),
        is_prefilling=torch.tensor([False, False], device="cuda"),
        live_cols=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
        materialize_src_cols=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
        materialize_dst_cols=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
        materialize_token_counts=torch.tensor([2, 0], dtype=torch.int32, device="cuda"),
        mamba_block_size=4,
        num_reqs=1,
    )
    torch.accelerator.synchronize()

    assert kernel.call_count == 1
    args = kernel.call_args.args
    kwargs = kernel.call_args.kwargs
    assert args[11] is ctx.src_slots
    assert args[12] is ctx.dst_slots
    assert args[13] is ctx.plan_ring_start
    assert args[14] is ctx.plan_flush_count
    assert kwargs["num_heads"] == 4
    assert kwargs["heads_per_group"] == 2
    assert kwargs["max_window"] == 16
    assert kwargs["ring_buffer_len"] == 20
    assert ctx.src_slots[:, 0].tolist() == [1, 1, 4, 4]
    assert ctx.dst_slots[:, 0].tolist() == [2, 2, 5, 5]
    assert ctx.src_slots[:, 1].tolist() == [NULL_BLOCK_ID] * 4
    assert ctx.dst_slots[:, 1].tolist() == [NULL_BLOCK_ID] * 4
    assert ctx.plan_ring_start.tolist() == [2, 0]
    assert ctx.plan_flush_count.tolist() == [6, -1]
    assert groups[0][0]._replayssm_prev_num_accepted[1].item() == 6
    assert groups[0][0]._replayssm_prev_num_accepted[2].item() == 0
    assert groups[1][0]._replayssm_prev_num_accepted[4].item() == 6
    assert groups[1][0]._replayssm_prev_num_accepted[5].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_modelwide_replayssm_postprocess_commits_checkpoint_boundary(monkeypatch):
    groups, config, forward_context, block_tables = _modelwide_replayssm_fixture()
    for mixers, source_slot in zip(groups, (1, 4)):
        mixers[0]._replayssm_ring_start[source_slot] = 2
        mixers[0]._replayssm_prev_num_accepted[source_slot] = 13
    kernel = Mock()
    monkeypatch.setattr(ssu_dispatch, "_load_replayssm_materialize", lambda: kernel)

    ctx = ReplaySSMModelContext.create(
        config,
        [0, 1],
        forward_context,
        block_tables,
        max_num_reqs=2,
    )
    assert ctx is not None
    ctx.postprocess(
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
        query_metadata=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        query_metadata_is_cumulative=True,
        num_computed_tokens=torch.tensor([4, 0], dtype=torch.int32, device="cuda"),
        num_computed_is_post_step=True,
        num_accepted_tokens=torch.tensor([3, 1], dtype=torch.int32, device="cuda"),
        is_prefilling=torch.tensor([False, False], device="cuda"),
        live_cols=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
        materialize_src_cols=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
        materialize_dst_cols=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
        materialize_token_counts=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
        mamba_block_size=4,
        num_reqs=1,
    )
    torch.accelerator.synchronize()

    assert kernel.call_count == 1
    assert ctx.plan_ring_start.tolist() == [15, 0]
    assert ctx.plan_flush_count.tolist() == [1, -1]
    for mixers, source_slot, destination_slot in zip(groups, (1, 4), (2, 5)):
        assert mixers[0]._replayssm_ring_start[source_slot].item() == 15
        assert mixers[0]._replayssm_prev_num_accepted[source_slot].item() == 3
        assert mixers[0]._replayssm_ring_start[destination_slot].item() == 0
        assert mixers[0]._replayssm_prev_num_accepted[destination_slot].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_modelwide_replayssm_postprocess_resets_prefill_slot(monkeypatch):
    groups, config, forward_context, block_tables = _modelwide_replayssm_fixture()
    for mixers, source_slots in zip(groups, ((1, 2), (4, 5))):
        for source_slot in source_slots:
            mixers[0]._replayssm_ring_start[source_slot] = 7
            mixers[0]._replayssm_prev_num_accepted[source_slot] = 9
    kernel = Mock()
    monkeypatch.setattr(ssu_dispatch, "_load_replayssm_materialize", lambda: kernel)

    ctx = ReplaySSMModelContext.create(
        config,
        [0, 1],
        forward_context,
        block_tables,
        max_num_reqs=2,
    )
    assert ctx is not None
    ctx.postprocess(
        idx_mapping=None,
        query_metadata=torch.tensor([8, 0], dtype=torch.int32, device="cuda"),
        query_metadata_is_cumulative=False,
        num_computed_tokens=torch.zeros(2, dtype=torch.int32, device="cuda"),
        num_computed_is_post_step=False,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="cuda"),
        is_prefilling=torch.tensor([True, False], device="cuda"),
        live_cols=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
        materialize_src_cols=torch.tensor([-1, -1], dtype=torch.int32, device="cuda"),
        materialize_dst_cols=torch.zeros(2, dtype=torch.int32, device="cuda"),
        materialize_token_counts=torch.zeros(2, dtype=torch.int32, device="cuda"),
        mamba_block_size=4,
        num_reqs=1,
    )
    torch.accelerator.synchronize()

    assert kernel.call_count == 1
    assert ctx.plan_flush_count.tolist() == [-1, -1]
    for mixers, source_slots in zip(groups, ((1, 2), (4, 5))):
        for source_slot in source_slots:
            assert mixers[0]._replayssm_ring_start[source_slot].item() == 0
            assert mixers[0]._replayssm_prev_num_accepted[source_slot].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_modelwide_replayssm_copies_reassigned_live_slot_once(monkeypatch):
    groups, config, forward_context, block_tables = _modelwide_replayssm_fixture()
    for mixers, source_slot in zip(groups, (1, 4)):
        mixers[0]._replayssm_ring_start[source_slot] = 2
        mixers[0]._replayssm_prev_num_accepted[source_slot] = 4
    kernel = Mock()
    monkeypatch.setattr(ssu_dispatch, "_load_replayssm_materialize", lambda: kernel)

    ctx = ReplaySSMModelContext.create(
        config,
        [0, 1],
        forward_context,
        block_tables,
        max_num_reqs=2,
    )
    assert ctx is not None
    ctx.materialize_reassigned_slots(
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
        src_cols=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
        dst_cols=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
        num_reqs=1,
    )
    torch.accelerator.synchronize()

    assert kernel.call_count == 1
    assert ctx.precopy_ring_start.tolist() == [2, 0]
    assert ctx.precopy_flush_count.tolist() == [4, -1]
    assert ctx.precopy_src_slots[:, 0].tolist() == [1, 1, 4, 4]
    assert ctx.precopy_dst_slots[:, 0].tolist() == [2, 2, 5, 5]
    for mixers, source_slot, destination_slot in zip(groups, (1, 4), (2, 5)):
        assert mixers[0]._replayssm_ring_start[source_slot].item() == 2
        assert mixers[0]._replayssm_prev_num_accepted[source_slot].item() == 4
        assert mixers[0]._replayssm_ring_start[destination_slot].item() == 0
        assert mixers[0]._replayssm_prev_num_accepted[destination_slot].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_modelwide_replayssm_copies_only_reassigned_cache_group(monkeypatch):
    groups, config, forward_context, block_tables = _modelwide_replayssm_fixture()
    block_tables[0][0, 1] = block_tables[0][0, 0]
    for mixers, source_slot in zip(groups, (1, 4)):
        mixers[0]._replayssm_ring_start[source_slot] = 2
        mixers[0]._replayssm_prev_num_accepted[source_slot] = 4
    kernel = Mock()
    monkeypatch.setattr(ssu_dispatch, "_load_replayssm_materialize", lambda: kernel)

    ctx = ReplaySSMModelContext.create(
        config,
        [0, 1],
        forward_context,
        block_tables,
        max_num_reqs=2,
    )
    assert ctx is not None
    ctx.materialize_reassigned_slots(
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
        src_cols=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
        dst_cols=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
        num_reqs=1,
    )
    torch.accelerator.synchronize()

    assert kernel.call_count == 1
    assert ctx.precopy_ring_start.tolist() == [2, 0]
    assert ctx.precopy_flush_count.tolist() == [4, -1]
    assert ctx.precopy_src_slots[:, 0].tolist() == [
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        4,
        4,
    ]
    assert ctx.precopy_dst_slots[:, 0].tolist() == [
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        5,
        5,
    ]
    assert groups[0][0]._replayssm_ring_start[1].item() == 2
    assert groups[0][0]._replayssm_prev_num_accepted[1].item() == 4
    assert groups[1][0]._replayssm_ring_start[4].item() == 2
    assert groups[1][0]._replayssm_prev_num_accepted[4].item() == 4
    assert groups[1][0]._replayssm_ring_start[5].item() == 0
    assert groups[1][0]._replayssm_prev_num_accepted[5].item() == 0
