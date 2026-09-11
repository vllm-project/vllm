# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    TritonSSUBackend,
    _postprocess_replayssm_kernel,
    _ReplaySSMGroupContext,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    selective_state_update,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("post_step", [False, True], ids=["v1", "v2"])
def test_replayssm_materializes_only_accepted_boundary_with_compacted_rows(post_step):
    """Publish a boundary without modifying its live or cached-prefix sources."""
    from flashinfer.mamba.replayssm_materialize import replayssm_materialize

    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    set_random_seed(42)
    slots, heads, dim, dstate, ring_len = 6, 8, 64, 128, 20
    state = torch.randn(slots, heads, dim, dstate, device="cuda") * 0.1
    # Slot 1 is an immutable cached prefix, copied to private live slot 3.
    state[3].copy_(state[1])
    original = state.clone()
    x = torch.randn(slots, heads, ring_len, dim, device="cuda", dtype=torch.bfloat16)
    dt = torch.full((slots, heads, ring_len), 0.01, device="cuda")
    B = torch.randn(slots, 1, ring_len, dstate, device="cuda", dtype=torch.bfloat16)
    A = -torch.ones(heads, device="cuda")
    mixer = SimpleNamespace(
        kv_cache=(torch.empty(0, device="cuda"), state),
        replayssm_cache=(x, dt, B),
        _replayssm_ring_start=ints([0, 0, 0, 7, 0, 2]),
        _replayssm_prev_num_accepted=ints([0, 0, 0, 2, 0, 1]),
        replayssm_buffer_len=16,
        A=A,
        mamba_config=MambaConfig(backend=MambaBackendEnum.FLASHINFER),
    )
    group = _ReplaySSMGroupContext.create(
        [mixer],
        ints(
            [
                [2, 3],
                [NULL_BLOCK_ID, 5],
                [NULL_BLOCK_ID, NULL_BLOCK_ID],
                [NULL_BLOCK_ID, NULL_BLOCK_ID],
            ]
        ),
        "all",
        16,
        4,
    )
    # Batch row 0 crosses token 16; row 1 rejects all drafts and stops at 15.
    # Rows 2 and 3 exercise padded physical slots and padded request indices.
    mapping = ints([2, 0, 1, -1])
    group.postprocess(
        idx_mapping=mapping,
        query_metadata=ints([0, 4, 8, 12, 16]) if post_step else ints([4] * 4),
        query_metadata_is_cumulative=post_step,
        num_computed_tokens=ints([15, 17, 17]) if post_step else ints([14] * 3),
        num_computed_is_post_step=post_step,
        num_accepted_tokens=ints([1, 3, 3]),
        is_prefilling=torch.zeros(4, dtype=torch.bool, device="cuda"),
        live_cols=ints([1, 1, 1]),
        num_reqs=4,
    )
    assert group.active_request_indices.tolist() == [0, -1, -1, -1]
    assert group.plan_flush_count.tolist() == [4, -1, -1, -1]
    group.materialize(replayssm_materialize)

    expected = original[3].clone()
    for offset in range(4):
        pos = (7 + offset) % ring_len
        delta = dt[3, :, pos]
        expected *= torch.exp(delta * A)[:, None, None]
        expected += (
            delta[:, None, None]
            * x[3, :, pos].float()[:, :, None]
            * B[3, 0, pos].float()[None, None, :]
        )
    torch.testing.assert_close(state[2], expected, atol=2e-3, rtol=2e-3)
    # The snapshot excludes the accepted tail past 16 and all rejected drafts.
    for slot in (0, 1, 3, 4, 5):
        torch.testing.assert_close(state[slot], original[slot], atol=0, rtol=0)
    assert mixer._replayssm_prev_num_accepted[3].item() == 5
    assert mixer._replayssm_prev_num_accepted[2].item() == 0


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
    shapes = MambaStateShapeCalculator.replayssm_ring_shapes(
        num_heads=16,
        head_dim=4,
        state_size=16,
        n_groups=4,
        tp_world_size=2,
        logical_window=16,
        backend=backend,
        num_speculative_tokens=num_speculative_tokens,
    )

    assert shapes == (
        (8, expected_ring_len, 4),
        (8, expected_ring_len),
        (2, expected_ring_len, 16),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("post_step", [False, True])
@pytest.mark.parametrize(
    ("computed_before", "query_len", "prefilling", "accepted", "expected"),
    [
        (256, 4, False, 1, 3),  # Padded cached prompt tail commits its real token.
        (1, 4, False, 1, 3),  # Rejected placeholders exceed the cached prefix.
        (256, 1, False, 1, 3),
        (0, 4, True, 1, 0),  # Initial prefill clears stale trackers.
        (256, 4, True, 1, 0),  # Multi-token prefill also clears trackers.
        (256, 4, False, 3, 5),  # Ordinary speculative decode commits acceptance.
    ],
)
def test_replayssm_postprocess_commits_staged_transition(
    post_step, computed_before, query_len, prefilling, accepted, expected
):
    """The staged kernel path must preserve accepted history on prompt tails."""

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    computed = computed_before
    if post_step:
        computed += query_len if prefilling else accepted
    ring_start = tensor([3])
    committed = tensor([2])
    plan_start, plan_flush = tensor([0]), tensor([-1])
    slots = tensor([[0]])
    _postprocess_replayssm_kernel[(1,)](
        tensor([0]),
        tensor([0, query_len]) if post_step else tensor([query_len]),
        tensor([computed]),
        tensor([accepted]),
        torch.tensor([prefilling], device="cuda"),
        None,
        tensor([[0, 0]]),
        ring_start,
        committed,
        slots,
        slots,
        plan_start,
        plan_flush,
        2,
        1,
        MAMBA_BLOCK_SIZE=256,
        LOGICAL_WINDOW=16,
        RING_BUFFER_LEN=20,
        NUM_LAYERS=1,
        PAD_SLOT_ID=-1,
        QUERY_METADATA_IS_CUMULATIVE=post_step,
        NUM_COMPUTED_IS_POST_STEP=post_step,
        HAS_IDX_MAPPING=post_step,
        MATERIALIZE_PREFIXES=False,
        LIVE_COL_IS_ZERO=True,
    )
    assert committed.item() == expected
    assert ring_start.item() == (0 if prefilling else 3)
