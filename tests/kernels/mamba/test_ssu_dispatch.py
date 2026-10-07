# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
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
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.utils import allocate_replayssm_caches

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


@pytest.mark.parametrize("post_step", [False, True], ids=["v1", "v2"])
def test_replayssm_materializes_only_accepted_boundary_with_compacted_rows(post_step):
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
    slots, heads, dim, dstate, ring_len = 6, 8, 64, 128, 20
    state = torch.randn(slots, heads, dim, dstate, device="cuda") * 0.1
    # Slot 1 is an immutable cached prefix, copied to private live slot 3.
    state[3].copy_(state[1])
    original = state.clone()
    spec = MambaSpec(
        block_size=16,
        shapes=((heads, dim, dstate),),
        dtypes=(torch.float32,),
        replayssm_shapes=(
            (heads, ring_len, dim),
            (heads, ring_len),
            (1, ring_len, dstate),
        ),
        replayssm_dtypes=(torch.bfloat16, torch.float32, torch.bfloat16),
    )
    config = KVCacheConfig(
        num_blocks=slots,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["mixer", "neighbor"], spec),
            KVCacheGroupSpec(["other_group"], spec),
        ],
    )
    rings = allocate_replayssm_caches(config, torch.device("cuda"))
    x, dt, B = rings["mixer"]
    x.normal_()
    dt.fill_(0.01)
    B.normal_()
    assert not x.is_contiguous() and not dt.is_contiguous() and not B.is_contiguous()
    assert x.data_ptr() == rings["other_group"][0].data_ptr()
    # The other layer is disjoint within this group. Group B owns physical4,
    # while materialization only uses group A's live3 and destination2.
    for tensor in rings["neighbor"]:
        tensor.fill_(7)
    neighbor_before = [tensor.clone() for tensor in rings["neighbor"]]
    for tensor in rings["other_group"]:
        tensor[4].fill_(9)
    other_owned_before = [tensor[4].clone() for tensor in rings["other_group"]]
    A = -torch.ones(heads, device="cuda")
    mixer = SimpleNamespace(
        kv_cache=(torch.empty(0, device="cuda"), state),
        replayssm_cache=(x, dt, B),
        _replayssm_ring_start=_cuda_ints([0, 0, 0, 7, 0, 2]),
        _replayssm_prev_num_accepted=_cuda_ints([0, 0, 0, 2, 0, 1]),
        replayssm_buffer_len=16,
        A=A,
        mamba_config=MambaConfig(backend=MambaBackendEnum.FLASHINFER),
    )
    group = _ReplaySSMGroupContext.create(
        [mixer],
        _cuda_ints(
            [
                [2, 3],
                [NULL_BLOCK_ID, 5],
                [NULL_BLOCK_ID, NULL_BLOCK_ID],
                [NULL_BLOCK_ID, NULL_BLOCK_ID],
            ]
        ),
        "align",
        16,
        4,
    )
    # Batch row 0 crosses token 16; row 1 rejects all drafts and stops at 15.
    # Rows 2 and 3 exercise padded physical slots and padded request indices.
    mapping = _cuda_ints([2, 0, 1, -1])
    group.postprocess(
        idx_mapping=mapping,
        query_metadata=_cuda_ints([0, 4, 8, 12, 16])
        if post_step
        else _cuda_ints([4] * 4),
        query_metadata_is_cumulative=post_step,
        num_computed_tokens=_cuda_ints([15, 17, 17])
        if post_step
        else _cuda_ints([14] * 3),
        num_computed_is_post_step=post_step,
        num_accepted_tokens=_cuda_ints([1, 3, 3]),
        is_prefilling=torch.zeros(4, dtype=torch.bool, device="cuda"),
        live_cols=_cuda_ints([1, 1, 1]),
        num_reqs=4,
    )
    assert group.active_request_indices.tolist() == [0, -1, -1, -1]
    assert group.plan_flush_count.tolist() == [4, -1, -1, -1]
    group.materialize()

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
    for tensor, before in zip(rings["neighbor"], neighbor_before, strict=True):
        torch.testing.assert_close(tensor, before, atol=0, rtol=0)
    for tensor, before in zip(rings["other_group"], other_owned_before, strict=True):
        torch.testing.assert_close(tensor[4], before, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("post_step", [False, True], ids=["v1", "v2"])
def test_replayssm_skips_canonical_self_copy_but_resets_prefill_trackers(post_step):
    """Skip prefill self-copies, reset trackers, and preserve in-place MTP replay."""
    mixer = SimpleNamespace(
        kv_cache=(
            torch.empty(0, device="cuda"),
            torch.empty(6, 1, 1, 1, device="cuda"),
        ),
        replayssm_cache=(
            torch.empty(6, 1, 20, 1, device="cuda"),
            torch.empty(6, 1, 20, device="cuda"),
            torch.empty(6, 1, 20, 1, device="cuda"),
        ),
        _replayssm_ring_start=_cuda_ints([3] * 6),
        _replayssm_prev_num_accepted=_cuda_ints([2] * 6),
        replayssm_buffer_len=16,
        A=torch.empty(1, device="cuda"),
    )
    group = _ReplaySSMGroupContext.create(
        [mixer],
        _cuda_ints(
            [
                [0, 1, 2],
                [3, 4, 5],
                [NULL_BLOCK_ID] * 3,
                [NULL_BLOCK_ID] * 3,
            ]
        ),
        "align",
        16,
        4,
    )
    # Reordered rows: canonical prefill in slot 2; in-place MTP replay in slot 4;
    # padded slot; padded request.
    group.postprocess(
        idx_mapping=_cuda_ints([1, 2, 0, -1]),
        query_metadata=(
            _cuda_ints([0, 48, 52, 68, 68]) if post_step else _cuda_ints([48, 4, 16, 0])
        ),
        query_metadata_is_cumulative=post_step,
        num_computed_tokens=_cuda_ints([16, 48, 32])
        if post_step
        else _cuda_ints([0, 0, 28]),
        num_computed_is_post_step=post_step,
        num_accepted_tokens=_cuda_ints([1, 1, 4]),
        is_prefilling=torch.tensor([True, False, True, True], device="cuda"),
        live_cols=_cuda_ints([0, 2, 1]),
        num_reqs=4,
    )

    assert group.plan_flush_count.tolist() == [-1, 6, -1, -1]
    assert group.active_request_indices.tolist() == [1, -1, -1, -1]
    assert group.src_slots[0, 1].item() == 4
    assert group.dst_slots[0, 1].item() == 4
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
    (
        "computed_before",
        "query_len",
        "prefilling",
        "accepted",
        "old_committed",
        "old_start",
        "expected_committed",
        "expected_start",
    ),
    [
        (256, 4, False, 1, 2, 3, 3, 3),  # Padded prompt tail.
        (256, 4, False, 3, 2, 3, 5, 3),  # Speculative decode.
        (0, 50_000, True, 1, 2, 3, 0, 0),  # Long initial prefill.
        (50_000, 4, True, 1, 2, 3, 0, 0),  # Later prefill chunk.
        (256, 4, False, 1, 15, 7, 1, 2),  # Checkpoint wraps the physical ring.
    ],
)
def test_replayssm_postprocess_commits_staged_transition(
    post_step,
    computed_before,
    query_len,
    prefilling,
    accepted,
    old_committed,
    old_start,
    expected_committed,
    expected_start,
):
    """Mode none updates only its live slot, including for long prompt chunks."""
    computed = computed_before
    if post_step:
        computed += query_len if prefilling else accepted
    ring_start = _cuda_ints([old_start, 7])
    committed = _cuda_ints([old_committed, 9])
    # Neighboring request rows provide valid canary IDs, making erroneous
    # column reads deterministically corrupt another slot rather than crash.
    block_table = torch.ones((256, 1), dtype=torch.int32, device="cuda")
    block_table[0, 0] = 0
    plan_start, plan_flush = _cuda_ints([0]), _cuda_ints([-1])
    slots = _cuda_ints([[0]])
    _postprocess_replayssm_kernel[(1,)](
        _cuda_ints([0]),
        _cuda_ints([0, query_len]) if post_step else _cuda_ints([query_len]),
        _cuda_ints([computed]),
        _cuda_ints([accepted]),
        torch.tensor([prefilling], device="cuda"),
        None,
        block_table[:1],
        ring_start,
        committed,
        slots,
        slots,
        plan_start,
        plan_flush,
        block_table.stride(0),
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
    assert committed.tolist() == [expected_committed, 9]
    assert ring_start.tolist() == [expected_start, 7]


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
    )
    mixer.replayssm_cache = (
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
