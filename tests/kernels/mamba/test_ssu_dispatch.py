# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    TritonSSUBackend,
    commit_replayssm_ring_trackers,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    reset_replayssm_ring_trackers,
    selective_state_update,
    selective_state_update_replayssm_flashinfer,
    update_replayssm_ring_trackers,
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

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

try:
    from flashinfer.mamba.checkpointing_ssu import CheckpointingSSURunner

    HAS_FLASHINFER_CHECKPOINTING_SSU = CheckpointingSSURunner is not None
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


def test_flashinfer_replayssm_ring_tracker_lifecycle():
    ring_start = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_num_accepted = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_query_len = torch.zeros(2, dtype=torch.int32, device="cuda")
    state_batch_indices = torch.tensor([1], dtype=torch.int32, device="cuda")

    observed = []
    for _ in range(33):
        update_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted,
            prev_query_len,
            state_batch_indices,
            logical_window=16,
            ring_buffer_len=17,
        )
        observed.append((int(ring_start[1]), int(prev_num_accepted[1])))

    assert observed[4] == (0, 5)
    assert observed[15] == (0, 16)
    assert observed[16] == (16, 1)
    assert observed[31] == (16, 16)
    assert observed[32] == (15, 1)

    reset_replayssm_ring_trackers(
        ring_start,
        prev_num_accepted,
        prev_query_len,
        state_batch_indices,
    )
    assert (
        ring_start[1].item(),
        prev_num_accepted[1].item(),
        prev_query_len[1].item(),
    ) == (0, 0, 0)


@pytest.mark.parametrize(
    ("accepted_sequence", "expected"),
    [
        pytest.param(
            [4] * 22,
            [
                (0, 0, 4),
                (0, 4, 4),
                (0, 8, 4),
                (0, 12, 4),
                (0, 16, 4),
                (16, 4, 4),
                (16, 8, 4),
                (16, 12, 4),
                (16, 16, 4),
                (12, 4, 4),
                (12, 8, 4),
                (12, 12, 4),
                (12, 16, 4),
                (8, 4, 4),
                (8, 8, 4),
                (8, 12, 4),
                (8, 16, 4),
                (4, 4, 4),
                (4, 8, 4),
                (4, 12, 4),
                (4, 16, 4),
                (0, 4, 4),
            ],
            id="all-accepted",
        ),
        pytest.param(
            [4, 4, 0, 3, 4, 2, 4, 1],
            [
                (0, 0, 4),
                (0, 4, 4),
                (0, 4, 4),
                (0, 7, 4),
                (0, 11, 4),
                (0, 13, 4),
                (13, 4, 4),
                (13, 5, 4),
            ],
            id="mixed",
        ),
    ],
)
def test_replayssm_commit_tracker_acceptance_sequence(accepted_sequence, expected):
    logical_window = 16
    num_speculative_tokens = 3
    query_len = 1 + num_speculative_tokens
    ring_buffer_len = logical_window + 1 + num_speculative_tokens
    ring_start = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_num_accepted = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_query_len = torch.zeros(2, dtype=torch.int32, device="cuda")
    state_batch_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device="cuda")

    observed = []
    for accepted in accepted_sequence:
        commit_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted,
            prev_query_len,
            state_batch_indices,
            torch.tensor([accepted], dtype=torch.int32, device="cuda"),
            query_start_loc,
            logical_window,
            ring_buffer_len,
        )
        snapshot = (
            ring_start[1].item(),
            prev_num_accepted[1].item(),
            prev_query_len[1].item(),
        )
        observed.append(snapshot)
        assert snapshot[1] + snapshot[2] <= ring_buffer_len

    assert observed == expected


def test_replayssm_resume_resets_commit_history():
    ring_start = torch.tensor([0, 13], dtype=torch.int32, device="cuda")
    prev_num_accepted = torch.tensor([0, 13], dtype=torch.int32, device="cuda")
    prev_query_len = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    state_batch_indices = torch.tensor([1], dtype=torch.int32, device="cuda")

    reset_replayssm_ring_trackers(
        ring_start,
        prev_num_accepted,
        prev_query_len,
        state_batch_indices,
    )
    assert (
        ring_start[1].item(),
        prev_num_accepted[1].item(),
        prev_query_len[1].item(),
    ) == (0, 0, 0)

    commit_replayssm_ring_trackers(
        ring_start,
        prev_num_accepted,
        prev_query_len,
        state_batch_indices,
        torch.tensor([3], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        logical_window=16,
        ring_buffer_len=20,
    )
    assert (
        ring_start[1].item(),
        prev_num_accepted[1].item(),
        prev_query_len[1].item(),
    ) == (0, 0, 4)


def test_replayssm_commit_tracker_ragged_query_lengths():
    ring_start = torch.zeros(3, dtype=torch.int32, device="cuda")
    prev_num_accepted = torch.zeros(3, dtype=torch.int32, device="cuda")
    prev_query_len = torch.zeros(3, dtype=torch.int32, device="cuda")
    state_batch_indices = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    query_start_loc = torch.tensor([0, 4, 6], dtype=torch.int32, device="cuda")

    observed = []
    for accepted in ([4, 2], [3, 1]):
        commit_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted,
            prev_query_len,
            state_batch_indices,
            torch.tensor(accepted, dtype=torch.int32, device="cuda"),
            query_start_loc,
            logical_window=16,
            ring_buffer_len=20,
        )
        observed.append(
            [
                (
                    ring_start[slot].item(),
                    prev_num_accepted[slot].item(),
                    prev_query_len[slot].item(),
                )
                for slot in (1, 2)
            ]
        )

    assert observed == [[(0, 0, 4), (0, 0, 2)], [(0, 3, 4), (0, 1, 2)]]


@pytest.mark.parametrize("operation", ["commit", "reset"])
def test_replayssm_tracker_kernels_mask_invalid_slots(operation):
    num_states = 3
    ring_start = torch.tensor([11, 2, 33], dtype=torch.int32, device="cuda")
    prev_num_accepted = torch.tensor([11, 3, 33], dtype=torch.int32, device="cuda")
    prev_query_len = torch.tensor([11, 4, 33], dtype=torch.int32, device="cuda")
    state_batch_indices = torch.tensor(
        [-1, num_states, NULL_BLOCK_ID, 1], dtype=torch.int32, device="cuda"
    )

    if operation == "commit":
        commit_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted,
            prev_query_len,
            state_batch_indices,
            torch.tensor([4, 4, 4, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device="cuda"),
            logical_window=16,
            ring_buffer_len=20,
        )
        expected_valid = (2, 5, 4)
    else:
        reset_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted,
            prev_query_len,
            state_batch_indices,
        )
        expected_valid = (0, 0, 0)

    assert (
        ring_start.tolist(),
        prev_num_accepted.tolist(),
        prev_query_len.tolist(),
    ) == (
        [11, expected_valid[0], 33],
        [11, expected_valid[1], 33],
        [11, expected_valid[2], 33],
    )


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


@pytest.mark.parametrize("layout", ["packed", "dense"])
def test_replayssm_flashinfer_call_forwards_mtp_layout(monkeypatch, layout):
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    kernel = Mock(return_value=torch.empty(0))
    monkeypatch.setattr(mod, "_flashinfer_replayssm_kernel", kernel)

    batch, max_seqlen, nheads, dim, dstate, ngroups = 2, 4, 2, 4, 8, 1
    state = torch.empty(2, nheads, dim, dstate)
    x_shape: tuple[int, ...]
    B_shape: tuple[int, ...]
    expected_x_shape: tuple[int, ...]
    expected_B_shape: tuple[int, ...]
    if layout == "packed":
        x_shape = (6, nheads, dim)
        B_shape = (6, ngroups, dstate)
        expected_x_shape = (1, 6, nheads, dim)
        expected_B_shape = (1, 6, ngroups, dstate)
        cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)
        kernel_max_seqlen = max_seqlen
    else:
        x_shape = (batch, max_seqlen, nheads, dim)
        B_shape = (batch, max_seqlen, ngroups, dstate)
        expected_x_shape = x_shape
        expected_B_shape = B_shape
        cu_seqlens = None
        kernel_max_seqlen = None
    x = torch.empty(x_shape)
    dt = torch.empty_like(x)
    A = torch.empty(nheads, dim, dstate)
    B = torch.empty(B_shape)
    C = torch.empty_like(B)
    out = torch.empty_like(x)
    x_cache = torch.empty(2, nheads, 20, dim)
    dt_cache = torch.empty(2, nheads, 20)
    B_cache = torch.empty(2, ngroups, 20, dstate)
    ring_start = torch.zeros(2, dtype=torch.int32)
    prev_num_accepted = torch.zeros(2, dtype=torch.int32)
    prev_query_len = torch.zeros(2, dtype=torch.int32)
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
        prev_query_len,
        logical_window=16,
        state_batch_indices=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens=cu_seqlens,
        max_seqlen=kernel_max_seqlen,
        update_trackers=False,
    )

    args = kernel.call_args.args
    assert args[6].shape == expected_x_shape
    assert args[7].shape == expected_x_shape
    assert args[9].shape == expected_B_shape
    assert args[10].shape == expected_B_shape
    assert args[11].shape == expected_x_shape
    assert kernel.call_args.kwargs["cu_seqlens"] is cu_seqlens
    assert kernel.call_args.kwargs["max_seqlen"] == kernel_max_seqlen


@pytest.mark.parametrize(
    ("query_start_loc", "expected_shape", "expected_max_seqlen"),
    [
        pytest.param([0, 4, 8], (2, 4, 2, 4), None, id="dense"),
        pytest.param([0, 4, 6], (6, 2, 4), 4, id="packed"),
    ],
)
def test_replayssm_mixer_selects_mtp_layout(
    monkeypatch, query_start_loc, expected_shape, expected_max_seqlen
):
    import vllm.model_executor.layers.mamba.mamba_mixer2 as mod

    mixer = MambaMixer2.__new__(MambaMixer2)
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
    mixer.replayssm_buffer_len = 16
    mixer._commits_replayssm_trackers = True
    mixer._updates_replayssm_trackers = False
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
    mixer._replayssm_prev_query_len = torch.zeros(3, dtype=torch.int32)
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
        num_computed_tokens_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=torch.tensor([[1], [2]], dtype=torch.int32),
        query_start_loc_d=query_start_loc_d,
        num_accepted_tokens=torch.tensor([4, 2], dtype=torch.int32),
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
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
    monkeypatch.setattr(mod, "commit_replayssm_ring_trackers", Mock())
    monkeypatch.setattr(
        mod, "causal_conv1d_update", lambda values, *args, **kwargs: values
    )
    monkeypatch.setattr(mod, "selective_state_update_replayssm_flashinfer", kernel)

    mixer.conv_ssm_forward(
        torch.empty(num_decode_tokens, 3), torch.empty(num_decode_tokens, 8)
    )

    assert kernel.call_args.args[1].shape == expected_shape
    if expected_max_seqlen is None:
        assert kernel.call_args.kwargs["cu_seqlens"] is None
    else:
        assert kernel.call_args.kwargs["cu_seqlens"] is query_start_loc_d
    assert kernel.call_args.kwargs["max_seqlen"] == expected_max_seqlen


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
