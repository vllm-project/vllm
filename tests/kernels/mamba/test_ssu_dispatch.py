# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from unittest.mock import Mock

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    TritonSSUBackend,
    commit_replayssm_ring_trackers,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    materialize_replayssm_prefix_gpu,
    materialize_replayssm_prefix_mtp_gpu,
    reset_replayssm_ring_trackers,
    selective_state_update,
    selective_state_update_replayssm_flashinfer,
    update_replayssm_ring_trackers,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

try:
    from flashinfer.mamba.checkpointing_ssu import (
        CheckpointingSSURunner,
        allocate_checkpointing_ssu_scratch,
    )
    from flashinfer.mamba.checkpointing_ssu import (
        checkpointing_ssu as checkpointing_ssu_kernel,
    )

    HAS_FLASHINFER_CHECKPOINTING_SSU = all(
        callable(symbol)
        for symbol in (CheckpointingSSURunner, allocate_checkpointing_ssu_scratch)
    )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
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
    assert (ring_start[1].item(), prev_num_accepted[1].item()) == (0, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_replayssm_ring_tracker_ignores_invalid_slots():
    ring_start = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_num_accepted = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_query_len = torch.zeros(2, dtype=torch.int32, device="cuda")
    state_batch_indices = torch.tensor([-1, 2, 1, 0], dtype=torch.int32, device="cuda")

    update_replayssm_ring_trackers(
        ring_start,
        prev_num_accepted,
        prev_query_len,
        state_batch_indices,
        logical_window=16,
        ring_buffer_len=17,
    )

    assert ring_start.tolist() == [0, 0]
    assert prev_num_accepted.tolist() == [0, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_replayssm_spec_ring_tracker_lifecycle():
    ring_start = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_num_accepted = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_query_len = torch.zeros(2, dtype=torch.int32, device="cuda")
    state_batch_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device="cuda")

    observed = []
    for accepted in (4, 3, 4, 4, 4, 2):
        commit_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted,
            prev_query_len,
            state_batch_indices,
            torch.tensor([accepted], dtype=torch.int32, device="cuda"),
            query_start_loc,
            logical_window=16,
            ring_buffer_len=20,
        )
        observed.append(
            (
                int(ring_start[1]),
                int(prev_num_accepted[1]),
                int(prev_query_len[1]),
            )
        )

    assert observed == [
        (0, 0, 4),
        (0, 3, 4),
        (0, 7, 4),
        (0, 11, 4),
        (0, 15, 4),
        (15, 2, 4),
    ]


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
    assert isinstance(get_mamba_ssu_backend(), TritonSSUBackend)


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

    backend(*(tensor,) * 8)

    assert kernel.call_args.kwargs["algorithm"] == expected


def test_uninitialized_backend_raises():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    try:
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_mamba_ssu_backend()
    finally:
        mod._mamba_ssu_backend = old


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
    batch_size = 2
    dim = 64
    dstate = 16
    state = torch.randn(batch_size, dim, dstate, device="cuda")
    x = torch.randn(batch_size, dim, device="cuda")
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device="cuda")
    dt_bias = torch.rand(dim, device="cuda") - 4.0
    A = -torch.rand(dim, dstate, device="cuda")
    B = torch.randn(batch_size, dstate, device="cuda")
    C = torch.randn(batch_size, dstate, device="cuda")
    D = torch.randn(dim, device="cuda")

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


def test_replayssm_flashinfer_call_forwards_explicit_controls(monkeypatch):
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    kernel = Mock(return_value=torch.empty(1, 1, 2, 4))
    monkeypatch.setattr(mod, "_flashinfer_replayssm_kernel", kernel)

    batch, nheads, dim, dstate, ngroups, window = 1, 2, 4, 8, 1, 16
    state = torch.empty(1, nheads, dim, dstate)
    x = torch.empty(batch, nheads, dim)
    dt = torch.empty(batch, nheads, dim)
    A = torch.empty(nheads, dim, dstate)
    B = torch.empty(batch, ngroups, dstate)
    C = torch.empty(batch, ngroups, dstate)
    out = torch.empty_like(x)
    x_cache = torch.empty(1, nheads, window, dim)
    dt_cache = torch.empty(1, nheads, window)
    B_cache = torch.empty(1, ngroups, window, dstate)
    ring_start = torch.zeros(1, dtype=torch.int32)
    prev_num_accepted = torch.zeros(1, dtype=torch.int32)
    prev_query_len = torch.zeros(1, dtype=torch.int32)
    scratch = (torch.empty(1), torch.empty(1), torch.empty(1))

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
        logical_window=window,
        scratch=scratch,
        algorithm="two-kernel",
        d_split=2,
        precompute_heads_per_cta=8,
        enable_stochastic_rounding=True,
        stochastic_rounding_philox_rounds=6,
        update_trackers=False,
        enable_pdl=True,
    )

    args = kernel.call_args.args
    kwargs = kernel.call_args.kwargs
    assert args[4] is ring_start
    assert args[5] is prev_num_accepted
    assert kwargs["algorithm"] == "two-kernel"
    assert kwargs["d_split"] == 2
    assert kwargs["precompute_heads_per_cta"] == 8
    assert kwargs["cb_scaled"] is scratch[0]
    assert kwargs["cumAdt_vec"] is scratch[1]
    assert kwargs["cb_old"] is scratch[2]
    assert kwargs["philox_rounds"] == 6
    assert kwargs["enable_pdl"] is True
    assert kwargs["rand_seed"].shape == (1,)
    assert kwargs["rand_seed"].dtype == torch.int64


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
    prev_query_len = torch.zeros(2, dtype=torch.int32)
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
        prev_query_len,
        logical_window=16,
        state_batch_indices=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        update_trackers=False,
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


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="compatible flashinfer checkpointing_ssu not available",
)
def test_replayssm_flashinfer_backend_rejects_missing_mtp_api(monkeypatch):
    checkpointing_ssu_module = import_module("flashinfer.mamba.checkpointing_ssu")

    def legacy_checkpointing_ssu():
        pass

    monkeypatch.setattr(
        checkpointing_ssu_module, "checkpointing_ssu", legacy_checkpointing_ssu
    )
    with pytest.raises(ImportError, match="native MTP and PDL support"):
        initialize_mamba_ssu_backend(
            MambaConfig(backend=MambaBackendEnum.FLASHINFER),
            _kv_cache_config_with_ssu(),
            use_replayssm=True,
        )


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


def _flashinfer_replayssm_mixer() -> Mock:
    mixer = Mock()
    mixer.use_replayssm = True
    mixer.mamba_config.backend = MambaBackendEnum.FLASHINFER
    mixer.kv_cache = [
        object(),
        object(),
        torch.empty(32, 1, 20, 1),
        object(),
        object(),
    ]
    mixer._replayssm_ring_start = torch.zeros(32, dtype=torch.int32)
    mixer._replayssm_prev_num_accepted = torch.zeros(32, dtype=torch.int32)
    mixer._replayssm_prev_query_len = torch.zeros(32, dtype=torch.int32)
    mixer.replayssm_buffer_len = 16
    return mixer


def test_materialize_replayssm_prefix_gpu_gathers_copied_slots(monkeypatch):
    """V2 gather: only src_col != dst_col rows are copied; hashed slot is src."""
    mixer = _flashinfer_replayssm_mixer()
    kv_cache_config = Mock()
    kv_cache_config.kv_cache_groups = [Mock(layer_names=["mixer"])]
    launches: list[dict[str, list]] = []

    mixer._replayssm_prev_num_accepted[10] = 2

    def fake_launch(mixers, src_row, dst_row, flush_count, _kernel):
        launches.append(
            {
                "src_row": src_row.tolist(),
                "dst_row": dst_row.tolist(),
                "flush_count": flush_count.tolist(),
            }
        )

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch."
        "_launch_replayssm_materialize",
        fake_launch,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch._load_replayssm_materialize",
        lambda: object(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch."
        "_replayssm_materialize_ready",
        lambda _mixers: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch."
        "reset_replayssm_ring_trackers",
        lambda *args, **kwargs: None,
    )

    # Request-state order: req0 copies 0->1, req1 stays on col 1, req2 is fresh.
    src_col_gpu = torch.tensor([0, 1, -1], dtype=torch.int32)
    dst_col_gpu = torch.tensor([1, 1, 0], dtype=torch.int32)
    # Batch order: req0, skipped, req1.
    idx_mapping = torch.tensor([0, -1, 1], dtype=torch.int32)
    block_tables = [
        torch.tensor(
            [
                [10, 11],
                [0, 0],
                [20, 21],
            ],
            dtype=torch.int32,
        )
    ]

    materialize_replayssm_prefix_gpu(
        kv_cache_config,
        [0],
        {"mixer": mixer},
        block_tables,
        src_col_gpu,
        dst_col_gpu,
        idx_mapping,
        num_reqs=3,
    )

    assert launches == [
        {
            "src_row": [10, 0, 21],
            "dst_row": [10, 0, 21],
            "flush_count": [2, -1, -1],
        }
    ]


def test_materialize_replayssm_prefix_mtp_gpu_uses_committed_boundary(
    monkeypatch,
):
    mixer = _flashinfer_replayssm_mixer()
    mixer.replayssm_buffer_len = 12
    mixer.kv_cache[2] = torch.empty(32, 1, 16, 1)
    mixer._replayssm_prev_num_accepted[10] = 4
    mixer._replayssm_prev_num_accepted[21] = 5
    mixer._replayssm_prev_num_accepted[23] = 12
    mixer._replayssm_prev_num_accepted[25] = 10
    mixer._replayssm_prev_num_accepted[27] = 8
    mixer._replayssm_prev_num_accepted[29] = 3
    mixer._replayssm_prev_query_len[10] = 4
    mixer._replayssm_prev_query_len[21] = 4
    mixer._replayssm_prev_query_len[23] = 4
    mixer._replayssm_prev_query_len[25] = 4
    mixer._replayssm_prev_query_len[27] = 4
    mixer._replayssm_ring_start[23] = 3
    mixer._replayssm_ring_start[25] = 5
    mixer._replayssm_ring_start[27] = 7
    mixer._replayssm_ring_start[29] = 9
    kv_cache_config = Mock()
    kv_cache_config.kv_cache_groups = [Mock(layer_names=["mixer"])]
    launches: list[dict[str, list]] = []
    resets: list[list[int]] = []

    def fake_launch(
        mixers, src_row, dst_row, flush_count, _kernel, *, ring_start=None
    ):
        launches.append(
            {
                "src_row": src_row.tolist(),
                "dst_row": dst_row.tolist(),
                "flush_count": flush_count.tolist(),
                "ring_start": ring_start.tolist(),
            }
        )

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch."
        "_launch_replayssm_materialize",
        fake_launch,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch._load_replayssm_materialize",
        lambda: object(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch."
        "_replayssm_materialize_ready",
        lambda _mixers: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch."
        "reset_replayssm_ring_trackers",
        lambda _start, _accepted, _query, slots: resets.append(slots.tolist()),
    )

    materialize_replayssm_prefix_mtp_gpu(
        kv_cache_config,
        [0],
        {"mixer": mixer},
        [
            torch.tensor(
                [
                    [10, 11, 12],
                    [0, 0, 0],
                    [20, 21, 22],
                    [23, 24, 0],
                    [25, 26, 0],
                    [27, 28, 0],
                    [29, 30, 0],
                ],
                dtype=torch.int32,
            )
        ],
        torch.tensor([0, -1, 1, 0, 0, 0, 0], dtype=torch.int32),
        torch.tensor([1, 0, 2, 1, 1, 1, 1], dtype=torch.int32),
        torch.tensor([2, 0, 3, 4, 2, 2, 1], dtype=torch.int32),
        num_reqs=7,
    )

    assert launches == [
        {
            "src_row": [10, 0, 21, 23, 25, 27, 29],
            "dst_row": [11, 0, 22, 24, 26, 28, 30],
            "flush_count": [6, -1, 8, 4, 2, 10, 3],
            "ring_start": [0, 0, 0, 15, 15, 7, 9],
        }
    ]
    assert resets == [[11, 22, 24, 26, 28, 30]]
