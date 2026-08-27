# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from unittest.mock import Mock

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    quantize_ssm_state,
)
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


def _call_replayssm_flashinfer(
    monkeypatch,
    state_dtype,
    *,
    state_scale=None,
    scratch=None,
    algorithm="auto",
    d_split=None,
    precompute_heads_per_cta=0,
    enable_pdl=False,
):
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    kernel = Mock(return_value=torch.empty(1, 1, 8, 64))
    monkeypatch.setattr(mod, "_flashinfer_replayssm_kernel", kernel)
    batch, nheads, dim, dstate, ngroups, window = 1, 8, 64, 128, 1, 16
    state = torch.empty(1, nheads, dim, dstate, dtype=state_dtype)
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
        state_scale=state_scale,
        scratch=scratch,
        algorithm=algorithm,
        d_split=d_split,
        precompute_heads_per_cta=precompute_heads_per_cta,
        enable_stochastic_rounding=True,
        stochastic_rounding_philox_rounds=6,
        update_trackers=False,
        enable_pdl=enable_pdl,
    )
    return kernel, ring_start, prev_num_accepted


def test_replayssm_flashinfer_call_forwards_explicit_controls(monkeypatch):
    scratch = (torch.empty(1), torch.empty(1), torch.empty(1))
    kernel, ring_start, prev_num_accepted = _call_replayssm_flashinfer(
        monkeypatch,
        torch.float16,
        scratch=scratch,
        algorithm="two-kernel",
        d_split=2,
        precompute_heads_per_cta=8,
        enable_pdl=True,
    )

    args = kernel.call_args.args
    kwargs = kernel.call_args.kwargs
    assert args[4] is ring_start
    assert args[5] is prev_num_accepted
    assert kwargs["algorithm"] == "two-kernel"
    assert kwargs["state_scale"] is None
    assert kwargs["d_split"] == 2
    assert kwargs["precompute_heads_per_cta"] == 8
    assert kwargs["cb_scaled"] is scratch[0]
    assert kwargs["cumAdt_vec"] is scratch[1]
    assert kwargs["cb_old"] is scratch[2]
    assert kwargs["philox_rounds"] == 6
    assert kwargs["enable_pdl"] is True
    assert kwargs["rand_seed"].shape == (1,)
    assert kwargs["rand_seed"].dtype == torch.int64


def test_replayssm_flashinfer_forwards_fp8_stochastic_rounding(monkeypatch):
    state_scale = torch.empty(1, 8, 64)
    kernel, _, _ = _call_replayssm_flashinfer(
        monkeypatch,
        torch.float8_e4m3fn,
        state_scale=state_scale,
    )

    kwargs = kernel.call_args.kwargs
    assert kwargs["state_scale"] is state_scale
    assert kwargs["d_split"] is None
    assert kwargs["algorithm"] == "auto"
    assert kwargs["philox_rounds"] == 6
    assert kwargs["rand_seed"].shape == (1,)
    assert kwargs["rand_seed"].dtype == torch.int64


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantized_ssm_state_uses_per_head_dim_scales(dtype):
    state = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 3.0, -4.0]],
                [[0.25, -0.5, 0.75, -1.0], [8.0, -4.0, 2.0, -1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    quantized, decode_scale = quantize_ssm_state(state, dtype)
    restored = quantized.float() * decode_scale.unsqueeze(-1)

    assert quantized.dtype == dtype
    assert decode_scale.dtype == torch.float32
    assert decode_scale.shape == state.shape[:-1]
    assert torch.isfinite(decode_scale).all()
    assert (decode_scale > 0).all()
    assert torch.allclose(restored, state, rtol=0.1, atol=0.04)
    if dtype == torch.int8:
        assert torch.equal(
            quantized[0, 0, 1],
            torch.tensor([32, -64, 95, -127], dtype=torch.int8),
        )


@pytest.mark.parametrize(
    ("config_dtype", "torch_dtype"),
    [("int8", torch.int8), ("fp8_e4m3fn", torch.float8_e4m3fn)],
)
def test_mamba2_quantized_ssm_dtype_does_not_quantize_conv(config_dtype, torch_dtype):
    assert MambaStateDtypeCalculator.mamba2_state_dtype(
        torch.bfloat16,
        "auto",
        config_dtype,
    ) == (torch.bfloat16, torch_dtype)


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


def test_quantized_replayssm_layout_appends_fp32_state_scale():
    base_shapes = ((64, 3), (8, 4, 16))
    shapes = MambaStateShapeCalculator.append_replayssm_ring(
        base_shapes,
        n_groups=4,
        tp_world_size=2,
        logical_window=16,
        backend=MambaBackendEnum.FLASHINFER,
        include_state_scale=True,
    )
    dtypes = MambaStateDtypeCalculator.append_replayssm_ring(
        (torch.bfloat16, torch.int8),
        torch.bfloat16,
    )

    assert shapes[-1] == base_shapes[1][:-1]
    assert dtypes[-1] == torch.float32
    assert len(shapes) == len(dtypes) == 6
