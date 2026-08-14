# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferReplaySSMBackend,
    FlashInferSSUBackend,
    TritonReplaySSMBackend,
    TritonSSUBackend,
    get_mamba_ssu_backend,
    get_replayssm_backend,
    initialize_mamba_ssu_backend,
    initialize_replayssm_backend,
    selective_state_update,
    selective_state_update_replayssm_flashinfer,
    selective_state_update_replayssm_triton,
    update_replayssm_ring_trackers,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
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
    from flashinfer.mamba import checkpointing_ssu  # noqa: F401

    HAS_FLASHINFER_CHECKPOINTING_SSU = True
except ImportError:
    HAS_FLASHINFER_CHECKPOINTING_SSU = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_replayssm_ring_tracker_lifecycle():
    ring_start = torch.zeros(2, dtype=torch.int32, device="cuda")
    prev_num_accepted = torch.zeros(2, dtype=torch.int32, device="cuda")
    state_batch_indices = torch.tensor([1], dtype=torch.int32, device="cuda")

    observed = []
    for _ in range(33):
        update_replayssm_ring_trackers(
            ring_start,
            prev_num_accepted,
            state_batch_indices,
            logical_window=16,
        )
        observed.append((int(ring_start[1]), int(prev_num_accepted[1])))

    assert observed[4] == (0, 5)
    assert observed[15] == (0, 16)
    assert observed[16] == (16, 1)
    assert observed[31] == (16, 16)
    assert observed[32] == (15, 1)


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

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    with pytest.raises(RuntimeError, match="not been initialized"):
        get_mamba_ssu_backend()
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


def test_replayssm_default_backend_is_triton():
    initialize_replayssm_backend(MambaConfig(), use_replayssm=True)
    backend = get_replayssm_backend()
    assert isinstance(backend, TritonReplaySSMBackend)
    assert backend.name == "triton"


def test_replayssm_explicit_triton_backend():
    initialize_replayssm_backend(
        MambaConfig(backend=MambaBackendEnum.TRITON), use_replayssm=True
    )
    backend = get_replayssm_backend()
    assert isinstance(backend, TritonReplaySSMBackend)


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
def test_replayssm_flashinfer_backend_init():
    initialize_replayssm_backend(
        MambaConfig(backend=MambaBackendEnum.FLASHINFER), use_replayssm=True
    )
    backend = get_replayssm_backend()
    assert isinstance(backend, FlashInferReplaySSMBackend)
    assert backend.name == "flashinfer"


def test_replayssm_disabled_clears_backend():
    initialize_replayssm_backend(MambaConfig(), use_replayssm=True)
    assert get_replayssm_backend() is not None
    initialize_replayssm_backend(MambaConfig(), use_replayssm=False)
    with pytest.raises(RuntimeError, match="not been initialized"):
        get_replayssm_backend()


def test_replayssm_cpu_backend_rejected():
    with pytest.raises(ValueError, match="does not support mamba backend"):
        initialize_replayssm_backend(
            MambaConfig(backend=MambaBackendEnum.CPU), use_replayssm=True
        )


def test_replayssm_uninitialized_backend_raises():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._replayssm_backend
    mod._replayssm_backend = None
    try:
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_replayssm_backend()
    finally:
        mod._replayssm_backend = old


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
def test_replayssm_triton_entry_rejects_flashinfer_backend():
    initialize_replayssm_backend(
        MambaConfig(backend=MambaBackendEnum.FLASHINFER), use_replayssm=True
    )
    tensor = torch.empty(1)
    with pytest.raises(RuntimeError, match="Triton ReplaySSM"):
        selective_state_update_replayssm_triton(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            out=tensor,
        )


@pytest.mark.skipif(
    not HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer.mamba.checkpointing_ssu not available",
)
def test_replayssm_flashinfer_call(monkeypatch):
    import flashinfer.mamba

    kernel = Mock(return_value=torch.empty(1, 1, 2, 4))
    monkeypatch.setattr(flashinfer.mamba, "checkpointing_ssu", kernel)
    initialize_replayssm_backend(
        MambaConfig(backend=MambaBackendEnum.FLASHINFER), use_replayssm=True
    )

    batch, nheads, dim, dstate, ngroups, L = 1, 2, 4, 8, 1, 16
    state = torch.empty(1, nheads, dim, dstate)
    x = torch.empty(batch, nheads, dim)
    dt = torch.empty(batch, nheads, dim)
    A = torch.empty(nheads, dim, dstate)
    B = torch.empty(batch, ngroups, dstate)
    C = torch.empty(batch, ngroups, dstate)
    D = torch.empty(nheads, dim)
    dt_bias = torch.empty(nheads, dim)
    out = torch.empty_like(x)
    x_cache = torch.empty(1, nheads, L, dim)
    dt_cache = torch.empty(1, nheads, L)
    B_cache = torch.empty(1, ngroups, L, dstate)
    ring_start = torch.zeros(1, dtype=torch.int32)
    prev_num_accepted = torch.zeros(1, dtype=torch.int32)

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
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )
    assert kernel.call_count == 1
    kwargs = kernel.call_args.kwargs
    assert kwargs["dt_softplus"] is True
    # ring_start / prev_num_accepted are positional after the caches.
    args = kernel.call_args.args
    assert args[4] is ring_start
    assert args[5] is prev_num_accepted


@pytest.mark.skipif(
    HAS_FLASHINFER_CHECKPOINTING_SSU,
    reason="flashinfer checkpointing_ssu is installed",
)
def test_replayssm_flashinfer_import_error():
    with pytest.raises(ImportError, match="FlashInfer is required"):
        FlashInferReplaySSMBackend(MambaConfig(backend=MambaBackendEnum.FLASHINFER))


def test_replayssm_dispatch_fn_uses_initialized_backend():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    called = Mock(return_value=torch.empty(1))
    old = mod._replayssm_backend
    mod._replayssm_backend = TritonReplaySSMBackend(MambaConfig())
    mod._replayssm_backend._kernel = called
    try:
        tensor = torch.empty(1)
        selective_state_update_replayssm_triton(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            out=tensor,
        )
        assert called.call_count == 1
    finally:
        mod._replayssm_backend = old
