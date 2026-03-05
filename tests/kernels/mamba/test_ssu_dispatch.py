# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    TritonSSUBackend,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    selective_state_update,
)

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


def test_default_backend_is_triton():
    initialize_mamba_ssu_backend(None)
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)
    assert backend.name == "triton"


def test_explicit_triton_backend():
    initialize_mamba_ssu_backend(MambaBackendEnum.TRITON)
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
def test_flashinfer_backend_init():
    initialize_mamba_ssu_backend(MambaBackendEnum.FLASHINFER)
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, FlashInferSSUBackend)
    assert backend.name == "flashinfer"


def test_uninitialized_backend_raises():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    with pytest.raises(RuntimeError, match="not been initialized"):
        get_mamba_ssu_backend()
    mod._mamba_ssu_backend = old


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
def test_flashinfer_dst_state_batch_indices_raises():
    initialize_mamba_ssu_backend(MambaBackendEnum.FLASHINFER)
    device = "cuda"
    state = torch.randn(1, 64, 16, device=device)
    x = torch.randn(1, 64, device=device)
    dt = torch.randn(1, 64, device=device)
    dt_bias = torch.rand(64, device=device) - 4.0
    A = -torch.rand(64, 16, device=device)
    B = torch.randn(1, 16, device=device)
    C = torch.randn(1, 16, device=device)
    D = torch.randn(64, device=device)

    with pytest.raises(ValueError, match="dst_state_batch_indices"):
        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            dt_softplus=True,
            dst_state_batch_indices=torch.zeros(1, dtype=torch.int32, device=device),
        )


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
def test_flashinfer_num_accepted_tokens_raises():
    initialize_mamba_ssu_backend(MambaBackendEnum.FLASHINFER)
    device = "cuda"
    state = torch.randn(1, 64, 16, device=device)
    x = torch.randn(1, 64, device=device)
    dt = torch.randn(1, 64, device=device)
    dt_bias = torch.rand(64, device=device) - 4.0
    A = -torch.rand(64, 16, device=device)
    B = torch.randn(1, 16, device=device)
    C = torch.randn(1, 16, device=device)
    D = torch.randn(64, device=device)

    with pytest.raises(ValueError, match="num_accepted_tokens"):
        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            dt_softplus=True,
            num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
        )


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
def test_flashinfer_cu_seqlens_raises():
    initialize_mamba_ssu_backend(MambaBackendEnum.FLASHINFER)
    device = "cuda"
    state = torch.randn(1, 64, 16, device=device)
    x = torch.randn(1, 64, device=device)
    dt = torch.randn(1, 64, device=device)
    dt_bias = torch.rand(64, device=device) - 4.0
    A = -torch.rand(64, 16, device=device)
    B = torch.randn(1, 16, device=device)
    C = torch.randn(1, 16, device=device)
    D = torch.randn(64, device=device)

    with pytest.raises(ValueError, match="cu_seqlens"):
        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            dt_softplus=True,
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=device),
        )


@pytest.mark.skipif(HAS_FLASHINFER, reason="flashinfer is installed")
def test_flashinfer_import_error():
    with pytest.raises(ImportError, match="FlashInfer is required"):
        FlashInferSSUBackend()


def test_triton_basic_call():
    initialize_mamba_ssu_backend(MambaBackendEnum.TRITON)
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
