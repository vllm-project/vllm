# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniMax QK RMS-norm: NCCL reference vs Lamport fused kernel."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.multiprocessing import spawn

from tests.kernels.utils import opcheck
from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.minimax_rms_norm import MiniMaxText01RMSNormTP
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed

_LAMPORT_WORKSPACE_MODULE = (
    "vllm.model_executor.layers.minimax_rms_norm.lamport_workspace"
)


def _patch_lamport_workspace(monkeypatch, get_allreduce_workspace):
    module = ModuleType(_LAMPORT_WORKSPACE_MODULE)
    module.get_allreduce_workspace = get_allreduce_workspace
    monkeypatch.setitem(sys.modules, _LAMPORT_WORKSPACE_MODULE, module)


def _build_minimax_norm(norm_cls):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        return norm_cls(8)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_minimax_rmsnorm_skips_workspace_when_custom_allreduce_disabled(
    monkeypatch,
):
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.minimax_rms_norm import rms_norm_tp

    monkeypatch.setattr(rms_norm_tp, "_MINIMAX_FUSED_AR_RMS_QK", object())
    monkeypatch.setattr(rms_norm_tp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(rms_norm_tp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "_ENABLE_CUSTOM_ALL_REDUCE", False)

    def fail_workspace(*args, **kwargs):
        pytest.fail("MiniMax workspace should honor disable_custom_all_reduce")

    _patch_lamport_workspace(monkeypatch, fail_workspace)

    norm = _build_minimax_norm(rms_norm_tp.MiniMaxText01RMSNormTP)

    assert norm.workspace is None


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_minimax_rmsnorm_skips_disabled_custom_allreduce_communicator(
    monkeypatch,
):
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.minimax_rms_norm import rms_norm_tp

    tp_group = SimpleNamespace(
        cpu_group=object(),
        device_communicator=SimpleNamespace(
            use_custom_allreduce=True,
            ca_comm=SimpleNamespace(disabled=True),
        ),
    )

    monkeypatch.setattr(rms_norm_tp, "_MINIMAX_FUSED_AR_RMS_QK", object())
    monkeypatch.setattr(rms_norm_tp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(rms_norm_tp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(rms_norm_tp, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(parallel_state, "_ENABLE_CUSTOM_ALL_REDUCE", True)

    def fail_workspace(*args, **kwargs):
        pytest.fail("MiniMax workspace should follow disabled CA communicator")

    _patch_lamport_workspace(monkeypatch, fail_workspace)

    norm = _build_minimax_norm(rms_norm_tp.MiniMaxText01RMSNormTP)

    assert norm.workspace is None


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_minimax_rmsnorm_uses_workspace_when_custom_allreduce_available(
    monkeypatch,
):
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.minimax_rms_norm import rms_norm_tp

    workspace = torch.empty(0)
    calls = []
    tp_group = SimpleNamespace(
        cpu_group=object(),
        device_communicator=SimpleNamespace(
            use_custom_allreduce=True,
            ca_comm=SimpleNamespace(disabled=False),
        ),
    )

    monkeypatch.setattr(rms_norm_tp, "_MINIMAX_FUSED_AR_RMS_QK", object())
    monkeypatch.setattr(rms_norm_tp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(rms_norm_tp, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(rms_norm_tp, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(parallel_state, "_ENABLE_CUSTOM_ALL_REDUCE", True)

    def fake_workspace(**kwargs):
        calls.append(kwargs)
        return workspace

    _patch_lamport_workspace(monkeypatch, fake_workspace)

    norm = _build_minimax_norm(rms_norm_tp.MiniMaxText01RMSNormTP)

    assert norm.workspace is workspace
    assert calls == [
        {
            "rank": 1,
            "world_size": 2,
            "max_tokens": 2048,
            "process_group": tp_group.cpu_group,
        }
    ]


@ensure_current_vllm_config()
def _worker_forward_qk(
    local_rank,
    world_size,
    port,
    num_tokens,
    hidden_q_full,
    hidden_k_full,
    dtype,
    seed,
    eps,
):
    """Per-rank worker: compare NCCL allreduce path vs Lamport fused kernel."""

    if not hasattr(torch.ops._C, "minimax_allreduce_rms_qk"):
        cleanup_dist_env_and_memory()
        return
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    hq = hidden_q_full // world_size
    hk = hidden_k_full // world_size

    q_norm = MiniMaxText01RMSNormTP(hidden_q_full, eps=eps).cuda()
    k_norm = MiniMaxText01RMSNormTP(hidden_k_full, eps=eps).cuda()

    set_random_seed(seed)
    qw = torch.randn(hidden_q_full, dtype=dtype, device="cuda")
    kw = torch.randn(hidden_k_full, dtype=dtype, device="cuda")
    q_norm.weight = nn.Parameter(qw[local_rank * hq : (local_rank + 1) * hq])
    k_norm.weight = nn.Parameter(kw[local_rank * hk : (local_rank + 1) * hk])

    torch.manual_seed(seed + 1000 + local_rank)
    qkv = torch.randn(num_tokens, hq + hk + hk, dtype=dtype, device="cuda")

    q_ref, k_ref, v_ref = qkv.clone().split([hq, hk, hk], dim=-1)
    ref_q, ref_k = MiniMaxText01RMSNormTP.forward_qk(q_norm, k_norm, q_ref, k_ref)

    # Set up Lamport workspace.
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.model_executor.layers.minimax_rms_norm.lamport_workspace import (
        get_allreduce_workspace,
    )

    workspace = get_allreduce_workspace(
        rank=local_rank,
        world_size=world_size,
        max_tokens=num_tokens,
        process_group=get_tp_group().cpu_group,
    )

    opcheck(
        torch.ops._C.minimax_allreduce_rms_qk,
        (
            qkv.clone(),
            q_norm.weight,
            k_norm.weight,
            workspace,
            hq,
            hk,
            local_rank,
            world_size,
            eps,
        ),
    )
    fused_q, fused_k = torch.ops._C.minimax_allreduce_rms_qk(
        qkv.clone(),
        q_norm.weight,
        k_norm.weight,
        workspace,
        hq,
        hk,
        local_rank,
        world_size,
        eps,
    )
    _, _, fused_v = qkv.split([hq, hk, hk], dim=-1)
    torch.accelerator.synchronize()

    torch.testing.assert_close(
        fused_q,
        ref_q,
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(fused_k, ref_k, atol=3e-2, rtol=3e-2)

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("num_tokens", [1, 128, 333])
@pytest.mark.parametrize(
    "hidden_dims",
    [(6144, 1024)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
def test_minimax_reduce_rms_qk(
    world_size,
    num_tokens,
    hidden_dims,
    dtype,
    eps,
    seed,
):
    num_gpus = current_platform.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    hidden_q_full, hidden_k_full = hidden_dims
    port = str(get_open_port())
    spawn(
        _worker_forward_qk,
        args=(
            world_size,
            port,
            num_tokens,
            hidden_q_full,
            hidden_k_full,
            dtype,
            seed,
            eps,
        ),
        nprocs=world_size,
        join=True,
    )
