# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm.model_executor.models import llama
from vllm.platforms import current_platform
from vllm.v1.worker.tpsp_profile import SPProfile, TPSPProjection, get_tpsp_backend


def _check_tpsp_backend(
    rank: int, rendezvous: str, world_size: int, availability: mp.SimpleQueue
) -> None:
    torch.accelerator.set_device_index(rank)
    device = torch.device(current_platform.device_type, rank)
    dist.init_process_group(
        current_platform.dist_backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    comm = None
    if device.type == "cuda":
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        comm = PyNcclCommunicator(dist.new_group(backend="gloo"), device)
    tp_group = SimpleNamespace(
        device_group=dist.group.WORLD,
        device_communicator=SimpleNamespace(pynccl_comm=comm),
    )
    try:
        backend = get_tpsp_backend(dist.group.WORLD.group_name, device)
        context = None
        if backend is not None:
            with patch(
                "vllm.distributed.parallel_state.get_tp_group", return_value=tp_group
            ):
                context = backend.open(
                    dtype=torch.bfloat16,
                    tp_size=world_size,
                    hidden_size=4096,
                    max_batched_tokens=129,
                    group_name=dist.group.WORLD.group_name,
                    device=device,
                )
        available = torch.tensor(int(context is not None), device=device)
        dist.all_reduce(available, op=dist.ReduceOp.MIN)
        if rank == 0:
            availability.put(bool(available.item()))
        if not available.item():
            if context is not None:
                backend.close(context)
            if backend is not None:
                backend.close()
            return
        assert backend is not None and context is not None
        for tokens, width in ((1, 64), (5, 2048), (128, 2048), (129, 7168)):
            generator = torch.Generator(device=device).manual_seed(123 + rank)
            a = torch.randn(
                (tokens + 1, width),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )[1:]
            b = torch.randn(
                (width + 1, 4096),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )[1:] / math.sqrt(width)
            weight = torch.ones(4096, device=device, dtype=torch.bfloat16)
            residual = torch.randn(
                (tokens, 4096),
                dtype=torch.bfloat16,
                device=device,
                generator=torch.Generator(device=device).manual_seed(407),
            )
            rows = (tokens + world_size - 1) // world_size
            local_residual = torch.zeros(
                (rows, 4096), device=device, dtype=torch.bfloat16
            )
            start = rank * rows
            count = min(rows, max(tokens - start, 0))
            local_residual[:count] = residual[start : start + count]

            expected = a @ b
            dist.all_reduce(expected)
            expected_residual = residual.clone()
            torch.ops._C.fused_add_rms_norm(expected, expected_residual, weight, 1e-5)
            reduced, _, gathered = backend.fused_gemm_rs_norm_ag(
                a, b, weight, local_residual, 1e-5, 64, context=context
            )
            torch.testing.assert_close(
                gathered,
                expected,
                rtol=0.02,
                atol=0.05,
                msg=f"rank={rank} tokens={tokens}",
            )
            torch.testing.assert_close(
                reduced[:count],
                expected_residual[start : start + count],
                rtol=0.02,
                atol=0.05,
            )
            if tokens == 129 and backend.supports_projection_bias:
                projection_bias = torch.full_like(weight, 0.125)
                norm_bias = torch.full_like(weight, 0.25)
                expected_with_bias = a @ b
                dist.all_reduce(expected_with_bias)
                expected_with_bias += projection_bias
                expected_residual_with_bias = residual + expected_with_bias
                expected_normalized = torch.nn.functional.layer_norm(
                    expected_residual_with_bias, (4096,), weight, norm_bias, 1e-5
                )
                reduced, _, gathered = backend.fused_gemm_rs_norm_ag(
                    a,
                    b,
                    weight,
                    local_residual,
                    1e-5,
                    64,
                    norm_type="layer_norm",
                    projection_bias=projection_bias,
                    norm_bias=norm_bias,
                    context=context,
                )
                torch.testing.assert_close(
                    reduced[:count],
                    expected_residual_with_bias[start : start + count],
                    rtol=0.02,
                    atol=0.05,
                )
                torch.testing.assert_close(
                    gathered, expected_normalized, rtol=0.02, atol=0.05
                )

        profile = backend.profile(
            tp_size=world_size,
            hidden_size=4096,
            input_width=64,
            max_batched_tokens=128,
            norm_eps=1e-5,
            time_budget_s=60,
            context=context,
        )
        assert profile.tp_size == world_size
        assert profile.hidden_size == 4096
        assert profile.max_batched_tokens == 128
        if profile.enabled:
            assert isinstance(profile.config, int)
        backend.close(context)
        backend.close()
    finally:
        if comm is not None:
            comm.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", (2, 4))
def test_tpsp_backend(tp_size: int):
    if current_platform.get_tpsp_backend_cls() is None:
        pytest.skip("TPSP is not supported on this platform")
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} devices")
    with tempfile.TemporaryDirectory() as directory:
        availability = mp.get_context("spawn").SimpleQueue()
        mp.spawn(
            _check_tpsp_backend,
            args=(os.path.join(directory, "rendezvous"), tp_size, availability),
            nprocs=tp_size,
        )
        if not availability.get():
            pytest.skip("TPSP backend is unavailable for this configuration")


def test_llama_tpsp_forward(monkeypatch):
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(llama, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        llama,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True),
    )

    class Projection(nn.Module):
        input_size_per_partition = 2
        bias = None

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(2, dtype=torch.bfloat16))

        def forward(self, x):
            return x @ self.weight, None

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.variance_epsilon = 1e-5

        def forward(self, x, residual):
            result = x + residual
            return result, result

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = Projection()

        def compute_attention(self, positions, hidden_states):
            return hidden_states + 1

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = Projection()

        def compute_down_proj_input(self, hidden_states):
            return hidden_states + 1

    context = object()

    class Backend:
        supports_projection_bias = False

        def __init__(self):
            self.calls = 0

        def fused_gemm_rs_norm_ag(
            self,
            x,
            weight,
            norm_weight,
            residual,
            eps,
            config,
            *,
            projection_bias,
            context,
        ):
            self.calls += 1
            assert projection_bias is None
            assert context is model.tpsp_projections["o"].context
            reduced = x @ weight + residual
            return reduced, None, reduced

    layer = llama.LlamaDecoderLayer.__new__(llama.LlamaDecoderLayer)
    nn.Module.__init__(layer)
    layer.hidden_size = 2
    layer.input_layernorm = nn.Identity()
    layer.self_attn = Attention()
    layer.post_attention_layernorm = Norm()
    layer.mlp = MLP()

    model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2)
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList([layer])
    model.norm = Norm()
    model.start_layer, model.end_layer = 0, 1
    model.aux_hidden_state_layers = ()
    model._aux_upstream_total_cached = 0
    backend = Backend()
    model.tpsp_projections = nn.ModuleDict(
        {name: TPSPProjection(2, 2, 1e-5, 1, "test") for name in ("o", "down")}
    )
    layer.tpsp_projections = dict(model.tpsp_projections.items())
    for projection in model.tpsp_projections.values():
        projection.profile = SPProfile(
            1, 2, 8, "enabled", "", threshold_tokens=1, config=64
        )
        projection.backend = backend
        projection.context = context

    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 2

    model.tpsp_projections["o"].profile = SPProfile(
        1, 2, 8, "enabled", "", threshold_tokens=3, config=64
    )
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 3
