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
from vllm.v1.worker import tpsp_profile
from vllm.v1.worker.tpsp_profile import (
    SPProfile,
    TPSPProfile,
    TPSPProjection,
    get_tpsp_backend,
    profile_tpsp,
    profile_tpsp_projections,
)


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
        with patch(
            "vllm.distributed.parallel_state.get_tp_group", return_value=tp_group
        ):
            large_context = backend.open(
                dtype=torch.bfloat16,
                tp_size=world_size,
                hidden_size=64,
                max_batched_tokens=1_000_000,
                group_name=dist.group.WORLD.group_name,
                device=device,
            )
        assert large_context is not None
        capped_rows = (
            backend.tpsp_max_microchunk_tokens + world_size - 1
        ) // world_size
        assert large_context.max_chunk_rows == capped_rows
        if large_context.workspace is not None:
            assert large_context.workspace.numel() == (
                4 * world_size * capped_rows * 64 + 3 * world_size * 4
            )
            with pytest.raises(ValueError, match="P2P workspace capacity"):
                backend.fused_gemm_rs_norm_ag(
                    torch.empty(
                        (world_size * (capped_rows + 1), 64),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    torch.empty((64, 64), dtype=torch.bfloat16, device=device),
                    torch.ones(64, dtype=torch.bfloat16, device=device),
                    torch.empty(
                        (capped_rows + 1, 64), dtype=torch.bfloat16, device=device
                    ),
                    1e-5,
                    capped_rows + 1,
                    context=large_context,
                )
        backend.close(large_context)
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
            with patch.object(
                torch.ops._C,
                "tpsp_fused_matmul_reduce_scatter_norm_all_gather",
                wraps=torch.ops._C.tpsp_fused_matmul_reduce_scatter_norm_all_gather,
            ) as fused_op:
                reduced, _, gathered = backend.fused_gemm_rs_norm_ag(
                    a, b, weight, local_residual, 1e-5, 64, context=context
                )
            assert fused_op.call_args.args[11] is context.workspace
            assert bool(fused_op.call_args.args[12]) == (context.workspace is not None)
            assert fused_op.call_args.args[14] == (
                rank if context.workspace is not None else -1
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
            if tokens == 129:
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

        class Projection(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.input_size_per_partition = width
                self.weight = nn.Parameter(
                    torch.randn(
                        4096,
                        width,
                        device=device,
                        dtype=torch.bfloat16,
                        generator=torch.Generator(device=device).manual_seed(
                            2026 + rank + width
                        ),
                    )
                    / math.sqrt(width)
                )
                self.bias = None

            def forward(self, x):
                result = torch.nn.functional.linear(x, self.weight)
                dist.all_reduce(result)
                return result, None

        class Norm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.ones(4096, device=device, dtype=torch.bfloat16)
                )
                self.variance_epsilon = 1e-5

            def forward(self, x, residual):
                torch.ops._C.fused_add_rms_norm(
                    x, residual, self.weight, self.variance_epsilon
                )
                return x, residual

        o_proj, down_proj = Projection(64), Projection(4096)
        o_norm, down_norm = Norm(), Norm()
        profile = backend.profile(
            projection=o_proj,
            norm=o_norm,
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
        chunk_profile = backend.profile(
            projection=o_proj,
            norm=o_norm,
            tp_size=world_size,
            hidden_size=4096,
            input_width=64,
            max_batched_tokens=16,
            norm_eps=1e-5,
            time_budget_s=60,
            context=context,
            config_only=True,
        )
        assert chunk_profile.status == "candidate"
        assert chunk_profile.config is not None
        assert chunk_profile.threshold_tokens is None

        plans = (
            TPSPProjection(64, 4096, 1e-5, world_size, dist.group.WORLD.group_name),
            TPSPProjection(4096, 4096, 1e-5, world_size, dist.group.WORLD.group_name),
        )
        with (
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=SimpleNamespace(
                    world_size=world_size,
                    rank_in_group=rank,
                    device_group=dist.group.WORLD,
                    device_communicator=tp_group.device_communicator,
                ),
            ),
        ):
            for plan in plans:
                plan.backend = backend
                plan.config = 64
                plan.context = backend.open(
                    dtype=torch.bfloat16,
                    tp_size=world_size,
                    hidden_size=4096,
                    max_batched_tokens=128,
                    group_name=dist.group.WORLD.group_name,
                    device=device,
                )
                assert plan.context is not None
            pair = profile_tpsp_projections(
                plans[0], o_proj, o_norm, plans[1], down_proj, down_norm, 128
            )
            assert pair.measurements and pair.measurements[0].tokens == 128
            for plan in plans:
                backend.close(plan.context)
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


def test_llama_tpsp_disables_compile_on_cpu(monkeypatch):
    model = nn.Module()
    model.make_empty_intermediate_tensors = lambda: None
    monkeypatch.setattr(
        llama.LlamaForCausalLM, "_init_model", lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        llama,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=2), enable_tpsp=True
        ),
        device_config=SimpleNamespace(device_type="cpu"),
        quant_config=None,
        lora_config=None,
    )

    llama.LlamaForCausalLM(vllm_config=config)

    assert model.tpsp_requested
    assert model.do_not_compile


def test_llama_tpsp_forward(monkeypatch):
    group = SimpleNamespace(world_size=1, rank_in_group=0, device_group=object())
    from vllm.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        llama,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True),
    )

    class Projection(nn.Module):
        input_size_per_partition = 2

        def __init__(self, bias=False):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
            self.bias = (
                nn.Parameter(torch.zeros(2, dtype=torch.bfloat16)) if bias else None
            )
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            result = x @ self.weight
            if self.bias is not None:
                result += self.bias
            return result, None

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
            self.o_proj = Projection(bias=True)

        def forward(self, positions, hidden_states, tpsp_active=False):
            result = hidden_states + 1
            return result if tpsp_active else self.o_proj(result)[0]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = Projection()

        def forward(self, hidden_states, tpsp_active=False):
            result = hidden_states + 1
            return result if tpsp_active else self.down_proj(result)[0]

    context = object()

    class Backend:
        def __init__(self):
            self.calls = 0
            self.biases = []

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
            self.biases.append(projection_bias)
            assert context is o_plan.context or context is down_plan.context
            reduced = x @ weight + residual
            if projection_bias is not None:
                reduced += projection_bias
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
    model.tpsp_requested = False
    model.max_tpsp_batched_tokens = 8
    backend = Backend()
    o_plan, down_plan = (TPSPProjection(2, 2, 1e-5, 1, "test") for _ in range(2))
    for projection in (o_plan, down_plan):
        projection.config = 64
        projection.backend = backend
        projection.context = context

    def set_profile(threshold: int | None) -> None:
        model.tpsp = TPSPProfile(threshold is not None, threshold, 8, o_plan, down_plan)
        if threshold is not None:
            layer.tpsp = model.tpsp

    set_profile(1)
    assert model.tpsp.is_active(2)
    with pytest.raises(ValueError, match="profiled range"):
        model.tpsp.is_active(9)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        model.forward(
            None,
            None,
            None,
            inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16),
            unexpected=1,
        )
    model.aux_hidden_state_layers = (1,)
    with pytest.raises(ValueError, match="auxiliary hidden states"):
        model.forward(
            None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
        )
    model.aux_hidden_state_layers = ()

    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 2
    assert backend.biases[0] is layer.self_attn.o_proj.bias
    assert backend.biases[1] is None
    assert layer.self_attn.o_proj.calls == 0
    assert layer.mlp.down_proj.calls == 0

    set_profile(3)
    assert not model.tpsp.is_active(2)
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 2
    assert layer.self_attn.o_proj.calls == 1
    assert layer.mlp.down_proj.calls == 1

    set_profile(1)
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 4
    assert layer.self_attn.o_proj.calls == 1
    assert layer.mlp.down_proj.calls == 1
    set_profile(3)
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 4
    assert layer.self_attn.o_proj.calls == 2
    assert layer.mlp.down_proj.calls == 2

    set_profile(None)
    assert not model.tpsp.enabled
    assert not model.tpsp.is_active(2)
    with pytest.raises(RuntimeError, match="invalid enabled profile"):
        TPSPProfile(True, None, 8, o_plan, down_plan).is_active(2)
    result = model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert backend.calls == 4
    assert layer.self_attn.o_proj.calls == 3
    assert layer.mlp.down_proj.calls == 3

    set_profile(1)
    first_hidden, first_residual = layer(
        None,
        torch.ones(2, 2, dtype=torch.bfloat16),
        None,
        next_norm=model.norm,
    )
    hidden, residual = layer(
        None,
        first_hidden,
        first_residual,
        next_norm=model.norm,
    )
    torch.testing.assert_close(hidden, torch.full((2, 2), 31, dtype=torch.bfloat16))
    torch.testing.assert_close(residual, hidden)

    set_profile(None)
    disabled_profile = model.tpsp
    model.tpsp = None
    model.tpsp_requested = True
    next_layer = nn.Module()
    next_layer.input_layernorm = Norm()
    model.layers.append(next_layer)
    calls = []

    def profile_once(*args):
        calls.append(args)
        return disabled_profile

    monkeypatch.setattr(llama, "profile_tpsp", profile_once)
    for _ in range(2):
        result = model.forward(
            None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(result, torch.full((2, 2), 7, dtype=torch.bfloat16))
    assert calls == [
        (
            layer.self_attn.o_proj,
            layer.post_attention_layernorm,
            layer.mlp.down_proj,
            next_layer.input_layernorm,
            8,
        )
    ]

    enabled_profile = TPSPProfile(True, 1, 8, o_plan, down_plan)
    model.tpsp = None
    monkeypatch.setattr(llama, "profile_tpsp", lambda *args: enabled_profile)
    model.forward(
        None, None, None, inputs_embeds=torch.ones(2, 2, dtype=torch.bfloat16)
    )
    assert layer.tpsp is enabled_profile


def test_llama_tpsp_profiles_next_layer_norm(monkeypatch):
    first = nn.Module()
    first.self_attn = nn.Module()
    first.self_attn.o_proj = nn.Module()
    first.post_attention_layernorm = nn.Module()
    first.mlp = nn.Module()
    first.mlp.down_proj = nn.Module()
    second = nn.Module()
    second.input_layernorm = nn.Module()

    model = llama.LlamaModel.__new__(llama.LlamaModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([first, second])
    model.norm = nn.Module()
    model.tpsp_requested = True
    model.tpsp = None
    model.max_tpsp_batched_tokens = 128

    def check_profile(*args):
        assert args[3] is second.input_layernorm
        raise RuntimeError("profile called")

    monkeypatch.setattr(llama, "profile_tpsp", check_profile)
    with pytest.raises(RuntimeError, match="profile called"):
        model.forward(None, None, None)


def test_tpsp_projection_profile_keeps_distinct_configs(monkeypatch):
    class Projection(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.input_size_per_partition = width
            self.weight = nn.Parameter(torch.ones(2, width))

    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = Projection(2)
    layer.post_attention_layernorm = nn.Module()
    layer.post_attention_layernorm.weight = nn.Parameter(torch.ones(2))
    layer.post_attention_layernorm.variance_epsilon = 1e-5
    layer.mlp = nn.Module()
    layer.mlp.down_proj = Projection(4)

    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    model.norm = nn.Module()
    model.norm.weight = nn.Parameter(torch.ones(2))
    model.norm.variance_epsilon = 1e-5
    model.config = SimpleNamespace(hidden_size=2)

    class Backend:
        def open(self, **kwargs):
            return object()

        def profile(self, *, input_width, config_only, **kwargs):
            assert config_only
            return SPProfile(
                2,
                2,
                8,
                "candidate",
                "",
                input_width=input_width,
                config=input_width * 8,
            )

    backend = Backend()
    monkeypatch.setattr(tpsp_profile, "get_tpsp_backend", lambda *args: backend)
    monkeypatch.setattr(
        tpsp_profile,
        "profile_tpsp_projections",
        lambda o, o_proj, o_norm, down, down_proj, down_norm, tokens: SPProfile(
            2, 2, 8, "enabled", "", threshold_tokens=3
        ),
    )
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=2, device_group=SimpleNamespace(group_name="test")
        ),
    )

    profile = profile_tpsp(
        layer.self_attn.o_proj,
        layer.post_attention_layernorm,
        layer.mlp.down_proj,
        model.norm,
        8,
    )
    assert profile.enabled and profile.threshold_tokens == 3
    assert (profile.o_proj.config, profile.down_proj.config) == (16, 32)
    assert profile.o_proj.context is not profile.down_proj.context
    assert profile.o_proj.backend is profile.down_proj.backend is backend
