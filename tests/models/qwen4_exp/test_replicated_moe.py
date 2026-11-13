# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of `--enable-hc-sp-replicated-moe` for the ROCm Qwen4Exp decoder.

Two ranks build the real model (one decoder layer), load its experts through
the model's `load_weights` and run the layer's forward over the real SP
collectives: on two GPUs where there are any, as in CI, else on CPU (gloo).
Replaced by torch versions, so that both devices run the same math: the
attention (a row-parallel projection), the HC Triton ops, the MoE backend
selection, the AITER top-k buffer (used only by the AITER kernel) and the
expert kernel (`MoERunner._forward_impl`, a dense torch MoE over the runner's
own weights).
"""

import zlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from transformers import Qwen4ExpTextConfig

import vllm.config.parallel as parallel_module
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    KernelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_tensor_model_parallel_rank,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import (
    expert_map_manager,
    unquantized_fused_moe_method,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.models.qwen4_exp.amd import hyperconnection as hc_module
from vllm.models.qwen4_exp.amd import model as model_module
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

H, HC, E, TOP_K, INTER = 128, 2, 4, 2, 256
T = 5  # odd: rank 1's SP shard ends in a pad row
FLAG = {"enable_hc_sp_replicated_moe": True}
ON_GPU = current_platform.is_cuda_alike()
pytestmark = pytest.mark.skipif(
    ON_GPU and torch.accelerator.device_count() < 2,
    reason="two ranks need two GPUs where any is visible",
)


class _RocmView:
    """The platform as `vllm.config.parallel` sees it: ROCm."""

    def __init__(self, platform) -> None:
        self._platform = platform

    def __getattr__(self, name):
        if name == "is_rocm":
            return lambda: True
        return getattr(self._platform, name)


def _make_config(*, monokernel: bool = False, **flags) -> VllmConfig:
    rocm = _RocmView(parallel_module.current_platform)
    with patch.object(parallel_module, "current_platform", rocm):
        config = VllmConfig(
            device_config=DeviceConfig(device="cuda" if ON_GPU else "cpu"),
            parallel_config=ParallelConfig(
                tensor_parallel_size=2, enable_hc_sp=True, **flags
            ),
            kernel_config=KernelConfig(monokernel=monokernel),
            compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        )
    config.model_config = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        is_moe=True,
        sleep_mode_offload_cudagraph=False,
        hf_text_config=Qwen4ExpTextConfig(
            vocab_size=128,
            eos_token_id=0,
            hidden_size=H,
            hc_count=HC,
            hc_lowrank=16,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            num_experts=E,
            num_experts_per_tok=TOP_K,
            moe_intermediate_size=INTER,
            shared_expert_intermediate_size=INTER,
            ple_layer_ids=[],
        ),
        dtype=torch.bfloat16,
    )
    return config


def _seeded(name: str, shape) -> torch.Tensor:
    generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
    return torch.randn(shape, generator=generator, device="cpu") * 0.08


# Torch versions of the HC Triton ops.
def _rmsnorm(x, weight, eps, groups):
    xg = x.float().unflatten(-1, (groups, -1))
    y = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    return (y + y * weight.float().view(groups, -1)).flatten(-2).to(x.dtype)


def _combine(residual, block, injection, hc_count):
    gain = 2 * torch.sigmoid(injection.float() / hc_count)
    stream = residual.float().unflatten(-1, (hc_count, -1))
    out = stream + block.float()[:, None, :] * gain[:, :, None]
    return out.flatten(-2).to(residual.dtype)


def _combine_norm(residual, block, injection, weight, eps, hc_count):
    out = _combine(residual, block, injection, hc_count)
    return out, _rmsnorm(out, weight, eps, hc_count)


def _silu(x, hc_count):
    x = x.float() / hc_count
    return (x * torch.sigmoid(x)).to(torch.bfloat16)


def _gate_mix(x, gate, hc_count):
    streams = x.float().unflatten(-1, (hc_count, -1))
    gates = torch.sigmoid(gate.float().unflatten(-1, (hc_count, -1)))
    return ((gates * streams).sum(-2) / hc_count).to(gate.dtype)


class _RowParallelAttention(nn.Module):
    """The attention's place in the layer: this rank's partial sum of a
    row-parallel projection, which the layer reduce-scatters."""

    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        half = self.weight.shape[1] // 2
        rank = get_tensor_model_parallel_rank()
        cols = slice(rank * half, (rank + 1) * half)
        return hidden_states[:, cols] @ self.weight[:, cols].T


def _dense_impl(self, hidden_states, router_logits, *args, **kwargs):
    """A dense MoE over this runner's weights: renormalized softmax top-k over
    the routed experts plus slot E, the fused shared expert, under its gate."""
    x = hidden_states.float()
    experts = self.routed_experts
    weights, ids = (x @ self.gate.weight.float().T).softmax(-1).topk(TOP_K)
    weights = weights / weights.sum(-1, keepdim=True)

    def expert(e):
        g, u = (x @ experts.w13_weight[e].float().T).chunk(2, dim=-1)
        return (F.silu(g) * u) @ experts.w2_weight[e].float().T

    out = torch.sigmoid(x @ self.shared_expert_gate.weight.float().T) * expert(E)
    for e in range(E):
        out = out + (weights * (ids == e)).sum(-1, keepdim=True) * expert(e)
    return out.to(hidden_states.dtype)


_CALLS = {"all_gather": 0, "reduce_scatter": 0, "sp_shard": 0}


def _counted(name, fn):
    def wrapper(*args, **kwargs):
        _CALLS[name] += 1
        return fn(*args, **kwargs)

    return wrapper


def _patches() -> None:
    for target, name, value in (
        (hc_module, "hc_silu", _silu),
        (hc_module, "hc_gate_mix", _gate_mix),
        (hc_module, "hc_combine", _combine),
        (hc_module, "hc_combine_norm", _combine_norm),
        (hc_module, "grouped_gemma_rmsnorm", _rmsnorm),
        (
            unquantized_fused_moe_method,
            "select_unquantized_moe_backend",
            lambda **kwargs: (UnquantizedMoeBackend.TRITON, None),
        ),
        (expert_map_manager, "init_aiter_topK_meta_data", lambda **kwargs: None),
        (rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: True),
        (model_module, "QwenGatedDeltaNetAttention", _RowParallelAttention),
        (MoERunner, "_forward_impl", _dense_impl),
        (model_module, "sp_all_gather", _counted("all_gather", sp_all_gather)),
        (
            model_module,
            "sp_reduce_scatter",
            _counted("reduce_scatter", model_module.sp_reduce_scatter),
        ),
        (model_module, "sp_shard", _counted("sp_shard", model_module.sp_shard)),
    ):
        patch.object(target, name, value).start()


def _checkpoint() -> dict[str, torch.Tensor]:
    """Layer 0's expert tensors as a checkpoint names them (relative to
    `Qwen4ExpModel`); the model loads the shared expert into slot E."""
    shapes = {"down_proj": (H, INTER), "gate_proj": (INTER, H), "up_proj": (INTER, H)}
    owners = [f"experts.{e}" for e in range(E)] + ["shared_expert"]
    return {
        f"layers.0.mlp.{owner}.{proj}.weight": _seeded(f"{owner}.{proj}", shape)
        for owner in owners
        for proj, shape in shapes.items()
    }


def _build(config: VllmConfig) -> nn.Module:
    """Layer 0 of the real model: the model's `load_weights` gives each rank its
    share of `_checkpoint` (all of each expert at TP size 1); every other weight
    is seeded by name."""
    with set_current_vllm_config(config):
        model = model_module.Qwen4ExpModel(vllm_config=config, prefix="model")
        for name, param in model.named_parameters():
            param.copy_(_seeded(name, param.shape))
        model.load_weights(iter(_checkpoint().items()))
    (layer,) = model.layers
    return layer


def _forward(layer: nn.Module, config: VllmConfig) -> tuple:
    """One forward on this rank's SP shard: the layer's collective calls and
    its gathered outputs."""
    hidden = _seeded("input", (T, HC * H))
    hidden = hidden.to(torch.get_default_device(), torch.bfloat16)
    _CALLS.update(dict.fromkeys(_CALLS, 0))
    with (
        set_current_vllm_config(config),
        set_forward_context(None, config, num_tokens=T),
    ):
        outputs = layer(
            sp_shard(hidden),
            None,
            None,
            torch.arange(T),
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
    calls = tuple(_CALLS.values())
    return calls, tuple(sp_all_gather(x.contiguous())[:T] for x in outputs)


def _check_replicated_experts() -> None:
    off_config, on_config = _make_config(), _make_config(**FLAG)
    off, on = _build(off_config), _build(on_config)
    assert off.mlp.experts.moe_config.tp_size == 2
    assert on.mlp.experts.moe_config.tp_size == 1
    _, want = _forward(off, off_config)
    calls, got = _forward(on, on_config)
    # The attention keeps its all-gather and reduce-scatter; the MoE has none.
    assert calls == (1, 1, 0)
    # TP2 rounds two bf16 partial sums, whole experts one: close, not equal.
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-2)


def _check_unsupported_refused() -> None:
    config = _make_config(**FLAG)
    with (
        patch.object(
            rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False
        ),
        set_current_vllm_config(config),
        pytest.raises(ValueError, match="needs the AITER fused shared expert"),
    ):
        model_module.Qwen4ExpDecoderLayer(config, "linear_attention", "model.layers.0")
    with pytest.raises(ValueError, match="does not support --monokernel"):
        model_module._replicated_moe(_make_config(monokernel=True, **FLAG))


def _run(rank: int, port: int, check) -> None:
    torch.set_num_threads(1)
    _patches()
    backend, device = ("nccl", f"cuda:{rank}") if ON_GPU else ("gloo", "cpu")
    if ON_GPU:
        torch.accelerator.set_device_index(rank)
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend=backend,
    )
    try:
        with set_current_vllm_config(_make_config()):
            initialize_model_parallel(tensor_model_parallel_size=2, backend=backend)
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device), torch.inference_mode():
            check()
    finally:
        cleanup_dist_env_and_memory()


def test_replicated_experts_run_whole_on_local_rows() -> None:
    """Each rank holds the whole experts and runs them on its own SP rows: the
    MoE seam loses its collectives and the layer's outputs match the TP-sharded
    layer's (a doubled or missing sum would not)."""
    mp.spawn(_run, args=(get_open_port(), _check_replicated_experts), nprocs=2)


def test_unsupported_layouts_are_refused() -> None:
    """A separate shared expert would stay TP-sharded and be added unreduced;
    `--monokernel`'s specialized decode would sum whole experts twice."""
    mp.spawn(_run, args=(get_open_port(), _check_unsupported_refused), nprocs=2)
