# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-M3 low-batch FFN (``specialized_decode/moe``) against the stock
MoE layer on random weights in the checkpoint's formats: NVFP4 routed experts
(compressed-tensors), MXFP8 shared expert, FP32 router."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="the low-batch FFN kernels run on SM100-family GPUs",
)

HIDDEN, EXPERTS, INTERMEDIATE, TOP_K = 6144, 128, 768, 4
DEVICE = "cuda"


@pytest.fixture(scope="module")
def layer():
    """One stock ``MiniMaxM3MoE`` with processed random weights, its config
    and its low-batch FFN."""
    from vllm.config import (
        KernelConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.models.minimax_m3.nvidia.model import MiniMaxM3MoE
    from vllm.models.minimax_m3.nvidia.specialized_decode.moe import (
        build_low_batch_ffn,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    quant = ModelOptMxFp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "weight_block_size": [1, 32],
            "activation_scheme": "dynamic",
            "moe_quant_algo": "NVFP4",
            "moe_quant_format": "nvfp4-pack-quantized",
            "group_size": 16,
            "ignored_layers": [],
        }
    )
    hf = SimpleNamespace(
        hidden_size=HIDDEN,
        num_local_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        intermediate_size=INTERMEDIATE,
        scoring_func="sigmoid",
        hidden_act="swigluoai",
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        use_routing_bias=True,
        routed_scaling_factor=2.0,
        n_shared_experts=1,
        num_hidden_layers=1,
    )
    config = VllmConfig(parallel_config=ParallelConfig(), kernel_config=KernelConfig())
    fd, path = tempfile.mkstemp()
    os.close(fd)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{path}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        init_workspace_manager(torch.device(DEVICE))
        moe = MiniMaxM3MoE(
            hf, 3, quant_config=quant, reduce_results=False, prefix="layers.3.moe"
        ).to(DEVICE)
        torch.set_default_dtype(default_dtype)
        _fill(moe, torch.Generator(device=DEVICE).manual_seed(0))
        for module in moe.modules():
            method = getattr(module, "quant_method", None)
            if isinstance(method, QuantizeMethodBase):
                method.process_weights_after_loading(module)
        ffn = build_low_batch_ffn(moe)
        assert ffn is not None
        yield moe, config, ffn
    cleanup_dist_env_and_memory()


def _fill(moe, generator):
    """Random weights of a checkpoint's magnitudes."""

    def randint(low, high, tensor):
        tensor.copy_(
            torch.randint(low, high, tensor.shape, generator=generator, device=DEVICE)
            .to(torch.uint8)
            .view(tensor.dtype)
        )

    routed = moe.experts.routed_experts
    for name in ("w13", "w2"):
        randint(0, 256, getattr(routed, f"{name}_weight_packed").data)
        # E4M3 scales between 0.5 and 4.
        randint(0x30, 0x48, getattr(routed, f"{name}_weight_scale").data)
        getattr(routed, f"{name}_weight_global_scale").data.fill_(1.5e4)
        getattr(routed, f"{name}_input_global_scale").data.fill_(1.0)
    moe.gate.weight.data.normal_(0, 0.02, generator=generator)
    moe.e_score_correction_bias.data.uniform_(0, 2, generator=generator)
    for linear in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
        values = torch.randn(linear.weight.shape, generator=generator, device=DEVICE)
        linear.weight.data.copy_((values * 64).clamp(-448, 448).to(torch.float8_e4m3fn))
        randint(114, 118, linear.weight_scale.data)


def _inputs(kind, rows, generator, moe):
    """BF16 rows routed in one of three ways (the bias is restored after)."""
    x = torch.randn(rows, HIDDEN, generator=generator, device=DEVICE) * 2
    bias = moe.e_score_correction_bias.data.clone()
    if kind == "one_expert_set":
        # Every row on the same four experts: full 8-row tiles at 8 rows.
        moe.e_score_correction_bias.data[5:9] += 20
    elif kind == "distinct":
        # Row r on experts 4r .. 4r + 3: four tiles per row.
        moe.e_score_correction_bias.data.zero_()
        x.mul_(0.01)
        x[:, :rows] = 0
        x[torch.arange(rows), torch.arange(rows)] = 1
        gate = moe.gate.weight.data
        saved = gate.clone()
        gate.zero_()
        gate[torch.arange(EXPERTS), torch.arange(EXPERTS) // 4] = 8
        return x.to(torch.bfloat16), (bias, saved)
    if rows > 1:
        x[1] = 0  # a zero row
    return x.to(torch.bfloat16), (bias, None)


def _restore(moe, saved):
    bias, gate = saved
    moe.e_score_correction_bias.data.copy_(bias)
    if gate is not None:
        moe.gate.weight.data.copy_(gate)


def _final(out):
    """FP32 rows of a MoEOutput: weighted routed rows plus the shared expert."""
    routed = out.routed
    rows = routed.expanded_idx_to_permuted_idx.long()
    picked = routed.gemm2_permuted[rows.view(-1)].float().view(*rows.shape, HIDDEN)
    weights = routed.expert_weights.float().unsqueeze(-1)
    return (picked * weights).sum(dim=1) + out.shared_output.float()


def _check(out, stock, x):
    """Within the NVFP4 rounding of the stock path."""
    assert out.routed.expert_weights.dtype == torch.bfloat16
    assert out.shared_output.dtype == torch.bfloat16
    assert bool((out.routed.expanded_idx_to_permuted_idx >= 0).all())
    final = _final(out)
    stock = stock.float()
    for row in range(x.shape[0]):
        if not bool(x[row].any()):
            assert torch.equal(final[row], stock[row]), "a zero row"
            continue
        error = (final[row] - stock[row]).norm() / stock[row].norm()
        assert error < 0.03, f"row {row}: relative error {error:.4f}"


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("kind", ["random", "one_expert_set", "distinct"])
@pytest.mark.parametrize("rows", [1, 3, 8])
def test_low_batch_ffn_matches_stock(layer, rows, kind, graph):
    """Eager calls and CUDA graph replays with new values in the captured
    input, against the stock layer's finalized output."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    moe, config, ffn = layer
    generator = torch.Generator(device=DEVICE).manual_seed(rows)
    x, saved = _inputs(kind, rows, generator, moe)
    try:
        with (
            set_current_vllm_config(config),
            set_forward_context(None, config),
            torch.inference_mode(),
        ):
            assert ffn.accepts(x)
            if not graph:
                out = ffn.run(x)
            else:
                static = torch.zeros_like(x)
                ffn.run(static)
                torch.cuda.synchronize()
                cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph):
                    out = ffn.run(static)
                static.copy_(x)
                cuda_graph.replay()
            stock = moe(x)
            torch.cuda.synchronize()
            _check(out, stock, x)
    finally:
        _restore(moe, saved)
