# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-M3 low-batch FFN (``specialized_decode/moe``) against the stock
code it replaces, on random weights in the checkpoint's formats (NVFP4 routed
experts, compressed-tensors; MXFP8 shared expert; FP32 router): the stock
TRT-LLM-gen experts' call stopped after GEMM2 and the stock shared MLP."""

import os
import tempfile
from contextlib import suppress

import pytest
import torch
from transformers import PretrainedConfig

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
    hf = PretrainedConfig(
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
    try:
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
            torch.set_default_dtype(torch.bfloat16)
            moe = MiniMaxM3MoE(
                hf, 3, quant_config=quant, reduce_results=False, prefix="layers.3.moe"
            ).to(DEVICE)
            torch.set_default_dtype(default_dtype)
            _fill(moe, torch.Generator(device=DEVICE).manual_seed(0))
            for module in moe.modules():
                method = getattr(module, "quant_method", None)
                if isinstance(method, QuantizeMethodBase):
                    method.process_weights_after_loading(module)
            yield moe, config, build_low_batch_ffn(moe)
    finally:
        torch.set_default_dtype(default_dtype)
        cleanup_dist_env_and_memory()
        with suppress(FileNotFoundError):
            os.remove(path)


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
    """BF16 rows routed in one of three ways, and the router's bias and gate
    weight to restore afterwards."""
    x = torch.randn(rows, HIDDEN, generator=generator, device=DEVICE) * 2
    saved = (moe.e_score_correction_bias.data.clone(), moe.gate.weight.data.clone())
    if kind == "one_expert_set":
        # Every row on the same four experts: full 8-row tiles at 8 rows,
        # several tiles per expert above.
        moe.e_score_correction_bias.data[5:9] += 20
    elif kind == "distinct":
        # Row r on experts 4r .. 4r + 3: four tiles per row (all 128 experts
        # at 32 rows).
        moe.e_score_correction_bias.data.zero_()
        x.mul_(0.01)
        x[:, :rows] = 0
        x[torch.arange(rows), torch.arange(rows)] = 1
        gate = moe.gate.weight.data
        gate.zero_()
        gate[torch.arange(EXPERTS), torch.arange(EXPERTS) // 4] = 8
    elif rows > 1:
        x[1] = 0  # a zero row
    return x.to(torch.bfloat16), saved


def _stock_routed(moe, x):
    """``TrtLlmNvFp4ExpertsMonolithic.apply``'s FlashInfer call on ``x``,
    stopped after GEMM2 (the deferred finalize the low-batch FFN replaces)."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    from vllm.model_executor.layers.fused_moe.moe_output import (
        convert_flashinfer_moe_output,
    )
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
        quantize_nvfp4_per_token_input,
    )

    routed = moe.experts.routed_experts
    impl = routed.quant_method.moe_kernel.fused_experts
    quant = impl.quant_config
    logits, _ = moe.gate(x)
    values, scales, per_token = quantize_nvfp4_per_token_input(x)
    result = trtllm_fp4_block_scale_moe(
        routing_logits=logits,
        routing_bias=moe.e_score_correction_bias,
        hidden_states=values,
        hidden_states_scale=scales.view(torch.float8_e4m3fn).reshape(x.shape[0], -1),
        gemm1_weights=routed.w13_weight,
        gemm1_weights_scale=quant.w1_scale.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=impl.gemm1_alpha,
        gemm1_beta=impl.gemm1_beta,
        gemm1_clamp_limit=impl.gemm1_clamp_limit,
        gemm2_weights=routed.w2_weight,
        gemm2_weights_scale=quant.w2_scale.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=impl.g1_scale_c,
        output1_scale_gate_scalar=quant.g1_alphas,
        output2_scale_scalar=quant.g2_alphas,
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=moe.routed_scaling_factor,
        routing_method_type=impl.routing_method_type,
        do_finalize=False,
        activation_type=activation_to_flashinfer_int(impl.moe_config.activation),
        per_token_scale=per_token,
        tune_max_num_tokens=fi_moe_largest_bucket(impl.moe_config),
    )
    return convert_flashinfer_moe_output(
        result, do_finalize=False, num_tokens=x.shape[0], top_k=TOP_K
    )


def _final(routed, shared):
    """FP32 rows: the weighted routed rows plus the shared output."""
    rows = routed.expanded_idx_to_permuted_idx.long()
    picked = routed.gemm2_permuted[rows.view(-1)].float().view(*rows.shape, HIDDEN)
    weights = routed.expert_weights.float().unsqueeze(-1)
    return (picked * weights).sum(dim=1) + shared.float()


def _check(out, moe, x):
    """The same routing weights (to one BF16 ulp) and outputs within the
    rounding of two NVFP4 / MXFP8 chains. They are mostly bitwise equal, but
    the GEMMs accumulate in different orders, so a BF16 or E2M1 rounding can
    flip, which moves a row by about a percent; a wrong expert, scale,
    clamp or row moves it by far more."""
    stock = _stock_routed(moe, x)
    stock_shared = moe.shared_experts(x)
    mine = out.routed
    assert mine.expert_weights.dtype == torch.bfloat16
    assert not mine.route_major
    torch.testing.assert_close(
        mine.expert_weights.float().sort(dim=-1).values,
        stock.expert_weights.float().sort(dim=-1).values,
        rtol=0,
        atol=2**-7,
    )
    final = _final(mine, out.shared_output)
    expected = _final(stock, stock_shared)
    for row in range(x.shape[0]):
        if not bool(x[row].any()):
            assert not bool(final[row].any()), "a zero row"
            continue
        for name, got, want in (
            ("shared", out.shared_output[row].float(), stock_shared[row].float()),
            ("final", final[row], expected[row]),
        ):
            error = float((got - want).norm() / want.norm())
            assert error < 0.02, f"row {row} {name}: relative error {error:.4f}"


def test_low_batch_ffn_builds(layer):
    """The kernels subclass and call the DeepSeek-V4.1 low-batch MoE's
    (``deepseek_v41/nvidia/specialized_decode/moe/``) and compile for every
    width when the layer is bound, so a change there that breaks them fails
    here."""
    _, _, ffn = layer
    assert ffn is not None


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("kind", ["random", "one_expert_set", "distinct"])
@pytest.mark.parametrize("rows", [1, 3, 8, 12, 16, 24, 32])
def test_low_batch_ffn_matches_stock(layer, rows, kind, graph):
    """Eager calls and CUDA graph replays with new values in the captured
    input, at each of the front's widths (8, 16, 32 rows)."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    moe, config, ffn = layer
    assert ffn is not None
    generator = torch.Generator(device=DEVICE).manual_seed(rows)
    x, (bias, gate) = _inputs(kind, rows, generator, moe)
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
            torch.cuda.synchronize()
            _check(out, moe, x)
    finally:
        moe.e_score_correction_bias.data.copy_(bias)
        moe.gate.weight.data.copy_(gate)
