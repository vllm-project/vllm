# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import vllm.model_executor.layers.fused_moe  # noqa
from tests.kernels.moe.utils import fused_moe, make_dummy_moe_config
from tests.kernels.utils import opcheck, stack_and_dev, torch_experts, torch_moe
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import init_distributed_environment
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import (
    fused_topk,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    batched_fused_marlin_moe,
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    modular_triton_fused_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_permute_bias,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
    rand_marlin_weight_nvfp4_like,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize,
    marlin_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager


def iterative_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
    renormalize: bool = False,
) -> torch.Tensor:
    """
    Baseline implementation of fused moe.

    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
        expert_map: [num_experts]
    """
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = hidden_states.dtype

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, global_num_experts)
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights, selected_experts = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    if expert_map is not None:
        selected_experts = expert_map[selected_experts]

    final_hidden_states = None
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_mask = selected_experts == expert_idx
        expert_weights = (topk_weights * expert_mask).sum(dim=-1, keepdim=True)
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[:, :intermediate_size])
        x = x[:, intermediate_size:] * gate
        x = F.linear(x, expert_w2)
        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    return final_hidden_states.view(orig_shape)  # type: ignore


NUM_EXPERTS = [8, 64, 192]
NUM_EXPERTS_LARGE = [128, 256]
EP_SIZE = [1, 4]
TOP_KS = [2, 6]
TOP_KS_SMALL = [1, 2]

MOE_MARLIN_QUANT_TEST_CONFIGS = [
    # AWQ-INT4
    {"b_type": scalar_types.uint4, "group_blocks": [-1, 2, 4, 8]},
    # GPTQ-INT4
    {
        "b_type": scalar_types.uint4b8,
        "support_act_order": True,
        "group_blocks": [-1, 2, 4, 8],
    },
    # GPTQ-INT8
    {
        "b_type": scalar_types.uint8b128,
        "support_act_order": True,
        "group_blocks": [-1, 2, 4, 8],
    },
    # FP8
    {"b_type": scalar_types.float8_e4m3fn, "group_blocks": [-1, 8]},
    # NVFP4
    {"b_type": scalar_types.float4_e2m1f, "group_blocks": [1]},
    # MXFP4
    {
        "a_type": [scalar_types.bfloat16],
        "b_type": scalar_types.float4_e2m1f,
        "group_blocks": [2],
    },
    # AWQ-INT4 with INT8 activation
    {
        "a_type": [scalar_types.int8],
        "b_type": scalar_types.uint4,
        "group_blocks": [-1, 2, 4, 8],
    },
    # GPTQ-INT4 with INT8 activation
    {
        "a_type": [scalar_types.int8],
        "b_type": scalar_types.uint4b8,
        "group_blocks": [-1, 2, 4, 8],
    },
    # GPTQ-INT4 with FP8 activation
    {
        "a_type": [scalar_types.float8_e4m3fn],
        "b_type": scalar_types.uint4b8,
        "group_blocks": [-1, 2, 4, 8],
    },
    # AWQ-INT4 with FP8 activation
    {
        "a_type": [scalar_types.float8_e4m3fn],
        "b_type": scalar_types.uint4,
        "group_blocks": [-1, 2, 4, 8],
    },
    # MXFP4 with FP8 activation
    {
        "a_type": [scalar_types.float8_e4m3fn],
        "b_type": scalar_types.float4_e2m1f,
        "c_type": [scalar_types.bfloat16],
        "group_blocks": [2],
    },
]

FUSED_MOE_MNK_FACTORS = [
    (1, 128, 128),
    (1, 2048, 128),
    (33, 2048, 128),
    (32768, 2048, 511),
    (40000, 1024, 1024),
]

FUSED_MOE_MNK_FACTORS_SMALL_M = [
    (1, 128, 128),
    (1, 2048, 128),
    (2, 2048, 128),
    (2, 2048, 511),
]

FUSED_MOE_WN16_MNK_FACTORS = [
    (1, 128, 128),
    (1, 1024, 1024),
    (32, 2048, 128),
    (222, 2048, 1024),
]

vllm_config = VllmConfig()


def run_moe_test(
    baseline: Callable | torch.Tensor,
    moe_fn: Callable,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    padding: bool = False,
    use_compile: bool = False,
    use_cudagraph: bool = False,
    atol: float = 2e-2,
    rtol: float = 0,
) -> torch.Tensor:
    if isinstance(baseline, torch.Tensor):
        baseline_output = baseline
    else:
        baseline_output = baseline(
            a,
            w1,
            w2,
            score,
            topk,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

    # Pad the weight if moe padding is enabled
    if padding:
        w1 = F.pad(w1, (0, 128), "constant", 0)[..., 0:-128]
        w2 = F.pad(w2, (0, 128), "constant", 0)[..., 0:-128]

    if use_compile:
        moe_fn = torch.compile(moe_fn, backend="inductor", fullgraph=True)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(score, 0)

    test_output = moe_fn(
        a,
        w1,
        w2,
        score,
        topk,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )

    if use_cudagraph:
        test_output.fill_(0)
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            test_output = moe_fn(
                a,
                w1,
                w2,
                score,
                topk,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

    torch.testing.assert_close(test_output, baseline_output, atol=atol, rtol=rtol)

    return baseline_output


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("chunk_size", [8192])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    padding: bool,
    chunk_size: int,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)

    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", str(chunk_size))

    #
    # Setup test data
    #

    #
    # Setup test data
    #

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0, e, (local_e,), device="cuda", dtype=torch.int32)
        e_map = torch.full((e,), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    #
    # Setup test functions
    #
    quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    m_fused_moe_fn = modular_triton_fused_moe(make_dummy_moe_config(), quant_config)

    def m_fused_moe(
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        score: torch.Tensor,
        topk: int,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)
        return m_fused_moe_fn(
            a,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

    fused_moe_fn = functools.partial(fused_moe, renormalize=False)

    #
    # Run tests
    #
    runner = functools.partial(
        run_moe_test,
        a=a,
        w1=w1,
        w2=w2,
        score=score,
        topk=topk,
        global_num_experts=e,
        expert_map=e_map,
        padding=padding,
    )

    # Note: for now use_compile will error out if the problem size is
    # large enough to trigger chunking. I'm leaving the flag and
    # setup code in case we are able to revisit this later.
    use_compile = False

    use_cudagraph = n >= 1024 and k >= 1024 and current_platform.is_cuda_alike()

    with set_current_vllm_config(vllm_config):
        baseline_output = runner(torch_moe, iterative_moe)
        runner(
            baseline_output,
            fused_moe_fn,
            use_compile=use_compile,
            use_cudagraph=use_cudagraph,
        )
        runner(
            baseline_output,
            m_fused_moe,
            use_compile=use_compile,
            use_cudagraph=use_cudagraph,
        )


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS_SMALL_M)
@pytest.mark.parametrize("e", NUM_EXPERTS_LARGE)
@pytest.mark.parametrize("topk", TOP_KS_SMALL)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("chunk_size", [8192])
def test_naive_block_assignment_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    padding: bool,
    chunk_size: int,
    monkeypatch,
    workspace_init,
):
    current_platform.seed_everything(7)

    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", str(chunk_size))

    #
    # Setup test data
    #

    #
    # Setup test data
    #

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    e_map = None

    #
    # Setup test functions
    #
    quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    m_fused_moe_fn = modular_triton_fused_moe(make_dummy_moe_config(), quant_config)

    def m_fused_moe(
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        score: torch.Tensor,
        topk: int,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)
        return m_fused_moe_fn(
            a,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

    fused_moe_fn = functools.partial(fused_moe, renormalize=False)

    #
    # Run tests
    #
    runner = functools.partial(
        run_moe_test,
        a=a,
        w1=w1,
        w2=w2,
        score=score,
        topk=topk,
        global_num_experts=e,
        expert_map=e_map,
        padding=padding,
    )

    # Note: for now use_compile will error out if the problem size is
    # large enough to trigger chunking. I'm leaving the flag and
    # setup code in case we are able to revisit this later.
    use_compile = False

    use_cudagraph = n >= 1024 and k >= 1024 and current_platform.is_cuda_alike()

    with set_current_vllm_config(vllm_config):
        baseline_output = runner(torch_moe, iterative_moe)
        runner(
            baseline_output,
            fused_moe_fn,
            use_compile=use_compile,
            use_cudagraph=use_cudagraph,
        )
        runner(
            baseline_output,
            m_fused_moe,
            use_compile=use_compile,
            use_cudagraph=use_cudagraph,
        )


@pytest.mark.parametrize("m,n,k", FUSED_MOE_WN16_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("has_zp", [True, False])
@pytest.mark.parametrize("weight_bits", [4, 8])
def test_fused_moe_wn16(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    group_size: int,
    has_zp: bool,
    weight_bits: int,
):
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    elif weight_bits == 8:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    w1_qweight = torch.empty(
        (e, 2 * n, k // pack_factor), device="cuda", dtype=torch.uint8
    )
    w2_qweight = torch.empty((e, k, n // pack_factor), device="cuda", dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size), device="cuda", dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size), device="cuda", dtype=dtype)
    w1_qzeros = torch.empty(
        (e, 2 * n // pack_factor, k // group_size), device="cuda", dtype=torch.uint8
    )
    w2_qzeros = torch.empty(
        (e, k // pack_factor, n // group_size), device="cuda", dtype=torch.uint8
    )

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
            w, w_ref, w_qweight, w_scales, w_qzeros = (
                w1,
                w1_ref,
                w1_qweight,
                w1_scales,
                w1_qzeros,
            )
        else:
            w, w_ref, w_qweight, w_scales, w_qzeros = (
                w2,
                w2_ref,
                w2_qweight,
                w2_scales,
                w2_qzeros,
            )
        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False
        )
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T
        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.uint8)
        if weight_bits == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
            if has_zp:
                qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

        w_ref[expert_id] = weight
        w_qweight[expert_id] = qweight
        w_scales[expert_id] = scales
        if has_zp:
            w_qzeros[expert_id] = qzeros

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0, e, (local_e,), device="cuda", dtype=torch.int32)
        e_map = torch.full((e,), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1_ref = w1_ref[e_ids]
        w2_ref = w2_ref[e_ids]
        w1_qweight = w1_qweight[e_ids]
        w2_qweight = w2_qweight[e_ids]
        w1_scales = w1_scales[e_ids]
        w2_scales = w2_scales[e_ids]
        w1_qzeros = w1_qzeros[e_ids]
        w2_qzeros = w2_qzeros[e_ids]
    else:
        e_map = None

    if weight_bits == 4:
        quant_config_builder = int4_w4a16_moe_quant_config
    else:
        assert weight_bits == 8
        quant_config_builder = int8_w8a16_moe_quant_config

    quant_config = quant_config_builder(
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=w1_qzeros if has_zp else None,
        w2_zp=w2_qzeros if has_zp else None,
        block_shape=[0, group_size],
    )

    with set_current_vllm_config(vllm_config):
        triton_output = fused_moe(
            a,
            w1_qweight,
            w2_qweight,
            score,
            topk,
            renormalize=False,
            global_num_experts=e,
            expert_map=e_map,
            quant_config=quant_config,
        )
        torch_output = torch_moe(a, w1_ref, w2_ref, score, topk, expert_map=e_map)

    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
@torch.inference_mode()
def test_mixtral_moe(
    default_vllm_config,
    dist_init,
    dtype: torch.dtype,
    padding: bool,
    use_rocm_aiter: bool,
    monkeypatch,
):
    """Make sure our Mixtral MoE implementation agrees with the one from
    huggingface."""

    # Explicitly set AITER env var based on test parameter to ensure
    # consistent behavior regardless of external environment
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if use_rocm_aiter else "0")
    rocm_aiter_ops.refresh_env_variables()

    if use_rocm_aiter and dtype == torch.float32:
        pytest.skip("AITER ROCm test skip for float32")

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12345")
    init_distributed_environment()
    init_workspace_manager(torch.cuda.current_device())

    # Instantiate our and huggingface's MoE blocks
    vllm_config.compilation_config.static_forward_context = dict()
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        config = MixtralConfig()
        hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
        vllm_moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=dtype,
            tp_size=1,
            dp_size=1,
        ).cuda()

        # Load the weights
        vllm_moe.gate.weight.data[:] = hf_moe.gate.weight.data
        for i in range(config.num_local_experts):
            weights = (
                hf_moe.experts[i].w1.weight.data,
                hf_moe.experts[i].w3.weight.data,
            )
            vllm_moe.experts.w13_weight[i][:] = torch.cat(weights, dim=0)
            vllm_moe.experts.w2_weight[i][:] = hf_moe.experts[i].w2.weight.data

        # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
        hf_inputs = torch.randn((1, 64, config.hidden_size)).to(dtype).to("cuda")
        # vLLM uses 1D query [num_tokens, hidden_dim]
        vllm_inputs = hf_inputs.flatten(0, 1)

        # Pad the weight if moe padding is enabled
        if padding:
            vllm_moe.experts.w13_weight = Parameter(
                F.pad(vllm_moe.experts.w13_weight, (0, 128), "constant", 0)[
                    ..., 0:-128
                ],
                requires_grad=False,
            )
            vllm_moe.experts.w2_weight = Parameter(
                F.pad(vllm_moe.experts.w2_weight, (0, 128), "constant", 0)[..., 0:-128],
                requires_grad=False,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # FIXME (zyongye) fix this after we move self.kernel
        # assignment in FusedMoE.__init__

        vllm_moe.experts.quant_method.process_weights_after_loading(vllm_moe.experts)

        # Run forward passes for both MoE blocks
        hf_states, _ = hf_moe.forward(hf_inputs)
        vllm_states = vllm_moe.forward(vllm_inputs)

    mixtral_moe_tol = {
        torch.float32: 1e-3,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }

    if use_rocm_aiter:
        # The values of rtol and atol are set based on the tests in ROCM AITER package.
        # https://github.com/ROCm/aiter/blob/dfed377f4be7da96ca2d75ac0761f569676f7240/op_tests/test_moe.py#L174
        torch.testing.assert_close(
            hf_states.flatten(0, 1), vllm_states, rtol=0.01, atol=100
        )
    else:
        torch.testing.assert_close(
            hf_states.flatten(0, 1),
            vllm_states,
            rtol=mixtral_moe_tol[dtype],
            atol=mixtral_moe_tol[dtype],
        )


def marlin_moe_generate_valid_test_cases():
    import itertools

    m_list = [1, 123, 666]
    n_list = [128, 1024]
    k_list = [256, 2048]
    e_list = [5, 12]
    topk_list = [2, 3]
    ep_size_list = [1, 4]
    act_order_list = [True, False]
    is_k_full_list = [True, False]

    all_combinations = itertools.product(
        MOE_MARLIN_QUANT_TEST_CONFIGS,
        m_list,
        n_list,
        k_list,
        e_list,
        topk_list,
        ep_size_list,
        act_order_list,
        is_k_full_list,
    )

    def is_invalid(
        a_type,
        b_type,
        c_type,
        group_blocks,
        m,
        n,
        k,
        e,
        topk,
        ep_size,
        act_order,
        is_k_full,
    ):
        group_size = group_blocks if group_blocks <= 0 else group_blocks * 16
        if group_size > 0 and k % group_size != 0:
            return False

        if act_order and group_size in [-1, k, n]:
            return False
        if group_size in [k, n]:
            return False
        if not act_order and is_k_full:
            return False

        return a_type.size_bits < 16 or a_type is c_type

    cases = []
    for case in all_combinations:
        quant_test_config, m, n, k, _, _, _, act_order, *_ = case
        if act_order and not quant_test_config.get("support_act_order", False):
            continue

        f16_types = [scalar_types.float16]
        inner_combinations = itertools.product(
            quant_test_config.get("a_type", f16_types),
            [quant_test_config["b_type"]],
            quant_test_config.get("c_type", f16_types),
            quant_test_config["group_blocks"],
        )

        for sub_case in inner_combinations:
            if (
                sub_case[0] == scalar_types.float8_e4m3fn
                and current_platform.get_device_capability() not in [89, 120]
            ):
                continue
            args = sub_case + (m, n, k) + case[4:]
            if is_invalid(*args):
                cases.append(args)
    return cases


@dataclass
class MarlinMoEWeightData:
    w_ref: torch.Tensor
    qweight: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor | None
    a_scales_factor: torch.Tensor | None
    g_idx: torch.Tensor | None
    zeros: torch.Tensor | None
    sort_indices: torch.Tensor | None
    marlin_bias: torch.Tensor | None

    @staticmethod
    def make(
        w: torch.Tensor,
        quant_type: ScalarType,
        group_size: int,
        act_order: bool | None = None,
        bias: torch.Tensor | None = None,
        input_type: ScalarType = None,
    ) -> "MarlinMoEWeightData":
        assert w.ndim == 3

        has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]
        k = w.shape[-1]

        if input_type == scalar_types.int8:
            input_dtype = torch.int8
        elif input_type == scalar_types.float8_e4m3fn:
            input_dtype = torch.float8_e4m3fn
        else:
            input_dtype = w.dtype

        w_ref_l: list[torch.Tensor] = []
        qweight_l: list[torch.Tensor] = []
        scales_l: list[torch.Tensor] = []
        global_scale_l: list[torch.Tensor] = []
        zeros_l: list[torch.Tensor] = []
        g_idx_l: list[torch.Tensor] = []
        sort_indices_l: list[torch.Tensor] = []
        bias_l: list[torch.Tensor] = []

        for i in range(w.shape[0]):
            if quant_type == scalar_types.float4_e2m1f:
                if group_size == 16:
                    w_ref, qweight, scales, global_scale = (
                        rand_marlin_weight_nvfp4_like(
                            w[i], group_size, input_dtype=input_dtype
                        )
                    )
                else:
                    w_ref, qweight, scales = rand_marlin_weight_mxfp4_like(
                        w[i], group_size, input_dtype=input_dtype
                    )
                    global_scale = None

                w_ref_l.append(w_ref.T)
                qweight_l.append(qweight)
                scales_l.append(scales)
                if global_scale is not None:
                    global_scale_l.append(global_scale)
            elif quant_type == scalar_types.float8_e4m3fn:
                w_ref, qweight, scales = marlin_quant_fp8_torch(
                    w[i], group_size, input_dtype=input_dtype
                )
                w_ref_l.append(w_ref.T)
                qweight_l.append(qweight)
                scales_l.append(scales)
            elif has_zp:
                w_ref, qweight, scales, zeros = awq_marlin_quantize(
                    w[i].transpose(1, 0),
                    quant_type,
                    group_size,
                    input_dtype=input_dtype,
                )

                w_ref_l.append(w_ref.T)
                qweight_l.append(qweight)
                scales_l.append(scales)
                zeros_l.append(zeros)
            else:
                test_perm = torch.randperm(k)
                w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
                    w[i].transpose(1, 0),
                    quant_type,
                    group_size,
                    act_order,
                    test_perm,
                    input_dtype=input_dtype,
                )

                w_ref_l.append(w_ref.T)
                qweight_l.append(qweight)
                scales_l.append(scales)
                g_idx_l.append(g_idx)
                sort_indices_l.append(sort_indices)

            if bias is not None:
                bias_l.append(marlin_permute_bias(bias[i]))

        w_ref = stack_and_dev(w_ref_l)
        qweight = stack_and_dev(qweight_l).contiguous()
        scales = stack_and_dev(scales_l)
        global_scale = stack_and_dev(global_scale_l) if global_scale_l else None
        g_idx = stack_and_dev(g_idx_l) if g_idx_l else None
        zeros = stack_and_dev(zeros_l) if zeros_l else None
        sort_indices = stack_and_dev(sort_indices_l) if sort_indices_l else None
        marlin_bias = stack_and_dev(bias_l) if bias_l else None

        a_scales_factor = None
        if input_type == scalar_types.int8 and group_size != -1:
            a_scales_factor = 1 / 4096 * scales.max().float()
            scales = scales / scales.max() * 4096
            scales = scales.round().to(torch.int16).view(w.dtype)

        return MarlinMoEWeightData(
            w_ref=w_ref,
            qweight=qweight,
            scales=scales,
            global_scale=global_scale,
            a_scales_factor=a_scales_factor,
            g_idx=g_idx,
            zeros=zeros,
            sort_indices=sort_indices,
            marlin_bias=marlin_bias,
        )


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    (
        "a_type, b_type, c_type, group_blocks,"
        "m, n, k, e, topk, ep_size, act_order, is_k_full"
    ),
    marlin_moe_generate_valid_test_cases(),
)
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
def test_fused_marlin_moe(
    a_type,
    b_type,
    c_type,
    group_blocks,
    m,
    n,
    k,
    e,
    topk,
    ep_size,
    act_order,
    is_k_full,
):
    torch.cuda.manual_seed(1)
    group_size = group_blocks if group_blocks <= 0 else group_blocks * 16

    if c_type == scalar_types.float16:
        dtype = torch.float16
    elif c_type == scalar_types.bfloat16:
        dtype = torch.bfloat16
    else:
        raise RuntimeError("unsupported c_type")

    if a_type == scalar_types.int8:
        a_dtype = torch.int8
    elif a_type == scalar_types.float8_e4m3fn:
        a_dtype = torch.float8_e4m3fn
    else:
        a_dtype = dtype

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randperm(e, device="cuda", dtype=torch.int32)[:local_e]
        e_map = torch.full((e,), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    w1_data = MarlinMoEWeightData.make(
        w=w1,
        quant_type=b_type,
        group_size=group_size,
        act_order=act_order,
        input_type=a_type,
    )

    w2_data = MarlinMoEWeightData.make(
        w=w2,
        quant_type=b_type,
        group_size=group_size,
        act_order=act_order,
        input_type=a_type,
    )

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    with set_current_vllm_config(vllm_config):
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        torch_output = torch_experts(
            a,
            w1_data.w_ref,
            w2_data.w_ref,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            global_num_experts=e,
            expert_map=e_map,
            quant_dtype=a_dtype,
            per_act_token_quant=True,
        )

    marlin_output = fused_marlin_moe(
        a,
        w1_data.qweight,
        w2_data.qweight,
        None,
        None,
        w1_data.scales,
        w2_data.scales,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=e_map,
        global_scale1=w1_data.global_scale,
        global_scale2=w2_data.global_scale,
        g_idx1=w1_data.g_idx,
        g_idx2=w2_data.g_idx,
        input_global_scale1=w1_data.a_scales_factor,
        input_global_scale2=w2_data.a_scales_factor,
        sort_indices1=w1_data.sort_indices,
        sort_indices2=w2_data.sort_indices,
        w1_zeros=w1_data.zeros,
        w2_zeros=w2_data.zeros,
        input_dtype=a_dtype,
        quant_type_id=b_type.id,
        is_k_full=is_k_full,
    )

    torch.testing.assert_close(marlin_output, torch_output, atol=4e-2, rtol=0)


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
@pytest.mark.parametrize("m", [1, 256])
def test_fused_marlin_moe_with_bias(m):
    torch.cuda.manual_seed(0)

    e, topk = 32, 4
    n, k = 2048, 2048
    group_size = 128
    act_order = False
    is_k_full = True
    quant_type = scalar_types.uint4b8
    dtype = torch.half

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    b_bias1 = torch.randn((e, 2 * n), device="cuda", dtype=dtype) / 10
    b_bias2 = torch.randn((e, k), device="cuda", dtype=dtype) / 10

    w1_data = MarlinMoEWeightData.make(
        w=w1,
        quant_type=quant_type,
        group_size=group_size,
        act_order=act_order,
        bias=b_bias1,
    )

    w2_data = MarlinMoEWeightData.make(
        w=w2,
        quant_type=quant_type,
        group_size=group_size,
        act_order=act_order,
        bias=b_bias2,
    )

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    with set_current_vllm_config(vllm_config):
        torch_output = torch_moe(
            a, w1_data.w_ref, w2_data.w_ref, score, topk, b_bias1, b_bias2
        )

    marlin_output = fused_marlin_moe(
        a,
        w1_data.qweight,
        w2_data.qweight,
        w1_data.marlin_bias,
        w2_data.marlin_bias,
        w1_data.scales,
        w2_data.scales,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=None,
        global_scale1=w1_data.global_scale,
        global_scale2=w2_data.global_scale,
        g_idx1=w1_data.g_idx,
        g_idx2=w2_data.g_idx,
        sort_indices1=w1_data.sort_indices,
        sort_indices2=w2_data.sort_indices,
        w1_zeros=w1_data.zeros,
        w2_zeros=w2_data.zeros,
        quant_type_id=quant_type.id,
        is_k_full=is_k_full,
    )

    torch.testing.assert_close(marlin_output, torch_output, atol=5e-2, rtol=0)


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
@pytest.mark.parametrize("m", [1, 64, 256])
@pytest.mark.parametrize("n,k", [(1024, 1024), (2048, 2048)])
@pytest.mark.parametrize("e,topk", [(8, 2), (64, 4)])
def test_fused_marlin_moe_non_gated(m: int, n: int, k: int, e: int, topk: int):
    """Test Marlin MoE with non-gated activation (relu2_no_mul).

    Non-gated activations like relu2 don't have the gate-up projection pattern,
    so w1 has shape (e, n, k) instead of (e, 2*n, k).
    """
    torch.cuda.manual_seed(42)

    group_size = 16  # NVFP4 group size
    is_k_full = True
    quant_type = scalar_types.float4_e2m1f
    dtype = torch.bfloat16

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    # Non-gated: w1 shape is (e, n, k) not (e, 2*n, k)
    w1 = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    w1_data = MarlinMoEWeightData.make(
        w=w1,
        quant_type=quant_type,
        group_size=group_size,
        act_order=False,
    )

    w2_data = MarlinMoEWeightData.make(
        w=w2,
        quant_type=quant_type,
        group_size=group_size,
        act_order=False,
    )

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    with set_current_vllm_config(vllm_config):
        torch_output = torch_moe(
            a,
            w1_data.w_ref,
            w2_data.w_ref,
            score,
            topk,
            activation="relu2",
        )

    marlin_output = fused_marlin_moe(
        a,
        w1_data.qweight,
        w2_data.qweight,
        None,  # bias1
        None,  # bias2
        w1_data.scales,
        w2_data.scales,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=None,
        global_scale1=w1_data.global_scale,
        global_scale2=w2_data.global_scale,
        g_idx1=w1_data.g_idx,
        g_idx2=w2_data.g_idx,
        sort_indices1=w1_data.sort_indices,
        sort_indices2=w2_data.sort_indices,
        w1_zeros=w1_data.zeros,
        w2_zeros=w2_data.zeros,
        quant_type_id=quant_type.id,
        is_k_full=is_k_full,
        activation="relu2_no_mul",
    )

    torch.testing.assert_close(marlin_output, torch_output, atol=1e-1, rtol=0)


@pytest.mark.parametrize("ep_size", [1, 2])
def test_moe_align_block_size_opcheck(ep_size):
    num_experts = 4
    block_size = 4

    expert_map = None
    if ep_size != 1:
        local_num_experts = num_experts // ep_size
        expert_ids = torch.randint(
            0, num_experts, (local_num_experts,), device="cuda", dtype=torch.int32
        )
        expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
        expert_map[expert_ids] = torch.arange(
            local_num_experts, device="cuda", dtype=torch.int32
        )

    topk_ids = torch.randint(0, num_experts, (3, 4), dtype=torch.int32, device="cuda")

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    opcheck(
        torch.ops._moe_C.moe_align_block_size,
        (
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map,
        ),
    )


def test_batched_moe_align_block_size_opcheck():
    max_tokens_per_batch = 512
    num_experts = 4
    block_size = 16

    expert_num_tokens = torch.randint(
        low=0,
        high=max_tokens_per_batch,
        size=(num_experts,),
        dtype=torch.int32,
        device="cuda",
    )

    max_num_tokens_padded = num_experts * max(max_tokens_per_batch, block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")

    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device="cuda")

    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device="cuda")

    opcheck(
        torch.ops._moe_C.batched_moe_align_block_size,
        (
            max_tokens_per_batch,
            block_size,
            expert_num_tokens,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        ),
    )


@pytest.mark.parametrize("m", [1, 33, 222])
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_sum(m: int, topk: int, k: int, dtype: torch.dtype):
    input = torch.randn((m, topk, k), device="cuda", dtype=dtype)
    actual = torch.empty((m, k), device="cuda", dtype=dtype)

    expected = input.sum(dim=1)
    torch.ops._moe_C.moe_sum(input, actual)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    opcheck(torch.ops._moe_C.moe_sum, (input, actual))


@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("n,k", [(128, 128)])
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("activation", ["silu"])
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only test")
def test_cpu_fused_moe_basic(m, n, k, e, topk, dtype, with_bias, activation):
    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import CPUFusedMOE

    device = "cpu"
    torch.manual_seed(7)

    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w13 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10
    router_logits = torch.randn((m, e), device=device, dtype=dtype)

    b1 = b2 = None
    if with_bias:
        b1 = torch.randn((e, 2 * n), device=device, dtype=dtype) / 10
        b2 = torch.randn((e, k), device=device, dtype=dtype) / 10

    ref = (
        torch_moe(a, w13, w2, router_logits, topk, b1, b2)
        if with_bias
        else torch_moe(a, w13, w2, router_logits, topk)
    )

    class _Dummy(torch.nn.Module):
        def __init__(self, w13, w2, b1=None, b2=None):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
            self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
            if b1 is not None:
                self.w13_bias = torch.nn.Parameter(b1, requires_grad=False)
            if b2 is not None:
                self.w2_bias = torch.nn.Parameter(b2, requires_grad=False)

    layer = _Dummy(w13, w2, b1, b2).to(dtype)
    fused = CPUFusedMOE(layer)
    out = fused(
        layer=layer,
        x=a,
        use_grouped_topk=False,
        top_k=topk,
        router_logits=router_logits,
        renormalize=False,
        global_num_experts=e,
        expert_map=None,
        custom_routing_function=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        apply_router_weight_on_input=False,
        activation=activation,
    )

    # Tolerances: fp32 tight; bf16 looser (esp. with bias)
    if dtype == torch.float32:
        atol = 1e-3
    elif with_bias:
        atol = 8e-2
    else:
        atol = 5e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=0)


@pytest.mark.parametrize("m", [16, 32, 64])
@pytest.mark.parametrize("n", [128])
@pytest.mark.parametrize("k", [128])
@pytest.mark.parametrize("e", [8, 12, 16, 32])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("max_tokens_per_batch", [16, 32, 64])
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
def test_batched_fused_marlin_moe(
    m: int, n: int, k: int, e: int, topk: int, max_tokens_per_batch: int
):
    print(
        f"testing m={m}, n={n}, k={k}, e={e}, "
        f"topk={topk}, "
        f"max_tokens_per_batch={max_tokens_per_batch}"
    )
    torch.cuda.manual_seed(0)

    dtype = torch.bfloat16
    quant_dtype = scalar_types.float4_e2m1f
    group_size = 32

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20

    w1_data = MarlinMoEWeightData.make(
        w=w1, quant_type=quant_dtype, group_size=group_size, act_order=None
    )
    w2_data = MarlinMoEWeightData.make(
        w=w2, quant_type=quant_dtype, group_size=group_size, act_order=None
    )

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    class BatchedRun:
        @staticmethod
        def _make_expert_num_tokens_cpu(
            e: int,  # num_experts
            topk_ids_cpu: torch.Tensor,
        ) -> torch.Tensor:
            expert_num_tokens_cpu = torch.zeros((e,), dtype=torch.int32, device="cpu")
            for topk_id in torch.flatten(topk_ids_cpu):
                expert_num_tokens_cpu[topk_id] += 1
            return expert_num_tokens_cpu

        def __init__(
            self,
            max_tokens_per_batch: int,
            num_experts: int,
            _topk_ids: torch.Tensor,
            _topk_weights: torch.Tensor,
        ):
            self.max_tokens_per_batch = max_tokens_per_batch
            self.e = num_experts
            self.topk_ids_cpu = _topk_ids.to("cpu")
            self.topk_weights_cpu = _topk_weights.to("cpu")
            self.expert_num_tokens_cpu = self._make_expert_num_tokens_cpu(
                self.e, self.topk_ids_cpu
            )

        def is_valid(self):
            """
            Return True only if the input can be represented in a Batched
            format.
            """
            return torch.all(self.expert_num_tokens_cpu <= self.max_tokens_per_batch)

        def _scatter(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states_cpu = hidden_states.to("cpu")
            K = hidden_states_cpu.size(1)
            batched_hidden_states_cpu = torch.empty(
                (e, max_tokens_per_batch, K),
                dtype=hidden_states_cpu.dtype,
                device="cpu",
            )

            counter_cpu = torch.zeros_like(self.expert_num_tokens_cpu)
            for t_idx, token in enumerate(hidden_states_cpu):
                for topk_id in self.topk_ids_cpu[t_idx]:
                    pos_in_batch = counter_cpu[topk_id]
                    batched_hidden_states_cpu[topk_id, pos_in_batch] = token
                    counter_cpu[topk_id] += 1
            assert torch.allclose(counter_cpu, self.expert_num_tokens_cpu)
            return batched_hidden_states_cpu.to("cuda")

        def _gather(
            self, batched_outputs: torch.Tensor, gather_outputs: torch.Tensor
        ) -> torch.Tensor:
            batched_outputs_cpu = batched_outputs.to("cpu")
            gather_outputs_cpu = torch.zeros_like(gather_outputs)

            counter_cpu = torch.zeros((e,), device="cpu", dtype=torch.int32)
            md = gather_outputs_cpu.size(0)
            for t_idx in range(md):
                token = None
                for topk_id, topk_weight in zip(
                    self.topk_ids_cpu[t_idx], self.topk_weights_cpu[t_idx]
                ):
                    pos_in_batch = counter_cpu[topk_id]
                    t = batched_outputs_cpu[topk_id, pos_in_batch] * topk_weight
                    if token is None:
                        token = t
                    else:
                        token += t
                    counter_cpu[topk_id] += 1
                assert token is not None
                gather_outputs_cpu[t_idx] = token
            gather_outputs.copy_(gather_outputs_cpu)
            return gather_outputs

        def run(
            self, hidden_states: torch.Tensor, fused_marlin_moe_kwargs: dict[Any, Any]
        ) -> torch.Tensor:
            assert hidden_states.ndim == 2
            assert self.is_valid()

            batched_hidden_states = self._scatter(hidden_states)

            kwargs = fused_marlin_moe_kwargs | {
                "hidden_states": batched_hidden_states,
                "expert_num_tokens": self.expert_num_tokens_cpu.to("cuda"),
            }
            batched_outputs = batched_fused_marlin_moe(**kwargs)

            output = torch.zeros_like(hidden_states)
            output = self._gather(batched_outputs, output)
            return output

    kwargs = {
        "w1": w1_data.qweight,
        "w2": w2_data.qweight,
        "bias1": None,
        "bias2": None,
        "w1_scale": w1_data.scales,
        "w2_scale": w2_data.scales,
        "gating_output": score,
        "global_num_experts": e,
        "expert_map": None,
        "global_scale1": w1_data.global_scale,
        "global_scale2": w2_data.global_scale,
        "g_idx1": w1_data.g_idx,
        "g_idx2": w2_data.g_idx,
        "sort_indices1": w1_data.sort_indices,
        "sort_indices2": w2_data.sort_indices,
        "w1_zeros": w1_data.zeros,
        "w2_zeros": w2_data.zeros,
        "quant_type_id": quant_dtype.id,
        "is_k_full": True,
    }

    # Reference
    fused_marlin_moe_kwargs = kwargs | {
        "hidden_states": a,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }
    ref_marlin_output = fused_marlin_moe(**fused_marlin_moe_kwargs)

    # Batched
    br = BatchedRun(max_tokens_per_batch, e, topk_ids, topk_weights)
    if not br.is_valid():
        pytest.skip("Cannot represent data in Batched Format.")
    marlin_output = br.run(a, kwargs)

    torch.testing.assert_close(marlin_output, ref_marlin_output, atol=1e-3, rtol=0)
