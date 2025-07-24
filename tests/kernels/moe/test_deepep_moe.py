# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test deepep dispatch-combine logic
"""

import dataclasses
from typing import Optional, Union

import pytest
import torch.distributed
from torch.distributed import ProcessGroup

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform
from vllm.utils import has_deep_ep

from .parallel_utils import ProcessGroupInfo, parallel_launch

if has_deep_ep():
    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (  # noqa: E501
        DeepEPHTPrepareAndFinalize)
    from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (  # noqa: E501
        DeepEPLLPrepareAndFinalize)

    from .parallel_utils import DeepEPHTArgs, DeepEPLLArgs, make_deepep_a2a

requires_deep_ep = pytest.mark.skipif(
    not has_deep_ep(),
    reason="Requires deep_ep kernels",
)

MAX_TOKENS_PER_RANK = 64


def make_weights(
        e, n, k, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2, w1_scale, w2_scale
    """
    if dtype in [torch.float16, torch.bfloat16]:
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        return w1, w2, None, None

    # per-out-channel weight quantization
    assert dtype == torch.float8_e4m3fn
    w1 = torch.empty((e, 2 * n, k), device="cuda", dtype=torch.float16)
    w2 = torch.empty((e, k, n), device="cuda", dtype=torch.float16)

    n_b_scales = 2 * n
    k_b_scales = k
    w1_q = torch.empty_like(w1, dtype=dtype)
    w2_q = torch.empty_like(w2, dtype=dtype)
    w1_scale = torch.empty((e, n_b_scales, 1),
                           device="cuda",
                           dtype=torch.float32)
    w2_scale = torch.empty((e, k_b_scales, 1),
                           device="cuda",
                           dtype=torch.float32)
    for expert in range(e):
        w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
            w1[expert], use_per_token_if_dynamic=True)
        w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
            w2[expert], use_per_token_if_dynamic=True)
    return w1_q, w2_q, w1_scale, w2_scale


@dataclasses.dataclass
class TestConfig:
    dtype: torch.dtype
    topk: int
    m: int
    k: int
    n: int
    num_experts: int


@dataclasses.dataclass
class TestTensors:
    rank_tokens: torch.Tensor  # all ranks make this many tokens
    rank_token_scales: Optional[torch.Tensor]
    topk: torch.Tensor
    topk_weights: torch.Tensor
    config: TestConfig

    @staticmethod
    def make(config: TestConfig, low_latency_mode: bool) -> "TestTensors":
        # TODO (varun) - check that float16 works ?
        assert config.dtype in [torch.bfloat16, torch.float8_e4m3fn]
        token_dtype = (torch.bfloat16 if config.dtype == torch.float8_e4m3fn
                       else config.dtype)
        rank_tokens = torch.randn(
            (config.m, config.k), device="cuda", dtype=token_dtype) / 10
        rank_token_scales = None

        topk = torch.randint(low=0,
                             high=config.num_experts,
                             size=(config.m, config.topk),
                             device="cuda").to(dtype=torch.int64)
        topk_weights = torch.randn(topk.shape,
                                   dtype=torch.float32,
                                   device="cuda")
        return TestTensors(rank_tokens=rank_tokens,
                           rank_token_scales=rank_token_scales,
                           topk=topk,
                           topk_weights=topk_weights,
                           config=config)


def make_modular_kernel(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    low_latency_mode: bool,
    hidden_size: int,
    dp_size: int,
    num_experts: int,
    num_local_experts: int,
    q_dtype: Optional[torch.dtype],
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
) -> FusedMoEModularKernel:

    is_quantized = q_dtype is not None

    ht_args: Optional[DeepEPHTArgs] = None
    ll_args: Optional[DeepEPLLArgs] = None

    if low_latency_mode:
        ll_args = DeepEPLLArgs(max_tokens_per_rank=MAX_TOKENS_PER_RANK,
                               hidden_size=hidden_size,
                               num_experts=num_experts,
                               use_fp8_dispatch=use_fp8_dispatch)
    else:
        assert not use_fp8_dispatch, (
            "FP8 Dispatch is valid only for low-latency kernels")
        ht_args = DeepEPHTArgs(num_local_experts=num_local_experts)

    a2a : Union[DeepEPHTPrepareAndFinalize, DeepEPLLPrepareAndFinalize] = \
        make_deepep_a2a(pg = pg,
                        pgi = pgi,
                        dp_size = dp_size,
                        q_dtype = q_dtype,
                        block_shape = None,
                        deepep_ht_args = ht_args,
                        deepep_ll_args = ll_args)

    num_dispatchers = pgi.world_size // dp_size

    if low_latency_mode:
        assert not per_act_token_quant, "not supported in ll mode"
        fused_experts = BatchedTritonExperts(
            max_num_tokens=MAX_TOKENS_PER_RANK,
            num_dispatchers=num_dispatchers,
            use_fp8_w8a8=is_quantized,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_act_token_quant=False,
        )
    else:
        fused_experts = TritonExperts(
            use_fp8_w8a8=is_quantized,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_act_token_quant=per_act_token_quant,
        )

    mk = FusedMoEModularKernel(prepare_finalize=a2a,
                               fused_experts=fused_experts)
    return mk


def deep_ep_moe_impl(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    low_latency_mode: bool,
    dp_size: int,
    test_tensors: TestTensors,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    num_experts: int,
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
) -> torch.Tensor:

    num_local_experts = w1.size(0)

    def build_expert_map():
        num_local_experts = w1.size(0)
        expert_map = torch.full((num_experts, ),
                                fill_value=-1,
                                dtype=torch.int32)
        s = pgi.rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
        return expert_map.to(device=torch.cuda.current_device(),
                             dtype=torch.int32)

    hidden_size = test_tensors.rank_tokens.size(1)
    is_quantized = w1.dtype == torch.float8_e4m3fn
    q_dtype = None
    if is_quantized:
        q_dtype = torch.float8_e4m3fn

    # Make modular kernel
    mk: FusedMoEModularKernel = make_modular_kernel(
        pg, pgi, low_latency_mode, hidden_size, dp_size, num_experts,
        num_local_experts, q_dtype, use_fp8_dispatch, per_act_token_quant)

    out_hidden_states = torch.empty_like(test_tensors.rank_tokens)
    total_num_tokens = test_tensors.rank_tokens.size(0)

    def process_chunk(chunk_start, chunk_end, skip_result_store=False):
        rank_tokens_chunk = test_tensors.rank_tokens[chunk_start:chunk_end]
        topk_weights_chunk = test_tensors.topk_weights[chunk_start:chunk_end]
        topk_chunk = test_tensors.topk[chunk_start:chunk_end]
        rank_token_scales_chunk = test_tensors.rank_token_scales
        if rank_token_scales_chunk is not None and rank_token_scales_chunk.size(
                0) == total_num_tokens:
            # per act token
            rank_token_scales_chunk = rank_token_scales_chunk[
                chunk_start:chunk_end]

        out = mk.forward(hidden_states=rank_tokens_chunk,
                         w1=w1,
                         w2=w2,
                         topk_weights=topk_weights_chunk,
                         topk_ids=topk_chunk,
                         inplace=False,
                         activation="silu",
                         global_num_experts=num_experts,
                         expert_map=build_expert_map(),
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         w1_zp=None,
                         w2_zp=None,
                         a1_scale=rank_token_scales_chunk,
                         a2_scale=None,
                         apply_router_weight_on_input=False)

        if not skip_result_store:
            out_hidden_states[chunk_start:chunk_end, :].copy_(
                out, non_blocking=True)

    max_num_tokens_per_dp = (MAX_TOKENS_PER_RANK
                             if low_latency_mode else total_num_tokens)

    for chunk_start_ in range(0, total_num_tokens, max_num_tokens_per_dp):
        chunk_start = chunk_start_
        chunk_end = min(chunk_start + max_num_tokens_per_dp, total_num_tokens)
        # clamp start and end
        chunk_start = min(chunk_start, total_num_tokens - 1)
        chunk_end = min(chunk_end, total_num_tokens)

        process_chunk(chunk_start,
                      chunk_end,
                      skip_result_store=chunk_start_ >= total_num_tokens)

    return out_hidden_states


def torch_moe_impl(
    test_tensors: TestTensors,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    using_fp8_dispatch: bool,
    per_act_token_quant: bool,
):

    a, topk_ids, topk_weights = (test_tensors.rank_tokens, test_tensors.topk,
                                 test_tensors.topk_weights)
    if using_fp8_dispatch:
        # The DeepEP implementation is requested to dispatch using FP8.
        # For numerical stability for testing, emulate the fp8 dispatch by
        # blockwise quant and de-quant.
        assert not per_act_token_quant
        a = test_tensors.rank_tokens
        aq, aq_scale = per_token_group_quant_fp8(a, 128)
        a = (aq.view(-1, 128).to(torch.float32) * aq_scale.view(-1, 1)).view(
            a.shape).to(a.dtype)

    is_quantized = w1.dtype == torch.float8_e4m3fn
    a_dtype = a.dtype
    if is_quantized:
        w1 = w1.to(dtype=torch.float32) * w1_scale
        w2 = w2.to(dtype=torch.float32) * w2_scale
        a = a.to(dtype=torch.float32)

    m, _ = a.shape
    topk = topk_ids.size(1)
    out = torch.zeros_like(a)

    for i in range(m):
        a_i = a[i]
        o_i = out[i]
        for j in range(topk):
            e = topk_ids[i][j]
            e_w = topk_weights[i][j]
            w1_e = w1[e]
            w2_e = w2[e]
            o_i += (SiluAndMul()
                    (a_i @ w1_e.transpose(0, 1)) @ w2_e.transpose(0, 1)) * e_w

    if is_quantized:
        out = out.to(dtype=a_dtype)

    return out


def _deep_ep_moe(
    pgi: ProcessGroupInfo,
    low_latency_mode: bool,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
):

    if not low_latency_mode:
        assert not use_fp8_dispatch, (
            "FP8 dispatch interface is available only in low-latency mode")

    is_quantized = w1.dtype == torch.float8_e4m3fn
    w1 = w1.to(device=torch.cuda.current_device())
    w2 = w2.to(device=torch.cuda.current_device())
    if is_quantized:
        w1_scale = w1_scale.to(  # type: ignore
            device=torch.cuda.current_device())
        w2_scale = w2_scale.to(  # type: ignore
            device=torch.cuda.current_device())

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(config, low_latency_mode)

    with set_current_vllm_config(VllmConfig()):
        # Reference
        torch_combined = torch_moe_impl(test_tensors, w1, w2, w1_scale,
                                        w2_scale, use_fp8_dispatch,
                                        per_act_token_quant)

        # Splice experts for this rank.
        num_local_experts = config.num_experts // pgi.world_size
        e_start = num_local_experts * pgi.rank
        e_end = e_start + num_local_experts
        w1_ep = w1[e_start:e_end]
        w2_ep = w2[e_start:e_end]

        w1_scale_ep, w2_scale_ep = None, None
        if is_quantized:
            w1_scale_ep = w1_scale[e_start:e_end]  # type: ignore
            w2_scale_ep = w2_scale[e_start:e_end]  # type: ignore
        deepep_combined = deep_ep_moe_impl(
            pg,
            pgi,
            low_latency_mode,
            dp_size,
            test_tensors,
            w1_ep,
            w2_ep,
            w1_scale_ep,
            w2_scale_ep,
            config.num_experts,
            use_fp8_dispatch,
            per_act_token_quant,
        )

    torch.testing.assert_close(
        torch_combined,
        deepep_combined,
        atol=6e-2,
        rtol=6e-2,
    )


MNKs = [
    (1, 128, 128),
    (2, 128, 512),
    (3, 1024, 2048),
    (32, 128, 1024),
    (45, 512, 2048),
    (64, 1024, 1024),
    (222, 1024, 2048),
]

DTYPES = [torch.bfloat16, torch.float8_e4m3fn]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@pytest.mark.parametrize("per_act_token_quant", [False, True])
@requires_deep_ep
def test_deep_ep_moe(
    dtype: torch.dtype,
    mnk: tuple[int, int, int],
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
    per_act_token_quant: bool,
):
    low_latency_mode = False
    use_fp8_dispatch = False
    m, n, k = mnk

    current_platform.seed_everything(7)
    world_size, dp_size = world_dp_size
    config = TestConfig(dtype=dtype,
                        topk=topk,
                        m=m,
                        k=k,
                        n=n,
                        num_experts=num_experts)

    w1, w2, w1_scale, w2_scale = make_weights(num_experts, n, k, dtype)

    parallel_launch(world_size, _deep_ep_moe, low_latency_mode, dp_size,
                    config, w1, w2, w1_scale, w2_scale, use_fp8_dispatch,
                    per_act_token_quant)


MNKs = [
    (1, 128, 2560),
    (2, 128, 2560),
    (3, 1024, 2560),
    (32, 128, 2560),
    (45, 512, 2560),
    (64, 1024, 2560),
    (222, 1024, 2560),
]
DTYPES = [torch.float8_e4m3fn, torch.bfloat16]
USE_FP8_DISPATCH = [True, False]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mnk", MNKs)
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@pytest.mark.parametrize("use_fp8_dispatch", USE_FP8_DISPATCH)
@requires_deep_ep
def test_low_latency_deep_ep_moe(dtype: torch.dtype, mnk: tuple[int, int, int],
                                 num_experts: int, topk: int,
                                 world_dp_size: tuple[int, int],
                                 use_fp8_dispatch: bool):

    low_latency_mode = True
    m, n, k = mnk

    if (low_latency_mode
            and k not in DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES):
        pytest.skip(
            f"Skipping test as hidden size {k} is not in list of supported "
            f"hidden sizes {DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES}"
        )

    current_platform.seed_everything(7)
    world_size, dp_size = world_dp_size
    config = TestConfig(dtype=dtype,
                        topk=topk,
                        m=m,
                        k=k,
                        n=n,
                        num_experts=num_experts)

    w1, w2, w1_scale, w2_scale = make_weights(num_experts, n, k, dtype)

    parallel_launch(world_size, _deep_ep_moe, low_latency_mode, dp_size,
                    config, w1, w2, w1_scale, w2_scale, use_fp8_dispatch,
                    False)
