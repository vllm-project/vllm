# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import replace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
    _expected_m_with_actual_floor,
)
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
    BatchedPrepareAndFinalize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    is_deep_gemm_supported,
    supports_deep_gemm_batch_invariance,
)

from .test_deepgemm import make_block_quant_fp8_weights
from .utils import make_dummy_moe_config

BLOCK_SIZE = [128, 128]


def test_expected_m_covers_skewed_live_expert_count():
    counts = torch.tensor([58, 7, 0], dtype=torch.int32)
    assert _expected_m_with_actual_floor(16, counts) == 64
    assert _expected_m_with_actual_floor(128, counts) == 128


@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
@pytest.mark.parametrize("E", [16, 32])  # number of experts
@pytest.mark.parametrize("T", [256, 512])  # tokens per expert
@pytest.mark.parametrize("K", [128, 256])  # hidden dim
@pytest.mark.parametrize("N", [512, 1024])  # intermediate dim per expert
@pytest.mark.parametrize("topk", [2, 4])
def test_batched_deepgemm_vs_triton(
    E: int, T: int, K: int, N: int, topk: int, monkeypatch, workspace_init
):
    """Compare BatchedDeepGemmExperts to BatchedTritonExperts."""

    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")

    device = "cuda"
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(E, N, K, BLOCK_SIZE)

    M = E * T  # total tokens
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16) / 10.0
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    a.clamp_(fp8_info.min, fp8_info.max)

    # random router outputs → top-k indices / weights
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    # token number for each expert
    cnt = torch.bincount(topk_ids.flatten(), minlength=E)
    max_cnt = int(cnt.max().item())
    # next power of 2 for max token number
    max_num_tokens = 1 << (max_cnt - 1).bit_length()

    prep_finalize = BatchedPrepareAndFinalize(
        max_num_tokens=max_num_tokens,
        num_local_experts=E,
        num_dispatchers=1,
        rank=0,
    )

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        per_act_token_quant=False,
        block_shape=BLOCK_SIZE,
    )

    # triton (reference)
    triton_experts = BatchedTritonExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
        moe_config=make_dummy_moe_config(),
    )
    mk_triton = FusedMoEKernel(
        prep_finalize,
        triton_experts,
    )

    out_triton = mk_triton.apply(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    # deepgemm
    deepgemm_experts = BatchedDeepGemmExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
        moe_config=make_dummy_moe_config(),
    )
    mk_deepgemm = FusedMoEKernel(
        prep_finalize,
        deepgemm_experts,
    )

    out_deepgemm = mk_deepgemm.apply(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 1e-3, f"Output diff too large: {diff}"


@pytest.mark.skipif(
    not supports_deep_gemm_batch_invariance(),
    reason="Requires batch-invariant masked grouped DeepGEMM",
)
def test_deepep_ll_selects_batched_deepgemm_in_batch_invariant_mode(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config = make_dummy_moe_config(
        num_experts=8,
        num_local_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=256,
    )
    config.moe_backend = "deep_gemm"
    config.moe_parallel_config = replace(
        config.moe_parallel_config,
        dp_size=2,
        ep_size=2,
        use_ep=True,
        all2all_backend="deepep_low_latency",
    )

    backend, experts_cls = select_fp8_moe_backend(
        config,
        weight_key=kFp8Static128BlockSym,
        activation_key=kFp8Dynamic128Sym,
    )

    assert backend is Fp8MoeBackend.BATCHED_DEEPGEMM
    assert experts_cls is BatchedDeepGemmExperts


@pytest.mark.skipif(
    not supports_deep_gemm_batch_invariance(),
    reason="Requires batch-invariant masked grouped DeepGEMM",
)
def test_batched_deepgemm_needle_batch_invariance(workspace_init):
    """A routed token is bitwise stable across companions and expected_m."""

    import deep_gemm
    import vllm.envs as envs
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    assert BatchedDeepGemmExperts._supports_batch_invariance()
    E, K, N, topk = 8, 128, 256, 2
    max_num_tokens = 256
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(E, N, K, BLOCK_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(17)
    tokens = torch.randn(
        8, K, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    topk_ids = torch.randint(
        E, (8, topk), device="cuda", dtype=torch.int64, generator=generator
    )
    topk_weights = torch.randn(
        8, topk, device="cuda", dtype=torch.float32, generator=generator
    ).softmax(-1)

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        per_act_token_quant=False,
        block_shape=BLOCK_SIZE,
    )
    prep_finalize = BatchedPrepareAndFinalize(
        max_num_tokens=max_num_tokens,
        num_local_experts=E,
        num_dispatchers=1,
        rank=0,
    )
    experts = BatchedDeepGemmExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
        moe_config=make_dummy_moe_config(),
    )
    kernel = FusedMoEKernel(prep_finalize, experts)

    def run(indices: list[int], metadata_tokens: int) -> torch.Tensor:
        config = VllmConfig()
        config.parallel_config.data_parallel_size = 1
        with (
            set_current_vllm_config(config),
            set_forward_context(
                None,
                config,
                num_tokens=len(indices),
                num_tokens_across_dp=torch.tensor(
                    [metadata_tokens], dtype=torch.int, device="cpu"
                ),
            ),
        ):
            return kernel.apply(
                hidden_states=tokens[indices],
                w1=w1,
                w2=w2,
                topk_weights=topk_weights[indices],
                topk_ids=topk_ids[indices],
                activation=MoEActivation.SILU,
                global_num_experts=E,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

    # Exercise the public vLLM runtime entrypoint.  Individual models and
    # tests must not configure the DeepGEMM process-global mode themselves.
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    envs.VLLM_BATCH_INVARIANT = True
    init_batch_invariance()
    assert deep_gemm.get_batch_invariant()
    reference = run([0], 1)[0]
    variants = (
        run([0, 1, 2, 3, 4, 5, 6, 7], 256)[0],
        run([1, 2, 3, 4, 5, 6, 7, 0], 1024)[-1],
        run([1, 3, 0, 5], 64)[2],
    )

    for actual in variants:
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
