# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DeepGEMM bf16 (unquantized) MoE backend.

Compares the DeepGEMM bf16 experts against the Triton reference on a single
GPU, and checks that the ``DEEP_GEMM`` unquantized backend is opt-in only.

Run: ``pytest tests/kernels/moe/test_deepgemm_bf16_moe.py``.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_bf16_moe import (
    DeepGemmBf16BatchedExperts,
    DeepGemmBf16Experts,
)
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    backend_to_kernel_cls,
    map_unquantized_backend,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
    BatchedPrepareAndFinalize,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    is_deep_gemm_bf16_grouped_supported,
    is_deep_gemm_bf16_masked_supported,
)


def _bf16_moe_weights(e: int, n: int, k: int):
    """(w1, w2) unquantized bf16 expert weights: w1 (E, 2N, K), w2 (E, K, N)."""
    dtype = torch.bfloat16
    w1 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) / 10
    w2 = torch.randn(e, k, n, device="cuda", dtype=dtype) / 10
    return w1, w2


# N, K aligned to 128 (DeepGEMM contiguous requirement); M >= 128.
CONTIG_MNKS = [
    (1024, 768, 128),
    (2048, 768, 512),
    (512, 1024, 1024),
]


@pytest.mark.parametrize(("m", "n", "k"), CONTIG_MNKS)
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.skipif(
    not is_deep_gemm_bf16_grouped_supported(),
    reason="Requires DeepGEMM bf16 grouped kernel",
)
def test_deepgemm_bf16_contiguous_vs_triton(
    m, n, k, topk, num_experts, monkeypatch, workspace_init
):
    """DeepGemmBf16Experts (Standard) == Triton reference."""
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        tokens = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).clamp_(-1, 1)
        w1, w2 = _bf16_moe_weights(num_experts, n, k)

        router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
        topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

        quant_config = FusedMoEQuantConfig.make()  # unquantized
        moe_config = make_dummy_moe_config()

        deepgemm_kernel = mk.FusedMoEKernel(
            prepare_finalize=maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            fused_experts=DeepGemmBf16Experts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        out_triton = fused_experts(
            hidden_states=tokens,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_config=quant_config,
        )
        out_deepgemm = deepgemm_kernel.apply(
            hidden_states=tokens,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=num_experts,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )
        diff = calc_diff(out_deepgemm, out_triton)
        assert diff < 1e-3, f"contiguous diff too large: {diff}"


@pytest.mark.parametrize("E", [16, 32])
@pytest.mark.parametrize("T", [256])
@pytest.mark.parametrize("K", [256])
@pytest.mark.parametrize("N", [512, 1024])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.skipif(
    not is_deep_gemm_bf16_masked_supported(),
    reason="Requires DeepGEMM bf16 masked kernel",
)
def test_deepgemm_bf16_masked_vs_triton(E, T, K, N, topk, monkeypatch, workspace_init):
    """DeepGemmBf16BatchedExperts (BatchedExperts) == BatchedTriton reference."""
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        w1, w2 = _bf16_moe_weights(E, N, K)
        M = E * T
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 10.0

        router_logits = torch.randn(M, E, device="cuda", dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
        topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

        cnt = torch.bincount(topk_ids.flatten(), minlength=E)
        max_num_tokens = 1 << (int(cnt.max().item()) - 1).bit_length()

        prep_finalize = BatchedPrepareAndFinalize(
            max_num_tokens=max_num_tokens,
            num_local_experts=E,
            num_dispatchers=1,
            rank=0,
        )
        quant_config = FusedMoEQuantConfig.make()  # unquantized
        moe_config = make_dummy_moe_config()

        common = dict(
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

        out_triton = mk.FusedMoEKernel(
            prep_finalize,
            BatchedTritonExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=1,
                quant_config=quant_config,
                moe_config=moe_config,
            ),
        ).apply(**common)

        out_deepgemm = mk.FusedMoEKernel(
            prep_finalize,
            DeepGemmBf16BatchedExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=1,
                quant_config=quant_config,
                moe_config=moe_config,
            ),
        ).apply(**common)

        diff = calc_diff(out_deepgemm, out_triton)
        assert diff < 1e-3, f"masked diff too large: {diff}"


def test_deep_gemm_backend_is_opt_in():
    """DEEP_GEMM must never be auto-selected, only via moe_backend='deep_gemm'."""
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        _get_priority_backends,
    )

    moe_config = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=2048, intermediate_size=1024
    )
    # Not in the auto-priority list.
    assert UnquantizedMoeBackend.DEEP_GEMM not in _get_priority_backends(moe_config)
    # Reachable only by explicit request.
    assert map_unquantized_backend("deep_gemm") == UnquantizedMoeBackend.DEEP_GEMM
    names = [c.__name__ for c in backend_to_kernel_cls(UnquantizedMoeBackend.DEEP_GEMM)]
    assert names == ["DeepGemmBf16Experts", "DeepGemmBf16BatchedExperts"]
