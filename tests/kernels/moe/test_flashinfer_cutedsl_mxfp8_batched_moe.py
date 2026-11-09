# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 (W8A8) FlashInfer CuteDSL experts on the batched activation format."""

import dataclasses

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe import (  # noqa: E501
    FlashInferCuteDSLMxfp8BatchedExperts,
    flashinfer_cutedsl_mxfp8_moe_masked,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    _mxfp8_backend_to_kernel_cls,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.utils.import_utils import has_deep_ep
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch
from .utils import make_dummy_moe_config

if has_deep_ep():
    from .parallel_utils import DeepEPLLArgs, make_deepep_a2a

if not FlashInferCuteDSLMxfp8BatchedExperts._supports_current_device():
    pytest.skip(
        reason="Requires SM100/SM103 and FlashInfer CuteDSL masked MXFP8 MoE.",
        allow_module_level=True,
    )

MXFP8_BLOCK_SHAPE = [1, 32]


def _quant_dequant_mxfp8(x: torch.Tensor) -> torch.Tensor:
    x_q, x_s = mxfp8_e4m3_quantize(x)
    return dequant_mxfp8_to_bf16(x_q, x_s.view(*x.shape[:-1], -1))


def _torch_mxfp8_expert(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """One expert with the kernel's quantization points: both GEMM inputs are
    MXFP8 quantize-dequantized and the gate/up output is rounded to bf16."""
    gate_up = (_quant_dequant_mxfp8(x).float() @ w1.float().T).to(x.dtype)
    n = gate_up.shape[-1] // 2
    act = (torch.nn.functional.silu(gate_up[:, :n].float()) * gate_up[:, n:]).to(
        x.dtype
    )
    return _quant_dequant_mxfp8(act).float() @ w2.float().T


def _torch_mxfp8_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(a, dtype=torch.float32)
    for e in range(w1.shape[0]):
        token_idx, k_idx = torch.nonzero(topk_ids == e, as_tuple=True)
        if token_idx.numel() == 0:
            continue
        y = _torch_mxfp8_expert(a[token_idx], w1[e], w2[e])
        out.index_add_(0, token_idx, y * topk_weights[token_idx, k_idx, None])
    return out.to(a.dtype)


def _make_mxfp8_weights(
    e: int, k: int, n: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random expert weights in the MXFP8 checkpoint layout: fp8 values and
    row-major UE8M0 scales."""
    w1 = torch.randn(e, 2 * n, k, device=device, dtype=torch.bfloat16) / k**0.5
    w2 = torch.randn(e, k, n, device=device, dtype=torch.bfloat16) / n**0.5
    w1_q, w1_s = mxfp8_e4m3_quantize(w1)
    w2_q, w2_s = mxfp8_e4m3_quantize(w2)
    return (
        w1_q.view(e, 2 * n, k),
        w1_s.view(e, 2 * n, k // 32),
        w2_q.view(e, k, n),
        w2_s.view(e, k, n // 32),
    )


def _make_mxfp8_kernel_weights(w1_q, w1_s, w2_q, w2_s):
    backend = Fp8MoeBackend.FLASHINFER_CUTEDSL_BATCHED
    w1_kernel, w2_kernel, w1_s_kernel, w2_s_kernel = convert_to_fp8_moe_kernel_format(
        backend, None, w1_q, w2_q, w1_s, w2_s, None, None
    )
    quant_config = make_fp8_moe_quant_config(
        fp8_backend=backend,
        w1_scale=w1_s_kernel,
        w2_scale=w2_s_kernel,
        a1_scale=None,
        a2_scale=None,
        block_shape=MXFP8_BLOCK_SHAPE,
    )
    return w1_kernel, w2_kernel, quant_config


@pytest.mark.parametrize("e,max_m,k,n", [(8, 64, 256, 128), (16, 128, 512, 384)])
@torch.inference_mode()
def test_flashinfer_cutedsl_mxfp8_moe_masked(e: int, max_m: int, k: int, n: int):
    """The first masked_m[i] rows of each expert match a per-expert reference,
    for empty, partially filled and full experts."""
    torch.manual_seed(0)
    device = "cuda"
    hidden_states = torch.randn(e, max_m, k, device=device, dtype=torch.bfloat16)
    masked_m = torch.randint(0, max_m + 1, (e,), device=device, dtype=torch.int32)
    masked_m[0], masked_m[-1] = 0, max_m

    w1_q, w1_s, w2_q, w2_s = _make_mxfp8_weights(e, k, n, device)
    w1_kernel, w2_kernel, quant_config = _make_mxfp8_kernel_weights(
        w1_q, w1_s, w2_q, w2_s
    )
    out = torch.empty_like(hidden_states)
    flashinfer_cutedsl_mxfp8_moe_masked(
        hidden_states=hidden_states,
        w1=w1_kernel,
        w1_blockscale=quant_config.w1_scale,
        w2=w2_kernel,
        w2_blockscale=quant_config.w2_scale,
        masked_m=masked_m,
        workspace=torch.empty(e, max_m, 2 * n, device=device, dtype=torch.bfloat16),
        out=out,
    )

    w1_dq = dequant_mxfp8_to_bf16(w1_q, w1_s)
    w2_dq = dequant_mxfp8_to_bf16(w2_q, w2_s)
    for i, rows in enumerate(masked_m.tolist()):
        if rows == 0:
            continue
        ref = _torch_mxfp8_expert(hidden_states[i, :rows], w1_dq[i], w2_dq[i])
        torch.testing.assert_close(
            out[i, :rows], ref.to(out.dtype), atol=3e-2, rtol=5e-2
        )


def test_flashinfer_cutedsl_mxfp8_batched_experts_support():
    assert _mxfp8_backend_to_kernel_cls(Fp8MoeBackend.FLASHINFER_CUTEDSL_BATCHED) == [
        FlashInferCuteDSLMxfp8BatchedExperts
    ]

    cls = FlashInferCuteDSLMxfp8BatchedExperts

    def supported(hidden_dim, intermediate_size, activation_format):
        moe_config = make_dummy_moe_config(
            num_experts=8,
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_size,
        )
        return cls.is_supported_config(
            cls, moe_config, kMxfp8Static, kMxfp8Dynamic, activation_format
        )[0]

    batched = mk.FusedMoEActivationFormat.BatchedExperts
    assert supported(4096, 2048, batched)
    assert not supported(4096, 2048, mk.FusedMoEActivationFormat.Standard)
    assert not supported(4160, 2048, batched)
    assert not supported(4096, 2880, batched)


@dataclasses.dataclass
class DeepEPLLConfig:
    num_experts: int
    topk: int
    hidden: int  # k
    intermediate: int  # n
    max_tokens_per_rank: int
    rank_tokens: tuple[int, ...]  # tokens per call on rank 0; rank r drops 3 * r


def _deepep_ll_mxfp8_moe_worker(pgi: ProcessGroupInfo, config: DeepEPLLConfig):
    torch.manual_seed(0)  # same weights on every rank
    device = pgi.device
    init_workspace_manager(device)
    e, topk = config.num_experts, config.topk
    k, n = config.hidden, config.intermediate
    w1_q, w1_s, w2_q, w2_s = _make_mxfp8_weights(e, k, n, device)
    w1_dq = dequant_mxfp8_to_bf16(w1_q, w1_s)
    w2_dq = dequant_mxfp8_to_bf16(w2_q, w2_s)

    num_local_experts = e // pgi.world_size
    first = pgi.rank * num_local_experts
    local = slice(first, first + num_local_experts)
    w1_kernel, w2_kernel, quant_config = _make_mxfp8_kernel_weights(
        w1_q[local], w1_s[local], w2_q[local], w2_s[local]
    )
    expert_map = torch.full((e,), -1, dtype=torch.int32, device=device)
    expert_map[local] = torch.arange(
        num_local_experts, dtype=torch.int32, device=device
    )

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    with set_current_vllm_config(VllmConfig()):
        a2a = make_deepep_a2a(
            pg=pg,
            pgi=pgi,
            dp_size=1,
            deepep_ht_args=None,
            deepep_ll_args=DeepEPLLArgs(
                max_tokens_per_rank=config.max_tokens_per_rank,
                hidden_size=k,
                num_experts=e,
                use_fp8_dispatch=False,
            ),
        )
        moe_config = make_dummy_moe_config(
            num_experts=e,
            num_local_experts=num_local_experts,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            max_num_tokens=config.max_tokens_per_rank,
        )
        kernel = FusedMoEKernel(
            a2a,
            FlashInferCuteDSLMxfp8BatchedExperts(
                moe_config=moe_config,
                quant_config=quant_config,
                max_num_tokens=config.max_tokens_per_rank,
                num_dispatchers=pgi.world_size,
            ),
        )

        # Compile the CuTe DSL GEMMs for the dispatch capacity before the first
        # DeepEP call: a rank that is still compiling holds the other ranks in
        # DeepEP's receive loop, which traps after its timeout.
        capacity = config.max_tokens_per_rank * pgi.world_size
        flashinfer_cutedsl_mxfp8_moe_masked(
            hidden_states=torch.zeros(
                num_local_experts, capacity, k, device=device, dtype=torch.bfloat16
            ),
            w1=w1_kernel,
            w1_blockscale=quant_config.w1_scale,
            w2=w2_kernel,
            w2_blockscale=quant_config.w2_scale,
            masked_m=torch.zeros(num_local_experts, device=device, dtype=torch.int32),
            workspace=torch.empty(
                num_local_experts, capacity, 2 * n, device=device, dtype=torch.bfloat16
            ),
            out=torch.empty(
                num_local_experts, capacity, k, device=device, dtype=torch.bfloat16
            ),
        )
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=pg)

        torch.manual_seed(1 + pgi.rank)  # different tokens and routing per rank
        # Several calls reuse the DeepEP buffer and the experts' workspaces.
        for m in config.rank_tokens:
            m = max(1, m - 3 * pgi.rank)
            a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            router_logits = torch.randn(m, e, device=device, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), k=topk)
            out = kernel.apply(
                hidden_states=a,
                w1=w1_kernel,
                w2=w2_kernel,
                topk_weights=topk_weights,
                topk_ids=topk_ids.to(torch.int64),
                activation=MoEActivation.SILU,
                global_num_experts=e,
                expert_map=expert_map,
                apply_router_weight_on_input=False,
            )
            ref = _torch_mxfp8_moe(a, w1_dq, w2_dq, topk_weights, topk_ids)
            torch.testing.assert_close(out, ref, atol=3e-2, rtol=5e-2)


DEEPEP_LL_CONFIGS = {
    # Qwen3-30B-A3B MoE geometry.
    "qwen3-30b-a3b": DeepEPLLConfig(128, 8, 2048, 768, 128, (128, 33, 1)),
    # DeepSeek-V3 hidden size, with a narrow intermediate size to keep it small.
    "h7168": DeepEPLLConfig(32, 8, 7168, 256, 128, (128, 33, 1)),
}


@pytest.mark.parametrize("config", DEEPEP_LL_CONFIGS.values(), ids=DEEPEP_LL_CONFIGS)
@pytest.mark.parametrize("world_size", [2, 4])
@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not has_deep_ep(), reason="Requires deep_ep kernels")
def test_deepep_ll_flashinfer_cutedsl_mxfp8_moe(
    config: DeepEPLLConfig, world_size: int
):
    """DeepEP low-latency bf16 dispatch into the MXFP8 batched experts matches the
    single-process reference on every rank."""
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs")
    parallel_launch(world_size, _deepep_ll_mxfp8_moe_worker, config)
