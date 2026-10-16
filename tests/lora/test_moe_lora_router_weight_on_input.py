# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE LoRA with apply_router_weight_on_input (top-1 routing, e.g. Llama4).

prepare() scales the input by the routing weight and the base GEMMs skip it on
the output, so the LoRA deltas must see the same scaled input and must not be
weighted again. With top-1 routing that makes the layer on (x, w) equal to the
layer on (x * w, 1).
"""

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config.lora import LoRAConfig
from vllm.lora.layers import LoRAMapping
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cuda():
    pytest.skip("MoE LoRA Triton kernels require CUDA", allow_module_level=True)

from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU  # noqa: E402

pytestmark = pytest.mark.skip_global_cleanup

E, H, INTER, RANK, MAX_LORAS = 8, 256, 512, 16, 2
DTYPE = torch.bfloat16
DEVICE = "cuda"


@pytest.fixture(autouse=True)
def _workspace():
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    init_workspace_manager(torch.device(DEVICE))
    yield
    reset_workspace_manager()


def _randn(*shape: int, scale: float) -> torch.Tensor:
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE) * scale


def _make_lora_context(num_tokens: int) -> MoELoRAContext:
    """Rank-RANK adapters in slot 0, every token mapped to LoRA id 1 (slot 0)."""
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=RANK, lora_dtype=DTYPE)
    punica = PunicaWrapperGPU(1024, 16, DEVICE, lora_config=lora_config)
    mapping = LoRAMapping([1] * num_tokens, [1], is_prefill=True)
    punica.update_metadata(mapping, [1] + [None] * (MAX_LORAS - 1), MAX_LORAS, 1024)

    def stacked(*shape: int, scale: float) -> torch.Tensor:
        # Only slot 0 holds an adapter; the other slots stay zero.
        t = torch.zeros(MAX_LORAS, E, *shape, device=DEVICE, dtype=DTYPE)
        t[0] = _randn(E, *shape, scale=scale)
        return t

    adapter_enabled = torch.zeros(MAX_LORAS + 1, dtype=torch.int32, device=DEVICE)
    adapter_enabled[0] = 1
    return MoELoRAContext(
        w13_lora_a_stacked=tuple(stacked(RANK, H, scale=H**-0.5) for _ in range(2)),
        w13_lora_b_stacked=tuple(
            stacked(INTER, RANK, scale=RANK**-0.5) for _ in range(2)
        ),
        w2_lora_a_stacked=(stacked(RANK, INTER, scale=INTER**-0.5),),
        w2_lora_b_stacked=(stacked(H, RANK, scale=RANK**-0.5),),
        adapter_enabled=adapter_enabled,
        max_loras=MAX_LORAS,
        top_k=1,
        w13_num_slices=2,
        fully_sharded=False,
        tp_rank=0,
        tp_size=1,
        local_num_experts=E,
        punica_wrapper=punica,
        use_tuned_config=False,
    )


def _make_kernel(lora_context: MoELoRAContext | None) -> FusedMoEKernel:
    moe_config = make_dummy_moe_config(
        num_experts=E, experts_per_token=1, hidden_dim=H, intermediate_size=INTER
    )
    experts = TritonExperts(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)
    if lora_context is not None:
        experts.set_lora_context(lora_context)
    return FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), experts)


def _run(kernel, x, w1, w2, topk_weights, topk_ids, on_input: bool):
    return kernel.apply(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        apply_router_weight_on_input=on_input,
    )


def _make_inputs(num_tokens: int):
    x = _randn(num_tokens, H, scale=1.0)
    w1 = _randn(E, 2 * INTER, H, scale=H**-0.5)
    w2 = _randn(E, H, INTER, scale=INTER**-0.5)
    topk_ids = torch.randint(0, E, (num_tokens, 1), device=DEVICE, dtype=torch.int32)
    # Keep weights well away from 1 so a missing or doubled weight shows.
    topk_weights = torch.rand(num_tokens, 1, device=DEVICE) * 0.6 + 0.2
    return x, w1, w2, topk_ids, topk_weights


# M=2 takes the naive per-token LoRA block assignment, M=32 the sorted one.
NUM_TOKENS = pytest.mark.parametrize(
    "num_tokens", [2, 32], ids=["naive-block", "sorted-block"]
)


@torch.inference_mode()
@NUM_TOKENS
def test_router_weight_on_input_matches_prescaled_input(num_tokens: int):
    set_random_seed(0)
    x, w1, w2, topk_ids, topk_weights = _make_inputs(num_tokens)
    kernel = _make_kernel(_make_lora_context(num_tokens))

    out = _run(kernel, x, w1, w2, topk_weights, topk_ids, on_input=True)
    ref = _run(
        kernel,
        x * topk_weights.to(DTYPE),
        w1,
        w2,
        torch.ones_like(topk_weights),
        topk_ids,
        on_input=False,
    )
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
@NUM_TOKENS
def test_lora_delta_is_nontrivial(num_tokens: int):
    # Guards the test above against passing because the LoRA does nothing.
    set_random_seed(0)
    x, w1, w2, topk_ids, topk_weights = _make_inputs(num_tokens)
    args = (x, w1, w2, topk_weights, topk_ids)

    with_lora = _run(_make_kernel(_make_lora_context(num_tokens)), *args, True)
    base = _run(_make_kernel(None), *args, True)
    rel = (with_lora - base).float().norm() / base.float().norm()
    assert rel > 0.3, f"LoRA delta too small to test anything: {rel:.3f}"
