# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

if not current_platform.is_cuda():
    pytest.skip("MoE LoRA Triton kernels require CUDA", allow_module_level=True)

from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU  # noqa: E402

E, H, INTER, RANK = 8, 256, 512, 16


def _rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * shape[-1] ** -0.5


def _make_kernel(num_tokens: int) -> FusedMoEKernel:
    punica = PunicaWrapperGPU(
        1024, 16, "cuda", lora_config=LoRAConfig(max_loras=1, max_lora_rank=RANK)
    )
    mapping = LoRAMapping([1] * num_tokens, [1], is_prefill=True)
    punica.update_metadata(mapping, [1], 1, 1024)
    experts = TritonExperts(
        make_dummy_moe_config(
            num_experts=E, experts_per_token=1, hidden_dim=H, intermediate_size=INTER
        ),
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    experts.set_lora_context(
        MoELoRAContext(
            w13_lora_a_stacked=(_rand(1, E, RANK, H), _rand(1, E, RANK, H)),
            w13_lora_b_stacked=(_rand(1, E, INTER, RANK), _rand(1, E, INTER, RANK)),
            w2_lora_a_stacked=(_rand(1, E, RANK, INTER),),
            w2_lora_b_stacked=(_rand(1, E, H, RANK),),
            adapter_enabled=torch.tensor([1, 0], dtype=torch.int32, device="cuda"),
            max_loras=1,
            top_k=1,
            w13_num_slices=2,
            fully_sharded=False,
            tp_rank=0,
            tp_size=1,
            local_num_experts=E,
            punica_wrapper=punica,
            use_tuned_config=False,
        )
    )
    return FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), experts)


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [2, 32], ids=["naive-block", "sorted-block"])
def test_router_weight_on_input_matches_prescaled_input(num_tokens: int):
    # prepare() scales the input by the top-1 weight, so the layer with LoRA on
    # (x, w) must equal the layer on (x * w, 1).
    set_random_seed(0)
    init_workspace_manager(torch.device("cuda"))
    kernel = _make_kernel(num_tokens)
    x = torch.randn(num_tokens, H, device="cuda", dtype=torch.bfloat16)
    w1, w2 = _rand(E, 2 * INTER, H), _rand(E, H, INTER)
    ids = torch.randint(0, E, (num_tokens, 1), device="cuda", dtype=torch.int32)
    weights = torch.rand(num_tokens, 1, device="cuda") * 0.6 + 0.2

    def run(x, weights, on_input):
        return kernel.apply(
            x, w1, w2, weights, ids, MoEActivation.SILU, E, None, on_input
        )

    out = run(x, weights, True)
    ref = run(x * weights.to(x.dtype), torch.ones_like(weights), False)
    reset_workspace_manager()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
