# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.forward_context import set_forward_context
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.layers.utils import LoRAMapping
from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.mk_fused_experts_lora_support import (
    mk_fused_experts_supports_lora,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)


def make_fused_moe_layer(
    num_experts: int,
    num_topk: int,
    hidden_size: int,
    intermediate_size: int,
    params_dtype: torch.dtype,
    device: str,
) -> FusedMoE:
    fml = FusedMoE(
        num_experts=num_experts,
        top_k=num_topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        prefix="dummy",
        activation="silu",
        is_act_and_mul=True,
        params_dtype=params_dtype,
    )
    fml.w13_weight.data = fml.w13_weight.data.to(device=device)
    fml.w2_weight.data = fml.w2_weight.data.to(device=device)

    # setup quant method to use ModularKernels
    fml.ensure_moe_quant_config_init()
    qm = fml.quant_method
    if not qm.using_modular_kernel:
        assert qm.topk_indices_dtype is None
        assert qm.fused_experts is None

        prepare_finalize = MoEPrepareAndFinalizeNoEP()
        mk_experts = qm.select_gemm_impl(prepare_finalize, fml)

        qm.topk_indices_dtype = prepare_finalize.topk_indices_dtype()
        qm.fused_experts = FusedMoEModularKernel(
            prepare_finalize, mk_experts, fml.shared_experts
        )

    return fml


def make_fused_moe_with_lora_layer(
    fused_moe_layer, lora_config: LoRAConfig
) -> FusedMoEWithLoRA:
    fml_lora = FusedMoEWithLoRA(fused_moe_layer)
    fml_lora.create_lora_weights(max_loras=8, lora_config=lora_config)
    return fml_lora


def fused_moe_forward_args(
    M: int, K: int, E: int, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn((M, K), device=device, dtype=dtype)
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    return (hidden_states, router_logits)


# TODO (varun) : Add stronger tests.
@pytest.mark.parametrize("MNK", [(16, 4096, 2048)])
@pytest.mark.parametrize("E", [16])
@pytest.mark.parametrize("num_topk", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe_lora_layer_weak(
    dist_init, MNK: tuple[int, int, int], E: int, num_topk: int, dtype: torch.dtype
):
    """
    Test to check that FusedMoEWithLoRA layer actually executes the
    LoRA operations.
    """

    device = "cuda"
    M, N, K = MNK
    MAX_LORAS = 3

    # Setup inputs
    hidden_states, router_logits = fused_moe_forward_args(
        M, K, E, device=device, dtype=dtype
    )

    def do_test():
        # Test Plan:
        # 1. Create FusedMoE layer
        # 2. Set all FusedMoE layer weights to 0. When we do a forward pass
        #    with just the FusedMoE layer the output should be all zeros.
        # 3. Create a FusedMoEWithLoRA layer using the same FusedMoELayer
        # 4. Set all LoRA Weights to non-zero. When we do a forward pass
        #    with the FusedMoEWithLoRA layer the output should be non-zero.

        vllm_config = get_current_vllm_config()
        lora_config = vllm_config.lora_config

        ## FusedMoE Layer ########
        # Make FusedMoE Layer
        fml = make_fused_moe_layer(E, num_topk, K, N, dtype, device)
        mk = fml.quant_method.fused_experts
        # Sanity check
        assert isinstance(mk, FusedMoEModularKernel)
        print(
            "FusedMoE => ModularKernel ( "
            f"prepare_finalize={mk.prepare_finalize}, "
            f"fused_experts={mk.fused_experts})"
        )
        assert mk_fused_experts_supports_lora(mk.fused_experts), (
            "Cannot test FusedMoEWithLoRA layer with a "
            "FusedMoEPermuteExpertsUnpermute implementation that supports LoRA"
        )

        # FusedMoE weights -> 0
        fml.w13_weight.data.fill_(0).to(device=device)
        fml.w2_weight.data.fill_(0).to(device=device)

        fml_out = fml.forward(hidden_states, router_logits)
        assert torch.all(fml_out == 0)

        ### FusedMoE Layer with LoRA ####
        # Make fused_moe_lora
        fml_lora: FusedMoEWithLoRA = make_fused_moe_with_lora_layer(fml, lora_config)
        punica_wrapper = PunicaWrapperGPU(
            max_num_batched_tokens=8192,
            max_batches=1024,
            device="cuda",
            max_loras=MAX_LORAS,
        )
        # setup punica wrapper
        punica_wrapper.update_metadata(
            mapping=LoRAMapping(index_mapping=tuple([0] * M), prompt_mapping=(0,)),
            lora_index_to_id=[0, 1, 2],
            max_loras=MAX_LORAS,
            # set to zero as it isn't really used
            vocab_size=0,
            extra_vocab_size=0,
        )
        fml_lora.set_mapping(punica_wrapper)

        # FusedMoEWithLoRA weights -> non-zero
        fml_lora.w1_lora_a_stacked.fill_(1)
        fml_lora.w1_lora_b_stacked.fill_(1)
        fml_lora.w2_lora_a_stacked.fill_(1)
        fml_lora.w2_lora_b_stacked.fill_(1)
        fml_lora.w3_lora_a_stacked.fill_(1)
        fml_lora.w3_lora_b_stacked.fill_(1)

        fml_lora_out = fml_lora.forward(hidden_states, router_logits)
        assert not torch.all(fml_lora_out == 0)

    # Setup configs
    lora_config = LoRAConfig()
    lora_config.max_lora_rank = 8
    lora_config.max_loras = MAX_LORAS
    lora_config.lora_dtype = dtype
    vllm_config = VllmConfig()
    vllm_config.lora_config = lora_config

    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config, num_tokens=M),
    ):
        do_test()
