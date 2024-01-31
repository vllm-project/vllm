"""Test the MOE layer of the Mixtral model.

Run `pytest tests/kernels/test_mixtral_moe.py`.
"""

import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from vllm.model_executor.models.mixtral import MixtralMoE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_mixtral_moe(dtype: torch.dtype):
    "Make sure our Mixtral MoE implementation agrees with the one from huggingface."

    # Instantiate our and huggingface's MoE blocks
    config = MixtralConfig()
    hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    vllm_moe = MixtralMoE(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        params_dtype=dtype,
        tp_size=1,
    )

    # Load the weights
    vllm_moe.gate.linear_weights["weight"][:] = hf_moe.gate.weight.data
    for i in range(config.num_local_experts):
        weights = (hf_moe.experts[i].w1.weight.data,
                   hf_moe.experts[i].w3.weight.data)
        vllm_moe.ws[i][:] = torch.cat(weights, dim=0)
        vllm_moe.w2s[i][:] = hf_moe.experts[i].w2.weight.data

    # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    inputs = torch.randn((1, 64, config.hidden_size)).to(dtype).to("cuda")

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(inputs)
    vllm_states = vllm_moe.forward(inputs)

    tol = {
        torch.float32: 1e-3,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }

    assert torch.allclose(hf_states, vllm_states, rtol=tol[dtype], atol=tol[dtype])
