"""Compare the outputs of HF and vLLM for the Mixtral model.

Currently we only compare the MoE layer because of GPU memory limits,
once a smaller version of Mixtral is available we should also run an
end-to-end tests.

Run `pytest tests/models/test_mixtral.py`.
"""

import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from vllm.model_executor.models.mixtral import MixtralMoE

@torch.inference_mode
def test_mixtral_moe():
    "Make sure our Mixtral MoE implementation agrees with the one from huggingface."

    # Instantiate our and huggingface's MoE blocks
    config = MixtralConfig()
    hf_moe = MixtralSparseMoeBlock(config).to("cuda")
    vllm_moe = MixtralMoE(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        tp_size=1,
    )

    # Load the weights
    vllm_moe.gate.linear_weights["weight"][:] = hf_moe.gate.weight.data
    for i in range(config.num_local_experts):
        weights = (hf_moe.experts[i].w1.weight.data,
                   hf_moe.experts[i].w3.weight.data)
        vllm_moe.ws[i][:] = torch.cat(weights, dim=0).to("cuda")
        vllm_moe.w2s[i][:] = hf_moe.experts[i].w2.weight.data

    # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    inputs = torch.randn((1, 64, config.hidden_size)).to("cuda")

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(inputs)
    vllm_states = vllm_moe.forward(inputs)

    assert torch.allclose(hf_states, vllm_states, rtol=1e-3, atol=1e-3)
