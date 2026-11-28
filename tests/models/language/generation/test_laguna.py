# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from safetensors.torch import load_file
from transformers import LagunaConfig, LagunaForCausalLM


def _routing_bias(model):
    return model.model.layers[1].mlp.experts.router.e_score_correction_bias.cpu()


@pytest.mark.core_model
def test_routes_with_checkpoint_e_score_correction_bias(
    vllm_runner, enable_pickle, tmp_path
):
    """The MoE router's load-balancing bias is read from the checkpoint."""
    config = LagunaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        mlp_layer_types=["dense", "sparse"],
        layer_types=["full_attention", "sliding_attention"],
    )
    torch.manual_seed(0)
    hf_model = LagunaForCausalLM(config).to(torch.bfloat16)
    torch.nn.init.normal_(hf_model.model.layers[1].mlp.gate.e_score_correction_bias)
    hf_model.save_pretrained(tmp_path)
    checkpoint = load_file(tmp_path / "model.safetensors")
    bias = checkpoint["model.layers.1.mlp.experts.e_score_correction_bias"]
    assert bias.count_nonzero() == bias.numel()

    with vllm_runner(
        str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
    ) as llm:
        [routing_bias] = llm.apply_model(_routing_bias)

    torch.testing.assert_close(routing_bias, bias.float(), rtol=0, atol=0)
