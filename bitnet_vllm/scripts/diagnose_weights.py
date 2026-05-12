"""Diagnose: compare HF model vs manual BitNet forward pass to find divergence."""
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    model_name = 'microsoft/bitnet-b1.58-2B-4T-bf16'
    device = 'cuda'

    print("=== Loading HF reference model ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device
    )
    hf_model.eval()

    # Tokenize
    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"Input: {prompt!r} -> tokens: {input_ids[0].tolist()}")

    # HF forward
    with torch.no_grad():
        hf_out = hf_model(input_ids)
        hf_logits = hf_out.logits[0, -1, :]  # last token

    hf_top = torch.topk(hf_logits, 5)
    print(f"\nHF top-5: {list(zip(hf_top.indices.tolist(), [round(v,2) for v in hf_top.values.tolist()]))}")

    # Now manually reproduce what vLLM's BitNet does using HF weights
    # The question: is our MLP activation correct?
    print("\n=== Testing MLP activation manually ===")
    layer0 = hf_model.model.layers[0]

    # Get the hidden state after input_layernorm
    with torch.no_grad():
        embed = hf_model.model.embed_tokens(input_ids)
        normed = layer0.input_layernorm(embed)

    # Test MLP path manually
    with torch.no_grad():
        # HF's MLP forward
        hf_gate = layer0.mlp.gate_proj(normed)
        hf_up = layer0.mlp.up_proj(normed)
        hf_act = layer0.mlp.act_fn  # This is the relu2 function

        # What HF computes
        hf_mlp_mid = hf_act(hf_gate) * hf_up
        hf_mlp_normed = layer0.mlp.ffn_sub_norm(hf_mlp_mid)
        hf_mlp_out = layer0.mlp.down_proj(hf_mlp_normed)

        # What our vLLM code computes (with merged gate_up_proj)
        # We combine gate and up into one linear, then split
        gate_weight = layer0.mlp.gate_proj.weight  # [6912, 2560]
        up_weight = layer0.mlp.up_proj.weight       # [6912, 2560]
        merged_weight = torch.cat([gate_weight, up_weight], dim=0)  # [13824, 2560]

        gate_up = F.linear(normed, merged_weight)  # [1, 6, 13824]
        d = gate_up.shape[-1] // 2
        gate = gate_up[..., :d]
        up = gate_up[..., d:]

        # Our relu² activation: torch.square(F.relu(gate)) * up
        vllm_mlp_mid = torch.square(F.relu(gate)) * up

        # Compare mid-MLP outputs
        mid_diff = (hf_mlp_mid.float() - vllm_mlp_mid.float()).abs()
        print(f"  MLP mid diff (after act*up): max={mid_diff.max():.6f} mean={mid_diff.mean():.6f}")

        # Check act_fn identity
        print(f"  HF act_fn type: {type(hf_act).__name__}")
        # Test: is hf_act(x) == relu(x)^2 ?
        test_x = torch.randn(10, device=device, dtype=torch.bfloat16)
        hf_result = hf_act(test_x)
        our_result = torch.square(F.relu(test_x))
        act_diff = (hf_result.float() - our_result.float()).abs()
        print(f"  act_fn test: max_diff={act_diff.max():.6f}")

    # Now test the full attention + MLP layer to see where divergence happens
    print("\n=== Layer-by-layer hidden state comparison ===")
    with torch.no_grad():
        # HF full forward pass, collect hidden states per layer
        hf_full = hf_model.model(input_ids, output_hidden_states=True)
        hf_hidden_states = hf_full.hidden_states  # tuple of [batch, seq, hidden]

        print(f"  Number of hidden states: {len(hf_hidden_states)} (embed + {len(hf_hidden_states)-1} layers)")

        # Check first few layers
        for i in range(min(4, len(hf_hidden_states))):
            hs = hf_hidden_states[i]
            print(f"  Layer {i}: shape={hs.shape} min={hs.min():.4f} max={hs.max():.4f} mean={hs.mean():.4f}")

    # The key question: does vLLM's Attention layer produce the same output?
    # Let's check if there's a RoPE issue by comparing Q,K after rotation
    print("\n=== RoPE sanity check ===")
    config = AutoConfig.from_pretrained(model_name)
    print(f"  rope_theta: {config.rope_theta}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  head_dim: {config.hidden_size // config.num_attention_heads}")
    print(f"  Has rope_scaling: {getattr(config, 'rope_scaling', None)}")
    print(f"  Has rope_parameters: {hasattr(config, 'rope_parameters')}")
    if hasattr(config, 'rope_parameters'):
        print(f"  rope_parameters: {config.rope_parameters}")

    # Final: compute logits from all hidden states to verify lm_head
    print("\n=== LM Head check ===")
    with torch.no_grad():
        final_hidden = hf_full.last_hidden_state  # [1, seq_len, hidden_size]
        # LM head: should be tied to embed_tokens
        print(f"  tie_word_embeddings: {config.tie_word_embeddings}")
        print(f"  lm_head weight is embed_tokens weight: {hf_model.lm_head.weight.data_ptr() == hf_model.model.embed_tokens.weight.data_ptr()}")


if __name__ == '__main__':
    main()
