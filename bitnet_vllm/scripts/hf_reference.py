"""HF Reference: Generate ground truth from HuggingFace transformers."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "microsoft/bitnet-b1.58-2B-4T-bf16"
    prompt = "Hello, my name is"

    print("=== Loading HF model ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    print(f"Prompt: {prompt!r}")
    print(f"Input tokens: {input_ids[0].tolist()}")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Last-token logits (next-token prediction)
    last_logits = logits[0, -1, :]  # [vocab_size]
    top_values, top_indices = torch.topk(last_logits, k=20)

    print("\nHF Top-20 next-token predictions:")
    for i, (val, idx) in enumerate(zip(top_values, top_indices)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1:2d}. token={idx.item():6d} ({token!r:>12s}) logit={val.item():.4f}")

    # Greedy generation
    gen_output = model.generate(
        input_ids,
        max_new_tokens=32,
        do_sample=False,
        temperature=1.0,
    )
    generated_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
    print(f"\nHF greedy generation: {generated_text!r}")

    # Save reference data
    ref_data = {
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "top20_token_ids": top_indices.tolist(),
        "top20_logits": [round(v, 4) for v in top_values.tolist()],
        "greedy_text": generated_text,
        "greedy_token_ids": gen_output[0].tolist(),
    }
    with open("/app/bitnet_vllm/scripts/hf_reference.json", "w") as f:
        json.dump(ref_data, f, indent=2)
    print("\nSaved reference to hf_reference.json")


if __name__ == "__main__":
    main()
