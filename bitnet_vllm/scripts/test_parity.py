"""vLLM parity test: compare against saved HF reference data.

Must be run as a script (not imported) due to multiprocessing spawn.
"""
import json
import os

import bitnet_vllm
bitnet_vllm.register()

from vllm import LLM, SamplingParams


def strip_quant_config(config):
    """Remove quantization_config so vLLM treats this as pure BF16."""
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    return config


def main():
    model_name = "microsoft/bitnet-b1.58-2B-4T-bf16"
    prompt = "Hello, my name is"

    # Load HF reference if available
    ref_path = "/app/bitnet_vllm/scripts/hf_reference.json"
    hf_ref = None
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            hf_ref = json.load(f)
        print("Loaded HF reference data")
        print(f"  HF greedy: {hf_ref['greedy_text']!r}")
        print(f"  HF top-5:  {hf_ref['top20_token_ids'][:5]}")
    else:
        print("No HF reference found. Run hf_reference.py first.")

    print("\n=== Loading vLLM model ===")
    llm = LLM(
        model=model_name,
        hf_overrides=strip_quant_config,
        dtype='bfloat16',
        max_model_len=512,
    )

    # Greedy generation
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        logprobs=20,
    )
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]

    print(f"\nvLLM greedy generation: {output.outputs[0].text!r}")

    # Print top logprobs for the first generated token
    if output.outputs[0].logprobs:
        first_token_logprobs = output.outputs[0].logprobs[0]
        sorted_lps = sorted(
            first_token_logprobs.items(),
            key=lambda x: x[1].logprob,
            reverse=True
        )
        print("\nvLLM Top-20 next-token predictions:")
        vllm_top20 = []
        for i, (token_id, lp) in enumerate(sorted_lps[:20]):
            tok_str = lp.decoded_token if lp.decoded_token else f"<{token_id}>"
            print(f"  {i+1:2d}. token={token_id:6d} ({tok_str!r:>12s}) logprob={lp.logprob:.4f}")
            vllm_top20.append(token_id)

        # Compare with HF
        if hf_ref:
            hf_top20 = hf_ref["top20_token_ids"]
            overlap = set(hf_top20) & set(vllm_top20)
            print(f"\n{'='*50}")
            print(f"PARITY RESULTS")
            print(f"{'='*50}")
            print(f"HF top-10:   {hf_top20[:10]}")
            print(f"vLLM top-10: {vllm_top20[:10]}")
            print(f"Overlap in top-10: {len(set(hf_top20[:10]) & set(vllm_top20[:10]))}/10")
            print(f"Overlap in top-20: {len(overlap)}/20")
            if hf_top20[0] == vllm_top20[0]:
                print("✅ Top-1 token matches!")
            else:
                print(f"❌ Top-1 mismatch: HF={hf_top20[0]}, vLLM={vllm_top20[0]}")
    else:
        print("⚠️  No logprobs available")


if __name__ == '__main__':
    main()
