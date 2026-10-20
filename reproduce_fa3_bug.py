#!/usr/bin/env python3
"""Reproduce FA3 split-KV wrong logits bug #62626.

Trigger A: Exactly 2 new tokens on a prefix-cache hit with FULL cudagraph mode.
Expected: Same logits as FULL_AND_PIECEWISE or prefix caching off.
Actual: 2nd token logprobs differ by several nats on some cache sizes.
"""

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

def run(**kw):
    """Run test with given kwargs, return results dict."""
    llm = LLM(
        "Qwen/Qwen3-8B",
        enable_prefix_caching=True,
        block_size=16,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        **kw
    )

    # Generate synthetic prompt
    ids = llm.get_tokenizer()(
        " ".join(f"item {i} is red;" for i in range(200))
    ).input_ids[:256]

    out = {}

    # Test different numbers of new tokens (q=1,2,3)
    for q in (1, 2, 3):
        llm.reset_prefix_cache()

        # Prime the cache with 128-token prefix
        llm.generate(
            [TokensPrompt(prompt_token_ids=ids[:128])],
            SamplingParams(max_tokens=1),
            use_tqdm=False
        )

        # Extend by q tokens, request logprobs
        o = llm.generate(
            [TokensPrompt(prompt_token_ids=ids[:128 + q])],
            SamplingParams(
                max_tokens=1,
                temperature=0,
                logprobs=5
            ),
            use_tqdm=False
        )[0]

        tok = o.outputs[0].token_ids[0]
        logprob = o.outputs[0].logprobs[0][tok].logprob
        out[q] = (tok, logprob)

    return out

if __name__ == "__main__":
    print("=" * 80)
    print("Reproducing FA3 split-KV bug #62626")
    print("=" * 80)

    print("\n1. FULL mode (BUGGY - should show divergence at q=2):")
    full_result = run(compilation_config={"cudagraph_mode": "FULL"})
    print(f"   Results: {full_result}")

    print("\n2. FULL_AND_PIECEWISE mode (CORRECT - baseline):")
    piecewise_result = run(compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"})
    print(f"   Results: {piecewise_result}")

    print("\n3. FULL mode with split-KV disabled (WORKAROUND):")
    workaround_result = run(
        compilation_config={"cudagraph_mode": "FULL"},
        attention_config={"flash_attn_max_num_splits_for_cuda_graph": 1}
    )
    print(f"   Results: {workaround_result}")

    # Compare q=2 across modes
    print("\n" + "=" * 80)
    print("COMPARISON AT q=2 (THE BUG):")
    print("=" * 80)
    full_q2_tok, full_q2_logp = full_result[2]
    piecewise_q2_tok, piecewise_q2_logp = piecewise_result[2]

    print(f"FULL mode:          token={full_q2_tok}, logprob={full_q2_logp:.6f}")
    print(f"FULL_AND_PIECEWISE: token={piecewise_q2_tok}, logprob={piecewise_q2_logp:.6f}")

    if full_q2_tok != piecewise_q2_tok:
        print(f"\n❌ BUG CONFIRMED: Different tokens! ({full_q2_tok} vs {piecewise_q2_tok})")
    elif abs(full_q2_logp - piecewise_q2_logp) > 0.01:
        print(f"\n⚠️  BUG LIKELY: Logprobs differ by {abs(full_q2_logp - piecewise_q2_logp):.6f} nats")
    else:
        print(f"\n✅ No bug detected (difference: {abs(full_q2_logp - piecewise_q2_logp):.6f} nats)")
