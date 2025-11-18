"""
Example 03: Comprehensive Sampling Parameters Guide

Demonstrates all key sampling parameters and their effects on generation.

Usage:
    python 03_sampling_params_guide.py
"""

from vllm import LLM, SamplingParams


def demonstrate_sampling_params():
    """Show different sampling parameter configurations."""
    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
    prompt = "The future of artificial intelligence"

    # Configuration 1: Greedy Decoding (deterministic)
    greedy = SamplingParams(temperature=0.0, max_tokens=50)

    # Configuration 2: Creative Generation (high temperature)
    creative = SamplingParams(temperature=1.5, max_tokens=50)

    # Configuration 3: Balanced (recommended for most cases)
    balanced = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_tokens=50
    )

    # Configuration 4: Beam Search (best quality, slower)
    beam = SamplingParams(
        best_of=4,
        use_beam_search=True,
        temperature=0.0,
        max_tokens=50
    )

    # Configuration 5: With repetition penalty
    no_repeat = SamplingParams(
        temperature=0.8,
        repetition_penalty=1.2,
        max_tokens=50
    )

    configs = [
        ("Greedy", greedy),
        ("Creative", creative),
        ("Balanced", balanced),
        ("Beam Search", beam),
        ("No Repetition", no_repeat),
    ]

    print("Prompt:", prompt)
    print("=" * 80)

    for name, params in configs:
        output = llm.generate([prompt], params)[0]
        print(f"\n{name}:")
        print(f"  {output.outputs[0].text}")


if __name__ == "__main__":
    demonstrate_sampling_params()
