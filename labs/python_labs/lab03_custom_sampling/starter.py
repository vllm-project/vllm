"""
Lab 03: Custom Sampling Strategies - Starter Code

Implement and experiment with different sampling strategies.
Complete the TODOs to explore various generation approaches.
"""

from typing import List, Dict, Any
from vllm import LLM, SamplingParams


def create_greedy_params() -> SamplingParams:
    """Create sampling params for greedy decoding."""
    # TODO 1: Create SamplingParams with temperature=0 for greedy decoding
    pass


def create_temperature_params(temperature: float) -> SamplingParams:
    """Create sampling params with specific temperature."""
    # TODO 2: Create SamplingParams with the given temperature
    # Set max_tokens=100
    pass


def create_topk_params(k: int = 50) -> SamplingParams:
    """Create sampling params for top-k sampling."""
    # TODO 3: Create SamplingParams with top_k parameter
    # Also set temperature=1.0, max_tokens=100
    pass


def create_topp_params(p: float = 0.95) -> SamplingParams:
    """Create sampling params for nucleus (top-p) sampling."""
    # TODO 4: Create SamplingParams with top_p parameter
    # Also set temperature=1.0, max_tokens=100
    pass


def create_beam_search_params(n: int = 4) -> SamplingParams:
    """Create sampling params for beam search."""
    # TODO 5: Create SamplingParams with:
    # - best_of=n (number of beams)
    # - use_beam_search=True
    # - temperature=0.0
    # - max_tokens=100
    pass


def compare_sampling_strategies(llm: LLM, prompt: str) -> Dict[str, str]:
    """
    Compare different sampling strategies on the same prompt.

    Args:
        llm: LLM instance
        prompt: Input prompt

    Returns:
        Dictionary mapping strategy name to generated text
    """
    results = {}

    # Greedy
    greedy_params = create_greedy_params()
    output = llm.generate([prompt], greedy_params)[0]
    results["greedy"] = output.outputs[0].text

    # TODO: Add more strategies
    # - Low temperature (0.3)
    # - Medium temperature (0.8)
    # - High temperature (1.5)
    # - Top-k (k=10, k=50)
    # - Top-p (p=0.9, p=0.95)
    # - Beam search (n=3)

    return results


def main() -> None:
    """Main function to run sampling experiments."""
    print("=== vLLM Custom Sampling Lab ===\n")

    # Initialize LLM
    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)

    # Test prompt
    prompt = "Once upon a time"

    print(f'Prompt: "{prompt}"\n')

    # Compare strategies
    results = compare_sampling_strategies(llm, prompt)

    # Display results
    for strategy, output in results.items():
        print(f"[{strategy}]")
        print(f"Output: {output}\n")


if __name__ == "__main__":
    main()
