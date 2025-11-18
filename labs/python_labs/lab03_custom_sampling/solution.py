"""
Lab 03: Custom Sampling Strategies - Complete Solution
"""

from typing import List, Dict, Any
from vllm import LLM, SamplingParams


def create_greedy_params() -> SamplingParams:
    """Create sampling params for greedy decoding."""
    return SamplingParams(temperature=0.0, max_tokens=100)


def create_temperature_params(temperature: float) -> SamplingParams:
    """Create sampling params with specific temperature."""
    return SamplingParams(temperature=temperature, max_tokens=100)


def create_topk_params(k: int = 50) -> SamplingParams:
    """Create sampling params for top-k sampling."""
    return SamplingParams(temperature=1.0, top_k=k, max_tokens=100)


def create_topp_params(p: float = 0.95) -> SamplingParams:
    """Create sampling params for nucleus (top-p) sampling."""
    return SamplingParams(temperature=1.0, top_p=p, max_tokens=100)


def create_beam_search_params(n: int = 4) -> SamplingParams:
    """Create sampling params for beam search."""
    return SamplingParams(
        best_of=n,
        use_beam_search=True,
        temperature=0.0,
        max_tokens=100
    )


def compare_sampling_strategies(llm: LLM, prompt: str) -> Dict[str, str]:
    """Compare different sampling strategies on the same prompt."""
    results = {}

    # Greedy
    greedy_params = create_greedy_params()
    output = llm.generate([prompt], greedy_params)[0]
    results["Greedy"] = output.outputs[0].text

    # Low temperature
    low_temp = create_temperature_params(0.3)
    output = llm.generate([prompt], low_temp)[0]
    results["Low Temperature (0.3)"] = output.outputs[0].text

    # Medium temperature
    med_temp = create_temperature_params(0.8)
    output = llm.generate([prompt], med_temp)[0]
    results["Medium Temperature (0.8)"] = output.outputs[0].text

    # High temperature
    high_temp = create_temperature_params(1.5)
    output = llm.generate([prompt], high_temp)[0]
    results["High Temperature (1.5)"] = output.outputs[0].text

    # Top-k
    topk_params = create_topk_params(k=10)
    output = llm.generate([prompt], topk_params)[0]
    results["Top-K (k=10)"] = output.outputs[0].text

    # Top-p
    topp_params = create_topp_params(p=0.9)
    output = llm.generate([prompt], topp_params)[0]
    results["Top-P (p=0.9)"] = output.outputs[0].text

    # Beam search
    beam_params = create_beam_search_params(n=3)
    output = llm.generate([prompt], beam_params)[0]
    results["Beam Search (n=3)"] = output.outputs[0].text

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
