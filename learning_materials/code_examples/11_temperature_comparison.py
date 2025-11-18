"""
Example 11: Temperature Comparison

Visual comparison of different temperature settings.

Usage:
    python 11_temperature_comparison.py
"""

from vllm import LLM, SamplingParams


def compare_temperatures():
    """Compare different temperature values."""
    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
    prompt = "The quick brown fox"

    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]

    print(f"Prompt: '{prompt}'\n")
    print("=" * 80)

    for temp in temperatures:
        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=30
        )

        output = llm.generate([prompt], sampling_params)[0]
        text = output.outputs[0].text

        print(f"\nTemperature: {temp}")
        print(f"Output: {text}")
        print("-" * 80)


if __name__ == "__main__":
    compare_temperatures()
