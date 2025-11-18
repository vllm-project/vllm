"""
Example 08: Custom Tokenizer Usage

Shows how to work with custom tokenizers in vLLM.

Usage:
    python 08_custom_tokenizer.py
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def tokenizer_demo():
    """Demonstrate tokenizer operations."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    text = "Hello, world! This is a tokenization example."

    # Tokenize
    tokens = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Decode
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")

    # Token details
    print("\nToken breakdown:")
    for token_id in tokens[:10]:
        token_str = tokenizer.decode([token_id])
        print(f"  {token_id} -> '{token_str}'")


def main():
    """Run tokenizer demo."""
    print("=== Custom Tokenizer Demo ===\n")
    tokenizer_demo()


if __name__ == "__main__":
    main()
