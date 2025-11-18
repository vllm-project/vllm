"""Lab 09: Tokenizer Integration - Complete Solution"""

from transformers import AutoTokenizer
from typing import List


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_text(tokenizer: AutoTokenizer, text: str) -> List[int]:
    """Tokenize text and return token IDs."""
    return tokenizer.encode(text)


def decode_tokens(tokenizer: AutoTokenizer, token_ids: List[int]) -> str:
    """Decode token IDs back to text."""
    return tokenizer.decode(token_ids)


def add_special_tokens(tokenizer: AutoTokenizer) -> None:
    """Add custom special tokens."""
    special_tokens = {"additional_special_tokens": ["[CUSTOM]"]}
    tokenizer.add_special_tokens(special_tokens)


def main():
    """Main tokenizer demo."""
    print("=== Tokenizer Integration Lab ===\n")

    tokenizer = load_tokenizer("gpt2")

    text = "Hello, world!"
    print(f"Text: {text}")

    tokens = tokenize_text(tokenizer, text)
    print(f"Tokens: {tokens}")

    decoded = decode_tokens(tokenizer, tokens)
    print(f"Decoded: {decoded}")

    add_special_tokens(tokenizer)
    print("\nCustom tokens added successfully!")


if __name__ == "__main__":
    main()
