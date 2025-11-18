"""Lab 09: Tokenizer Integration - Starter Code"""

from transformers import AutoTokenizer
from typing import List


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a tokenizer."""
    # TODO 1: Load tokenizer from HuggingFace
    pass


def tokenize_text(tokenizer: AutoTokenizer, text: str) -> List[int]:
    """Tokenize text and return token IDs."""
    # TODO 2: Tokenize and return IDs
    pass


def decode_tokens(tokenizer: AutoTokenizer, token_ids: List[int]) -> str:
    """Decode token IDs back to text."""
    # TODO 3: Decode tokens
    pass


def add_special_tokens(tokenizer: AutoTokenizer) -> None:
    """Add custom special tokens."""
    # TODO 4: Add special tokens
    pass


def main():
    """Main tokenizer demo."""
    print("=== Tokenizer Integration Lab ===\n")

    # TODO 5: Demonstrate tokenization workflow


if __name__ == "__main__":
    main()
