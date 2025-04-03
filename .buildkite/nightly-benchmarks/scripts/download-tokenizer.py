# SPDX-License-Identifier: Apache-2.0

import argparse

from transformers import AutoTokenizer


def main(model, cachedir):
    # Load the tokenizer and save it to the specified directory
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.save_pretrained(cachedir)
    print(f"Tokenizer saved to {cachedir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save Hugging Face tokenizer")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Name of the model")
    parser.add_argument("--cachedir",
                        type=str,
                        required=True,
                        help="Directory to save the tokenizer")

    args = parser.parse_args()
    main(args.model, args.cachedir)
