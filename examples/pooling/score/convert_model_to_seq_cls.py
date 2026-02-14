# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

"""
Script to convert Large Language Models (LLMs) to Sequence Classification models.
This is particularly useful for converting reranker models that use next-token
prediction to a sequence classification format for compatibility with standard
classification and rerank pipelines.

Usage examples:
- For BAAI/bge-reranker-v2-gemma:
  python convert_model_to_seq_cls.py --model_name BAAI/bge-reranker-v2-gemma \
    --classifier_from_tokens '["Yes"]' --method no_post_processing \
    --path ./bge-reranker-v2-gemma-seq-cls

- For mxbai-rerank-v2:
  python convert_model_to_seq_cls.py --model_name mixedbread-ai/mxbai-rerank-base-v2 \
    --classifier_from_tokens '["0", "1"]' --method from_2_way_softmax \
    --path ./mxbai-rerank-base-v2-seq-cls

- For Qwen3-Reranker:
  python convert_model_to_seq_cls.py --model_name Qwen/Qwen3-Reranker-0.6B \
    --classifier_from_tokens '["no", "yes"]' --method from_2_way_softmax \
    --path ./Qwen3-Reranker-0.6B-seq-cls

Note: For BAAI/bge-reranker-v2-gemma, "Yes" and "yes" are different tokens.
"""

import argparse
import json

import torch
import transformers


def from_2_way_softmax(causal_lm, seq_cls_model, tokenizer, tokens, device):
    """
    This method extracts the difference between weights for 'true' and 'false' tokens
    from the language model head to create a single classification weight vector.

    Args:
        causal_lm: The original causal language model
        seq_cls_model: The target sequence classification model
        tokenizer: Model tokenizer
        tokens: List of two tokens representing [false_token, true_token]
        device: Target device (cpu/cuda)

    Reference: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    """
    assert len(tokens) == 2, (
        "Method requires exactly two tokens for binary classification"
    )

    # Get the language model head weights (vocabulary_size x hidden_size)
    lm_head_weights = causal_lm.lm_head.weight

    # Convert token strings to their corresponding token IDs
    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])

    # Compute the classification weight as the difference between true and false token weights
    # This follows the approach in: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    score_weight = lm_head_weights[true_id].to(device).to(
        torch.float32
    ) - lm_head_weights[false_id].to(device).to(torch.float32)

    # Copy the computed weights to the sequence classification model
    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight.unsqueeze(0))
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()


def no_post_processing(causal_lm, seq_cls_model, tokenizer, tokens, device):
    """
    Directly use token weights from the language model head for classification.

    This method maps each classification label directly to a corresponding token
    in the vocabulary without additional transformation.

    Args:
        causal_lm: The original causal language model
        seq_cls_model: The target sequence classification model
        tokenizer: Model tokenizer
        tokens: List of tokens representing class labels
        device: Target device (cpu/cuda)
    """
    # Get the language model head weights (vocabulary_size x hidden_size)
    lm_head_weights = causal_lm.lm_head.weight

    # Convert all tokens to their corresponding token IDs
    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]

    # Extract weights for the specific tokens (num_tokens x hidden_size)
    score_weight = lm_head_weights[token_ids].to(device)

    # Copy the weights to the sequence classification model
    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight)
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()


method_map = {
    function.__name__: function for function in [from_2_way_softmax, no_post_processing]
}


def converting(
    model_name, classifier_from_tokens, path, method, use_sep_token=False, device="cpu"
):
    """
    Main conversion function to transform a CausalLM model to SequenceClassification.

    Args:
        model_name: Name or path of the pretrained model
        classifier_from_tokens: List of tokens used for classification
        path: Output path to save the converted model
        method: Conversion method ('from_2_way_softmax' or 'no_post_processing')
        use_sep_token: Whether to use separating token in the sequence classification model
        device: Device to load the model on ('cpu' or 'cuda')
    """
    assert method in method_map, f"Unknown method: {method}"

    # Determine number of labels based on conversion method
    if method == "from_2_way_softmax":
        assert len(classifier_from_tokens) == 2
        num_labels = 1
    else:
        num_labels = len(classifier_from_tokens)

    # Load tokenizer and original causal language model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    causal_lm = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device
    )

    # Load an empty sequence classification model with the same architecture
    seq_cls_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        device_map=device,
    )

    # Apply the selected conversion method to transfer weights
    method_map[method](
        causal_lm, seq_cls_model, tokenizer, classifier_from_tokens, device
    )

    # Configure separating token settings
    # Note: `llm as reranker` defaults to not using separating token.
    seq_cls_model.config.use_sep_token = use_sep_token
    seq_cls_model.config.sep_token_id = tokenizer.sep_token_id

    # Save the converted model and tokenizer
    seq_cls_model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Converting *ForCausalLM models to "
        "*ForSequenceClassification models."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-reranker-v2-gemma",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--classifier_from_tokens",
        type=str,
        default='["Yes"]',
        help="JSON string of tokens used for classification labels",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="no_post_processing",
        help="Conversion method to use",
    )
    parser.add_argument(
        "--use-sep-token",
        action="store_true",
        help="Enable separating token in the sequence classification model",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./bge-reranker-v2-gemma-seq-cls",
        help="Output directory to save the converted model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    converting(
        model_name=args.model_name,
        classifier_from_tokens=json.loads(args.classifier_from_tokens),
        method=args.method,
        use_sep_token=args.use_sep_token,
        path=args.path,
    )
