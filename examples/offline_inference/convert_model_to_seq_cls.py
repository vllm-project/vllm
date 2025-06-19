# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import argparse
import json

import torch
import transformers

# Usage:
# for Qwen3-Reranker
# python convert_model_to_seq_cls.py --model_name Qwen/Qwen3-Reranker-0.6B --classifier_from_tokens '["no", "yes"]' --method from_2_way_softmax --path ./Qwen3-Reranker-0.6B-seq-cls
# for BAAI/bge-reranker-v2-gemma
# python convert_model_to_seq_cls.py --model_name BAAI/bge-reranker-v2-gemma --classifier_from_tokens '["Yes"]' --method no_post_processing --path ./bge-reranker-v2-gemma-seq-cls


def from_2_way_softmax(
    causal_lm, seq_cls_model, tokenizer, classifier_from_tokens, device
):
    # for Qwen3-Reranker
    # Adapted from https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    assert len(classifier_from_tokens) == 2

    lm_head_weights = causal_lm.lm_head.weight

    a = tokenizer.convert_tokens_to_ids(classifier_from_tokens[0])
    b = tokenizer.convert_tokens_to_ids(classifier_from_tokens[1])

    score_weight = lm_head_weights[b].to(torch.float32).to(device).to(
        torch.float32
    ) - lm_head_weights[a].to(device)

    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight.unsqueeze(0))
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()


def no_post_processing(
    causal_lm, seq_cls_model, tokenizer, classifier_from_tokens, device
):
    # for BAAI/bge-reranker-v2-gemma

    lm_head_weights = causal_lm.lm_head.weight
    tokens = [tokenizer.convert_tokens_to_ids(t) for t in classifier_from_tokens]
    score_weight = lm_head_weights[tokens].to(device)

    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight)
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()


method_map = {
    function.__name__: function for function in [from_2_way_softmax, no_post_processing]
}


def converting(
    model_name, classifier_from_tokens, path, method, use_pad_token=False, device="cpu"
):
    assert method in method_map

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    causal_lm = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device
    )

    seq_cls_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True, device_map=device
    )

    method_map[method](
        causal_lm, seq_cls_model, tokenizer, classifier_from_tokens, device
    )

    seq_cls_model.config.pad_token_id = tokenizer.pad_token_id
    seq_cls_model.config.use_pad_token = use_pad_token

    seq_cls_model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Converting *ForCausalLM models to "
        "*ForSequenceClassification models."
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B", help="Model name"
    )
    parser.add_argument(
        "--classifier_from_tokens",
        type=str,
        default='["no", "yes"]',
        help="classifier from tokens",
    )
    parser.add_argument(
        "--use-pad-token", action="store_true", help="Whether to use pad_token"
    )
    parser.add_argument(
        "--method", type=str, default="from_2_way_softmax", help="Converting converting"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./converted_model",
        help="Path to save converted model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    converting(
        model_name=args.model_name,
        classifier_from_tokens=json.loads(args.classifier_from_tokens),
        method=args.method,
        use_pad_token=args.use_pad_token,
        path=args.path,
    )
