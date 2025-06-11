# SPDX-License-Identifier: Apache-2.0
import pytest

model_name = "Qwen/Qwen3-Reranker-4B"

text_1 = "What is the capital of France?"
texts_2 = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
]


def vllm_reranker(model_name):
    from vllm import LLM

    model = LLM(model=model_name,
                task="score",
                hf_overrides={
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "classifier_from_token": ["no", "yes"],
                    "is_original_qwen3_reranker": True,
                },
                dtype="float32")

    text_1 = "What is the capital of France?"
    texts_2 = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    outputs = model.score(text_1, texts_2)

    return [output.outputs.score for output in outputs]


def hf_reranker(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    max_length = 8192

    def process_inputs(pairs):
        inputs = tokenizer(pairs,
                           padding=False,
                           truncation='longest_first',
                           return_attention_mask=False,
                           max_length=max_length)
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = ele
        inputs = tokenizer.pad(inputs,
                               padding=True,
                               return_tensors="pt",
                               max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(inputs, **kwargs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    pairs = [(text_1, texts_2[0]), (text_1, texts_2[1])]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)

    return scores


@pytest.mark.parametrize("model_name", [model_name])
def test_model(model_name):
    hf_outputs = hf_reranker(model_name)
    vllm_outputs = vllm_reranker(model_name)

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)
