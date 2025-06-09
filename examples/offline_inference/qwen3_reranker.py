# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

from vllm import LLM

model_name = "Qwen/Qwen3-Reranker-0.6B"

model = LLM(
    model=model_name,
    task="score",
    hf_overrides={
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    },
    dtype="float32",
)

# Why do we need hf_overrides:
# Qwen3-Reranker is a language model that doing reranker by using the
# logits of "no" and "yes" tokens.
# vllm converts it to Qwen3ForSequenceClassification when loaded for
# better performance.
# - Firstly, we need using `"architectures": ["Qwen3ForSequenceClassification"],`
# to manually route to Qwen3ForSequenceClassification.
# - Then, we will extract the vector corresponding to classifier_from_token
# from lm_head using `"classifier_from_token": ["no", "yes"]`.
# - Third, we will convert these two vectors into one vector.  The use of
# conversion logic is controlled by `using "is_original_qwen3_reranker": True`.

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

instruction = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

queries = [
    "What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

queries = [
    query_template.format(prefix=prefix, instruction=instruction, query=query)
    for query in queries
]
documents = [
    document_template.format(doc=doc, suffix=suffix) for doc in documents
]

outputs = model.score(queries, documents)

print([output.outputs.score for output in outputs])
