# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

from vllm import LLM

model_name = "Qwen/Qwen3-Reranker-0.6B"

# What is the difference between the official original version and one
# that has been converted into a sequence classification model?
# Qwen3-Reranker is a language model that doing reranker by using the
# logits of "no" and "yes" tokens.
# It needs to computing 151669 tokens logits, making this method extremely
# inefficient, not to mention incompatible with the vllm score API.
# A method for converting the original model into a sequence classification
# model was proposed. See：https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
# Models converted offline using this method can not only be more efficient
# and support the vllm score API, but also make the init parameters more
# concise, for example.
# model = LLM(model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", task="score")

# If you want to load the official original version, the init parameters are
# as follows.

model = LLM(
    model=model_name,
    task="score",
    hf_overrides={
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    },
)

# Why do we need hf_overrides for the official original version:
# vllm converts it to Qwen3ForSequenceClassification when loaded for
# better performance.
# - Firstly, we need using `"architectures": ["Qwen3ForSequenceClassification"],`
# to manually route to Qwen3ForSequenceClassification.
# - Then, we will extract the vector corresponding to classifier_from_token
# from lm_head using `"classifier_from_token": ["no", "yes"]`.
# - Third, we will convert these two vectors into one vector.  The use of
# conversion logic is controlled by `using "is_original_qwen3_reranker": True`.

# Please use the query_template and document_template to format the query and
# document for better reranker results.

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

if __name__ == "__main__":
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

    queries = [
        query_template.format(prefix=prefix, instruction=instruction, query=query)
        for query in queries
    ]
    documents = [document_template.format(doc=doc, suffix=suffix) for doc in documents]

    outputs = model.score(queries, documents)

    print([output.outputs.score for output in outputs])
