# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

from vllm import LLM

model_name = "Qwen/Qwen3-Reranker-0.6B"

model = LLM(model=model_name,
            task="score",
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_qwen3_rerank": True,
            })

# Why do we need hf_overrides:
# - **Qwen3ForSequenceClassification**, Qwen3 Embedding & Reranker both
# use the same architecture Qwen3ForCausalLM. We need to manually route
# Reranker to Qwen3ForSequenceClassification.
# - **classifier_from_token**, A more efficient approach is to extract
# token_false_id = 2152 and token_true_id = 9693 into a 2-class
# classification task rather than the current 151669-class classification task.
# - **is_qwen3_rerank**, We need to convert the 2-way classifier into a
# 1-way head classifier. This way, it will be completely consistent with
# the Qwen3ForSequenceClassification format.

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

instruction = 'Given a web search query, retrieve relevant passages that answer the query'

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
