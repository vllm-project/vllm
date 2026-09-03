# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import torch.nn.functional as F

from vllm import LLM

query = "What are the health benefits of green tea?"
documents = [
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
    "Basketball is one of the most popular sports in the United States.",
    "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
]


def main():
    # Initialize model
    llm = LLM(
        model="jinaai/jina-reranker-v3",
        runner="pooling",
    )

    # Generate scores.
    outputs = llm.score(query, documents)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for document, output in zip(documents, outputs):
        score = output.outputs.score
        print(f"Pair: {[query, document]!r} \nScore: {score}")
        print("-" * 60)

    # Generate embeddings.
    # The JinaForRanking model concatenates docs first, then query.
    # Let's stay consistent with this novel design.
    outputs = llm.encode(documents + [query], pooling_task="token_embed")
    embeds = outputs[0].outputs.data.float()

    doc_embeds = embeds[:-1]
    query_embeds = embeds[-1]

    scores = F.cosine_similarity(query_embeds, doc_embeds)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for document, score in zip(documents, scores):
        print(f"Pair: {[query, document]!r} \nScore: {score}")
        print("-" * 60)


if __name__ == "__main__":
    main()
