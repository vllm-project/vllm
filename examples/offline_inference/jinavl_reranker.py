# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

from vllm import LLM

model_name = "jinaai/jina-reranker-m0"

# refer to https://huggingface.co/jinaai/jina-reranker-m0/blob/main/modeling.py
mm_processor_kwargs = {
    "min_pixels": 3136,
    "max_pixels": 602112,
}

limit_mm_per_prompt = {"image": 4}


def get_model() -> LLM:
    """Initializes and returns the LLM model for JinaVL Reranker."""
    return LLM(
        model=model_name,
        task="score",
        dtype="float16",
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )


def main() -> None:
    query = ["slm markdown"]
    documents = {
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
                },
            },
        ]
    }

    model = get_model()
    outputs = model.score(query, documents)

    print("-" * 30)
    print([output.outputs.score for output in outputs])
    print("-" * 30)


if __name__ == "__main__":
    main()
