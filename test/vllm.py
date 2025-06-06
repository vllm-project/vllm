# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B")


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)
    return output


max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

instruction = 'Given a web search query, retrieve relevant passages that answer the query'


def process_inputs(query, doc):
    messages = format_instruction(instruction, query, doc)
    messages = prefix + messages + suffix
    return messages


if __name__ == '__main__':
    from vllm import LLM

    model = LLM(model="Qwen/Qwen3-Reranker-4B",
                task="score",
                hf_overrides={
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "classifier_from_token": ["no", "yes"]
                })

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    outputs = model.score(queries, documents, process_inputs=process_inputs)

    print([output.outputs.score for output in outputs])
