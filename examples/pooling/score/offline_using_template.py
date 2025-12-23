# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
from pathlib import Path

from vllm import LLM

model_name = "nvidia/llama-nemotron-rerank-1b-v2"

# Path to template file
template_path = Path(__file__).parent / "template" / "nemotron-rerank.jinja"
chat_template = template_path.read_text()

llm = LLM(model=model_name, runner="pooling", trust_remote_code=True)

query = "how much protein should a female eat?"
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    "Calorie intake should not fall below 1,200 a day in women or 1,500 a day in men, except under the supervision of a health professional.",
]

outputs = llm.score(query, documents, chat_template=chat_template)

print("-" * 30)
print([output.outputs.score for output in outputs])
print("-" * 30)
