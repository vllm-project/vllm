"""
Example 14: Prompt Templates and Formatting

Shows how to use prompt templates for consistent formatting.

Usage:
    python 14_prompt_templates.py
"""

from typing import List, Dict
from vllm import LLM, SamplingParams


class PromptTemplate:
    """Simple prompt template class."""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        return self.template.format(**kwargs)


# Define templates
INSTRUCTION_TEMPLATE = """Instruction: {instruction}
Input: {input}
Output:"""

CHAT_TEMPLATE = """User: {user_message}
Assistant:"""

QA_TEMPLATE = """Question: {question}
Answer:"""


def main():
    """Demo prompt templates."""
    print("=== Prompt Templates Demo ===\n")

    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

    # Example 1: Instruction template
    instruction_prompt = PromptTemplate(INSTRUCTION_TEMPLATE)
    prompt1 = instruction_prompt.format(
        instruction="Summarize the text",
        input="AI is transforming technology"
    )

    # Example 2: Chat template
    chat_prompt = PromptTemplate(CHAT_TEMPLATE)
    prompt2 = chat_prompt.format(
        user_message="What is machine learning?"
    )

    # Example 3: QA template
    qa_prompt = PromptTemplate(QA_TEMPLATE)
    prompt3 = qa_prompt.format(
        question="What is vLLM?"
    )

    prompts = [prompt1, prompt2, prompt3]

    print("Generating with templates...\n")
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs, 1):
        print(f"Template {i}:")
        print(output.prompt)
        print(f"Output: {output.outputs[0].text}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
