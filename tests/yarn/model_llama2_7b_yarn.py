import os
from typing import List
from base_model import BaseModel
import transformers
import torch

from verification_prompt import PROMPT

MODEL_ID = 'NousResearch/Yarn-Llama-2-7b-64k'
MODEL_DIR = os.path.expanduser(f'~/models/{MODEL_ID}')


class Model(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_id = MODEL_ID
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=MODEL_DIR,  # Use MODEL_ID here to download the model using HF
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    @property
    def max_context_size(self) -> int:
        return self.pipeline.model.base_model.config.max_position_embeddings

    def generate(self, prompt: str, *, n: int, max_new_tokens: int) -> List[str]:
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            top_p=1.0,
            num_return_sequences=n,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
        )
        return [seq['generated_text'] for seq in sequences]

    def count_tokens(self, text: str) -> int:
        return len(self.pipeline.tokenizer.tokenize(text))


def main():
    model = Model()
    print(f'Maximum context size: {model.max_context_size}')
    print(f'The prompt has {model.count_tokens(PROMPT)} tokens:')
    print(PROMPT)
    print()
    for output in model.generate(PROMPT, n=1, max_new_tokens=50):
        print(f'This output has {model.count_tokens(output)} tokens:')
        print(output)
        print()


if __name__ == '__main__':
    main()
