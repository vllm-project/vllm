import os.path
from typing import List

from base_model import BaseModel
from vllm import LLM, SamplingParams
from verification_prompt import PROMPT

# MODEL_ID = 'Llama2/Llama-2-7B-fp16'
MODEL_ID = 'NousResearch/Yarn-Llama-2-7b-64k'
MODEL_DIR = os.path.expanduser(f'~/models/{MODEL_ID}')


class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_id = MODEL_ID
        self.llm = LLM(model=MODEL_DIR,  # Use MODEL_ID here to download the model using HF
                       # tokenizer='hf-internal-testing/llama-tokenizer',
                       tensor_parallel_size=2,
                       swap_space=8,
                       seed=42)

    @property
    def max_context_size(self) -> int:
        return self.llm.llm_engine.get_model_config().get_max_model_len()

    def generate(self, prompt: str, n: int, max_new_tokens: int) -> List[str]:
        params = SamplingParams(n=n, max_tokens=max_new_tokens, temperature=0.5)
        outputs = self.llm.generate([prompt], params, use_tqdm=False)[0].outputs
        return [output.text for output in outputs]

    def count_tokens(self, text: str) -> int:
        return len(self.llm.get_tokenizer().tokenize(text))


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
