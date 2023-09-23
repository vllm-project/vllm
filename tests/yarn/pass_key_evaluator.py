import random
from typing import Tuple, Iterable, Optional

from base_model import BaseModel


class PassKeyEvaluator:
    system = ('There is an important pass key hidden inside a lot of irrelevant text. Find this key and memorize it. '
              'I will quiz you about the key.\n')

    garbage = 'The grass is green. The sky is blue. The Sun is yellow. Here we go. There and back again.\n'

    information = 'The pass key is {pass_key}. Remember it. {pass_key} is the pass key.\n'

    quiz_question = 'What is the pass key? The pass key is: '

    def __init__(self, model: BaseModel, seed: int = 42):
        super().__init__()
        self.model = model
        self.rng = random.Random(seed)

    def format_prompt(self, garbage_count: int, key_position: int) -> Tuple[str, str, int]:
        """Generates a text file and inserts an execute line at a random position."""
        assert 0 <= key_position <= garbage_count, f'key_position={key_position}, garbage_count={garbage_count}'

        garbage_prefix = ''.join(self.garbage for _ in range(key_position))
        garbage_suffix = ''.join(self.garbage for _ in range(garbage_count - key_position))

        pass_key = f'%06d' % random.randrange(1000000)
        information = self.information.format(pass_key=pass_key)

        fragments = [
            self.system,
            garbage_prefix,
            information,
            garbage_suffix,
            self.quiz_question
        ]

        return ''.join(fragments), pass_key, self.model.count_tokens(garbage_prefix)

    def evaluate(self, max_tokens: int, resolution: int = 100, n: int = 10) -> Iterable[Tuple[int, int, int]]:
        assert max_tokens > 0
        assert resolution > 1

        garbage_count = max_tokens // self.model.count_tokens(self.garbage)
        while garbage_count and self.model.count_tokens(self.format_prompt(garbage_count, 0)[0]) > max_tokens:
            garbage_count -= 1
        assert garbage_count

        for position in range(resolution):
            key_position = int(round(garbage_count * position / (resolution - 1)))
            prompt, pass_key, prefix_token_count = self.format_prompt(garbage_count, key_position)
            outputs = self.model.generate(prompt, n=n, max_new_tokens=self.model.count_tokens(pass_key) + 1)
            success_count = sum((pass_key in output for output in outputs), 0)
            yield key_position, prefix_token_count, success_count


def evaluate_vllm(model: BaseModel, context_size_limit: Optional[int] = None):
    context_size = model.max_context_size
    if context_size_limit is not None:
        context_size = context_size_limit

    print(f'Model: {model.model_id}')
    print(f'Model context size: {context_size}')

    evaluator = PassKeyEvaluator(model)
    for result in evaluator.evaluate(context_size, 100, 2):
        print(result)


def main():
    # Select the model to test here
    from model_llama2_7b_vllm import Model
    # from model_llama2_7b_yarn import Model
    model = Model()

    # If you run out of VRAM, then pass a smaller context size here

    # Limited to 8k
    evaluate_vllm(model, 8192)

    # Unlimited
    # evaluate_vllm(model)


if __name__ == '__main__':
    main()
