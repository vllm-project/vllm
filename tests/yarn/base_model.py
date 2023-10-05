from typing import List


class BaseModel:
    model_id: str = ""

    @property
    def max_context_size(self) -> int:
        raise NotImplementedError()

    def generate(self, prompt: str, n: int, max_new_tokens: int) -> List[str]:
        raise NotImplementedError()

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError()
