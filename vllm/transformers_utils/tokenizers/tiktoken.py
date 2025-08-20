# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import AutoTokenizer, PreTrainedTokenizer


class TikTokenTokenizer(PreTrainedTokenizer):
    """Adapter for TikToken tokenizers in vLLM"""

    def __init__(self, tokenizer_name: str, **kwargs):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                        trust_remote_code=True,
                                                        **kwargs)

        self.bos_token = self._tokenizer.bos_token
        self.eos_token = self._tokenizer.eos_token
        self.pad_token = self._tokenizer.pad_token
        self.unk_token = self._tokenizer.unk_token

        if hasattr(self._tokenizer, "special_tokens"):
            map_fn = lambda x: self._tokenizer.special_tokens[x]
        elif hasattr(self._tokenizer, "convert_tokens_to_ids"):
            map_fn = lambda x: self._tokenizer.convert_tokens_to_ids(x)
        else:
            raise ValueError(
                f"Invalid tokenizer type: {type(self._tokenizer)}")

        self.bos_token_id = map_fn(self.bos_token)
        self.eos_token_id = map_fn(self.eos_token)
        self.pad_token_id = map_fn(self.pad_token)
        self.unk_token_id = map_fn(self.unk_token)

        self.vocab_size = len(self._tokenizer.get_vocab())
        self.model_max_length = 131072

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)
