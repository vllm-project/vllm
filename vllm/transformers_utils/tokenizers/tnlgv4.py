# Adapted from https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/tokenization_qwen.py
import os
from typing import Collection, List, Optional, Dict, Set, Tuple, Union

import base64

from transformers import PreTrainedTokenizer, AddedToken
import tiktoken


"""
    This tokenizer is almost identical to tiktoken.get_encoding("cl100k_base")
    with a few additional special tokens to support the ChatML format.

    TODO(bapatra): Right now, I do not save the special tokens to the vocab file.
    Maybe in the future, that would be useful? Can add that support later.

"""

def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


SPECIAL_TOKENS = {
    # tiktoken.get_encoding("cl100k_base")._special_tokens
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276,
    # ChatML Related Special Tokens
    "<|im_start|>": 100264,
    "<|im_end|>": 100265,   
}

class TLGv4Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "cl100k_base.tiktoken"
    }

    model_input_names: List[str] = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        errors: str = "replace",
        **kwargs
    ) -> None:
        # PreTrainedTokenizer's init calls _add_tokens, which in turn checks
        # if the token is present in `self.special_tokens``. Hence instantiating it here.
        # The way Qwen gets around this is by checking against SPECIAL_TOKENS
        # But I think it's better to check against the objects own `special_tokens`
        # in case we eventually want to allow the tokenizer to have special tokens.
        self.special_tokens = SPECIAL_TOKENS

        super().__init__(**kwargs)
        self.errors = errors

        base = tiktoken.get_encoding("cl100k_base")
        if vocab_file is None:
            self.mergeable_ranks: Dict[bytes, int] = base._mergeable_ranks
        else:
            self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)

        self.pat_str = base._pat_str
        
        enc = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=self.pat_str,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

        self.decoder: Dict[int, bytes] = {
            v: k for k, v in self.mergeable_ranks.items()
        }
        self.decoder.update({v: k for k, v in self.special_tokens.items()})
        
        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = self.special_tokens["<|im_start|>"]
        self.im_end_id = self.special_tokens["<|im_end|>"]
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        enc = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=self.pat_str,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc
    
    def __len__(self):
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[Union[str, bytes], int]:
        return {**self.mergeable_ranks, **self.special_tokens}
    
    def convert_tokens_to_ids(
        self,
        tokens: Union[bytes, str, List[Union[bytes, str]]]
    ) -> Union[int, List[int]]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        ids: List[int] = []
        for token in tokens:
            ids.append(self.convert_tokens_to_ids(token))
        return ids

    def _add_tokens(
            self,
            new_tokens: Union[List[str], List[AddedToken]],
            special_tokens: bool = False,
    ) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Only special tokens can be added to this tokenizer")
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in self.special_tokens:
                raise ValueError(
                    "For now, we do not support unknown special tokens\n"
                    "In the future, if there is a need for this, we can add special tokens to the tokenizer\n"
                    "starting from rank 100261 - 100263 and then 100266 - 100275.\n"
                    "And finally, we can re-construct the enc object back\n"
                )
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        file_path = os.path.join(save_directory, "cl100k_base.tiktoken")
        with open(file_path, "w") as f:
            for token, rank in self.mergeable_ranks.items():
                line = base64.b64encode(token).decode("utf-8") + " " + str(rank) + "\n"
                f.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs
    ) -> List[Union[bytes, str]]:
        tokens: List[Union[bytes, str]] = []
        for token_id in self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[token_id])
        return tokens

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab
    
    @property
    def eos_token_id(self) -> int:
        return self.eod_id
    
    @property
    def eos_token(self) -> str:
        return self._convert_id_to_token(self.eos_token_id)

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).
        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)
