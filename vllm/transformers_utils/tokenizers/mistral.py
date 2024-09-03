import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
# yapf: disable
from mistral_common.tokens.tokenizers.mistral import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import (
    MistralTokenizer as PublicMistralTokenizer)
# yapf: enable
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer)
from mistral_common.tokens.tokenizers.tekken import (SpecialTokenPolicy,
                                                     Tekkenizer)

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ConversationMessage


@dataclass
class Encoding:
    input_ids: List[int]


def find_tokenizer_file(files: List[str]):
    file_pattern = re.compile(r"^tokenizer\.model\.v.*$|^tekken\.json$")

    matched_files = [file for file in files if file_pattern.match(file)]
    if len(matched_files) > 1:
        raise OSError(f"Found {len(matched_files)} files matching the "
                      "pattern: {matched_files}. Make sure only one Mistral "
                      "tokenizer is present in {tokenizer_name}.")
    elif len(matched_files) == 0:
        raise OSError(f"Found {len(matched_files)} files matching the "
                      "pattern: {matched_files}. Make sure that a Mistral "
                      "tokenizer is present in {tokenizer_name}.")

    return matched_files[0]


class MistralTokenizer:

    def __init__(self, tokenizer: PublicMistralTokenizer) -> None:
        self.mistral = tokenizer
        self.instruct = tokenizer.instruct_tokenizer
        self.tokenizer = tokenizer.instruct_tokenizer.tokenizer

        self.vocab_size = len(self.tokenizer.vocab())

        assert isinstance(self.tokenizer,
                          (Tekkenizer, SentencePieceTokenizer)), type(
                              self.tokenizer)

        if (is_tekken := isinstance(self.tokenizer, Tekkenizer)):
            # Make sure special tokens will not raise
            self.tokenizer.special_token_policy = SpecialTokenPolicy.IGNORE

        self._is_tekken = is_tekken

        # the following attributes are set to fit VLLM's design
        self.is_fast = True
        self.chat_template = True
        self.all_special_ids: List[Any] = []
        self.all_special_tokens: List[Any] = []
        self.all_special_tokens_extended: List[Any] = []

    @classmethod
    def from_pretrained(cls,
                        path_or_repo_id: str,
                        *,
                        revision: Optional[str] = None) -> "MistralTokenizer":
        if not Path(path_or_repo_id).exists():
            assert len(path_or_repo_id.split("/")) == 2, (
                "You have either provided a non-existent path: "
                "{path_or_repo_id} or an invalid HF Hub repo id.")
            tokenizer_file = cls._download_mistral_tokenizer_from_hf(
                path_or_repo_id, revision)
        elif Path(path_or_repo_id).is_dir():
            tokenizer_file_name = find_tokenizer_file(
                os.listdir(path_or_repo_id))
            tokenizer_file = str(Path(path_or_repo_id) / tokenizer_file_name)
        else:
            assert Path(
                path_or_repo_id).is_file(), f"Invalid path: {path_or_repo_id}"

        mistral_tokenizer = PublicMistralTokenizer.from_file(tokenizer_file)
        return cls(mistral_tokenizer)

    @staticmethod
    def _download_mistral_tokenizer_from_hf(tokenizer_name: str,
                                            revision: Optional[str]) -> str:
        api = HfApi()
        repo_info = api.model_info(tokenizer_name)
        files = [s.rfilename for s in repo_info.siblings]

        filename = find_tokenizer_file(files)

        tokenizer_file = hf_hub_download(tokenizer_name,
                                         filename=filename,
                                         revision=revision)
        return tokenizer_file

    def __call__(
        self,
        prompt: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        # Mistral Tokenizers should not add special tokens
        input_ids = self.encode(prompt)

        if truncation:
            input_ids = input_ids[:max_length]

        return Encoding(input_ids=input_ids)

    def get_added_vocab(self) -> List[str]:
        # Mistral tokenizers have no added vocabulary
        return []

    def encode(self, prompt: str) -> List[int]:
        # `encode ` should only be used for prompt completion
        # it should never be used for chat_completion.
        # For chat completion use `apply_chat_template`
        return self.tokenizer.encode(prompt, bos=True, eos=False)

    def apply_chat_template(self,
                            conversation: List["ConversationMessage"],
                            tools: Optional[Dict[str, Any]] = None,
                            **kwargs) -> List[int]:
        assert tools is None, "`tools` are not yet supported."

        request = ChatCompletionRequest(
            messages=conversation)  # type: ignore[type-var]
        encoded = self.mistral.encode_chat_completion(request)

        # encode-decode to get clean prompt
        return encoded.tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        if self._is_tekken:
            return "".join(tokens)
        else:
            return self.tokenizer.decode(tokens)  # type: ignore[arg-type]

    def decode(self, ids: Union[List[int], int]) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.decode(ids)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_id

    def convert_ids_to_tokens(
            self,
            ids: List[int],
            skip_special_tokens: Optional[bool] = True) -> List[str]:
        # TODO(Patrick) - potentially allow special tokens to not be skipped
        assert (
            skip_special_tokens
        ), "Skipping special tokens is not supported for Mistral tokenizers."

        assert isinstance(self.tokenizer,
                          (Tekkenizer, SentencePieceTokenizer)), type(
                              self.tokenizer)

        tokens = [self.tokenizer.id_to_piece(id) for id in ids]
        return tokens

    def __len__(self):
        return self.vocab_size
