import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download
from mistral_common.protocol.instruct.request import ChatCompletionRequest
# yapf: disable
from mistral_common.tokens.tokenizers.mistral import (
    MistralTokenizer as PublicMistralTokenizer)
# yapf: enable
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer)
from mistral_common.tokens.tokenizers.tekken import (SpecialTokenPolicy,
                                                     Tekkenizer)

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


@dataclass
class Encoding:
    input_ids: List[int]


def list_local_repo_files(repo_id: str, revision: Optional[str]) -> List[str]:
    repo_cache = os.path.join(
        huggingface_hub.constants.HF_HUB_CACHE,
        huggingface_hub.constants.REPO_ID_SEPARATOR.join(
            ["models", *repo_id.split("/")]))

    if revision is None:
        revision_file = os.path.join(repo_cache, "refs", "main")
        if os.path.isfile(revision_file):
            with open(revision_file) as file:
                revision = file.read()

    if revision:
        revision_dir = os.path.join(repo_cache, "snapshots", revision)
        if os.path.isdir(revision_dir):
            return os.listdir(revision_dir)

    return []


def find_tokenizer_file(files: List[str]):
    file_pattern = re.compile(r"^tokenizer\.model\.v.*$|^tekken\.json$")

    matched_files = [file for file in files if file_pattern.match(file)]
    if len(matched_files) > 1:
        raise OSError(f"Found {len(matched_files)} files matching the "
                      f"pattern: {file_pattern}. Make sure only one Mistral "
                      f"tokenizer is present in {files}.")
    elif len(matched_files) == 0:
        raise OSError(f"Found {len(matched_files)} files matching the "
                      f"pattern: {file_pattern}. Make sure that a Mistral "
                      f"tokenizer is present in {files}.")

    return matched_files[0]


class MistralTokenizer:

    def __init__(self, tokenizer: PublicMistralTokenizer) -> None:
        self.mistral = tokenizer
        self.instruct = tokenizer.instruct_tokenizer

        tokenizer_ = tokenizer.instruct_tokenizer.tokenizer
        if isinstance(tokenizer_, Tekkenizer):
            # Make sure special tokens will not raise
            tokenizer_.special_token_policy = SpecialTokenPolicy.IGNORE

            self._vocab = {
                token: idx
                for idx, token in enumerate(tokenizer_.vocab())
            }
        elif isinstance(tokenizer_, SentencePieceTokenizer):
            self._vocab = {
                token: idx
                for idx, token in enumerate(tokenizer_.vocab())
            }
        else:
            raise TypeError(f"Unsupported tokenizer: {type(tokenizer_)}")

        self.tokenizer = tokenizer_

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
        try:
            hf_api = HfApi()
            files = hf_api.list_repo_files(repo_id=tokenizer_name,
                                           revision=revision)
        except ConnectionError as exc:
            files = list_local_repo_files(repo_id=tokenizer_name,
                                          revision=revision)

            if len(files) == 0:
                raise exc

        filename = find_tokenizer_file(files)

        tokenizer_file = hf_hub_download(tokenizer_name,
                                         filename=filename,
                                         revision=revision)
        return tokenizer_file

    # the following attributes are set to fit VLLM's design
    @property
    def all_special_tokens_extended(self) -> List[str]:
        return []

    @property
    def all_special_tokens(self) -> List[str]:
        return []

    @property
    def all_special_ids(self) -> List[int]:
        return []

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def __len__(self) -> int:
        return self.vocab_size

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

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab

    def get_added_vocab(self) -> Dict[str, int]:
        # Mistral tokenizers have no added vocabulary
        return {}

    def encode(self, prompt: str) -> List[int]:
        # `encode` should only be used for prompt completion
        # it should never be used for chat_completion.
        # For chat completion use `apply_chat_template`
        return self.tokenizer.encode(prompt, bos=True, eos=False)

    def apply_chat_template(self,
                            messages: List["ChatCompletionMessageParam"],
                            tools: Optional[Dict[str, Any]] = None,
                            **kwargs) -> List[int]:

        last_message = cast(Dict[str, Any], messages[-1])
        if last_message["role"] == "assistant":
            last_message["prefix"] = True

        request = ChatCompletionRequest(messages=messages,
                                        tools=tools)  # type: ignore[type-var]
        encoded = self.mistral.encode_chat_completion(request)

        # encode-decode to get clean prompt
        return encoded.tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        if isinstance(self.tokenizer, Tekkenizer):
            tokens = [
                t for t in tokens
                if t not in self.tokenizer._all_special_tokens
            ]

            if any(isinstance(t, bytes) for t in tokens):
                # we need to encode and decode all tokens again
                shift = self.tokenizer.num_special_tokens
                byte_tokens = [
                    t.encode("utf-8") if not isinstance(t, bytes) else t
                    for t in tokens
                ]
                ids = [
                    self.tokenizer._tekken_token2id_nospecial[t] + shift
                    for t in byte_tokens
                ]
                decoded = self.tokenizer.decode(ids)
            else:
                decoded = "".join(tokens)
        else:
            decoded = self.tokenizer.decode(tokens)  # type: ignore[arg-type]

        return decoded

    def decode(self, ids: Union[List[int], int]) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.decode(ids)

    def convert_ids_to_tokens(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        # TODO(Patrick) - potentially allow special tokens to not be skipped
        assert (
            skip_special_tokens
        ), "Skipping special tokens is not supported for Mistral tokenizers."

        assert isinstance(self.tokenizer,
                          (Tekkenizer, SentencePieceTokenizer)), type(
                              self.tokenizer)

        tokens = [self.tokenizer.id_to_piece(id) for id in ids]

        if any(t.strip() == "�" for t in tokens):
            # if any stripped decoded token is undefined
            # because it's invalid unicode then pass bytes
            # See: https://github.com/vllm-project/vllm/pull/8640
            tokens = [self.tokenizer.id_to_byte_piece(id) for id in ids]

        return tokens
