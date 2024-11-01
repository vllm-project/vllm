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

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

logger = init_logger(__name__)


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

        elif isinstance(tokenizer_, SentencePieceTokenizer):
            pass
        else:
            raise TypeError(f"Unsupported tokenizer: {type(tokenizer_)}")

        self._vocab = tokenizer_.vocab()
        # Convert to a Dict[str, int] to match protocol, but this is a lossy
        # conversion. There may be multiple token ids that decode to the same
        # string due to partial UTF-8 byte sequences being converted to �
        self._vocab_dict = {
            token: idx
            for idx, token in enumerate(self._vocab)
        }
        self.tokenizer = tokenizer_
        self._max_token_id = self.vocab_size - 1

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

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

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
        # NB: the dictionary form of the vocabulary collapses token ids that map
        # to the same string but have different bytes
        return self._vocab_dict

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

                def _token_to_id(t: str):
                    t_bytes = t.encode("utf-8") \
                        if not isinstance(t, bytes) else t
                    try:
                        return shift + \
                            self.tokenizer._tekken_token2id_nospecial[t_bytes]
                    except KeyError:
                        logger.warning(
                            "Failed to convert token %s to id,"
                            " replacing with <unk>", t_bytes)
                        return self.tokenizer.unk_id

                ids = [_token_to_id(t) for t in tokens]
                decoded = self.tokenizer.decode(ids)
            else:
                decoded = "".join(tokens)
        else:
            decoded = self.tokenizer.decode(tokens)  # type: ignore[arg-type]

        return decoded

    def decode(self,
               ids: Union[List[int], int],
               skip_special_tokens: bool = True) -> str:
        assert (
            skip_special_tokens
        ), "Skipping special tokens is not supported for Mistral tokenizers."

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

        if any("�" in t for t in tokens):
            # if a decoded token contains the replacement character, then the
            # token has an incomplete UTF-8 character so we must use bytes
            # See: https://github.com/vllm-project/vllm/pull/8640
            #      https://github.com/vllm-project/vllm/pull/9625
            tokens = [self.tokenizer.id_to_byte_piece(id) for id in ids]

        return tokens
