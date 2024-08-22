import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from huggingface_hub import HfApi, hf_hub_download
# yapf: disable
from mistral_common.tokens.tokenizers.mistral import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import (
    MistralTokenizer as PublicMistralTokenizer)
# yapf: enable
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer)
from mistral_common.tokens.tokenizers.tekken import Tekkenizer

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ConversationMessage


@dataclass
class Encoding:
    input_ids: List[int]
    prompt: Optional[str] = None


class MistralTokenizer:

    def __init__(self, tokenizer: PublicMistralTokenizer) -> None:
        self.mistral = tokenizer
        self.instruct = tokenizer.instruct_tokenizer
        self.tokenizer = tokenizer.instruct_tokenizer.tokenizer

        self.vocab_size = len(self.tokenizer.vocab())

        assert isinstance(self.tokenizer,
                          (Tekkenizer, SentencePieceTokenizer)), type(
                              self.tokenizer)
        self._is_tekken = isinstance(self.tokenizer, Tekkenizer)

        # the following attributes are set to fit VLLM's design
        self.is_fast = True
        self.chat_template = True
        self.all_special_ids: List[Any] = []
        self.all_special_tokens: List[Any] = []
        self.all_special_tokens_extended: List[Any] = []

    @classmethod
    def from_pretrained(cls,
                        repo_id: str,
                        *,
                        revision: Optional[str] = None) -> "MistralTokenizer":
        tokenizer_file = cls._download_mistral_tokenizer_from_hf(
            repo_id, revision)

        mistral_tokenizer = PublicMistralTokenizer.from_file(tokenizer_file)
        return cls(mistral_tokenizer)

    @staticmethod
    def _download_mistral_tokenizer_from_hf(tokenizer_name: str,
                                            revision: Optional[str]) -> str:
        api = HfApi()
        repo_info = api.model_info(tokenizer_name)
        files = [s.rfilename for s in repo_info.siblings]
        pattern = re.compile(r"^tokenizer\.model\.v.*$|^tekken\.json$")

        matched_files = [file for file in files if pattern.match(file)]
        if len(matched_files) > 1:
            raise OSError(
                f"Found {len(matched_files)} files matching the "
                "pattern: {matched_files}. Make sure only one Mistral "
                "tokenizer is present in {tokenizer_name}.")
        elif len(matched_files) == 0:
            raise OSError(f"Found {len(matched_files)} files matching the "
                          "pattern: {matched_files}. Make sure that a Mistral "
                          "tokenizer is present in {tokenizer_name}.")

        tokenizer_file = hf_hub_download(tokenizer_name,
                                         filename=matched_files[0],
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
        return self.tokenizer.encode(prompt, bos=False, eos=False)

    def apply_chat_template(self,
                            conversation: List["ConversationMessage"],
                            tools: Optional[Dict[str, Any]] = None,
                            **kwargs) -> Encoding:
        assert tools is None, "`tools` are not yet supported."

        request = ChatCompletionRequest(
            messages=conversation)  # type: ignore[type-var]
        encoded = self.mistral.encode_chat_completion(request)

        return Encoding(input_ids=encoded.tokens, prompt=encoded.text)

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
