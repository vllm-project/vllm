import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import SpecialTokens
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


def maybe_serialize_tool_calls(request: ChatCompletionRequest):
    # SEE: https://github.com/vllm-project/vllm/pull/9951
    # Credits go to: @gcalmettes
    # NOTE: There is currently a bug in pydantic where attributes
    # declared as iterables are replaced in in the instances by
    # pydantic-core ValidatorIterator instance. In particular, this
    # affects tool_calls defined in ChatCompletionAssistantMessageParam
    # model:
    # see:
    #   - https://github.com/pydantic/pydantic/issues/9467
    # As a result, tool_calls from assistant messages are never
    # deserialized in the request object if the tool_calls iterator is
    # not consumed. This affect messages passed to the MistralTokenizer
    # since no chat template is applied and therefore the tools_calls
    # iterator is not directly consumed.
    # Issue is tracked on Pydantic side, with resolution planned for
    # v2.11 release. In the meantime, the official workaround is to
    # consume the iterator so the tool_calls are correctly deserialized
    # in the OpenAI ChatCompletionAssistantMessageParam object
    # https://github.com/pydantic/pydantic/issues/9467#issuecomment-2442097291 # noqa: E501
    # Official Pydantic Issues:
    #   - https://github.com/pydantic/pydantic/issues/9541
    # TODO: remove when pydantic v2.11 is released
    for i, message in enumerate(request.messages):
        if message.get("role") == 'assistant':
            tool_calls_validator = message.get("tool_calls", ().__iter__())
            validated_tool_calls = []
            while True:
                try:
                    tool_call = next(tool_calls_validator)  # type: ignore
                    validated_tool_calls.append(tool_call)
                except StopIteration:
                    break

            request.messages[i]["tool_calls"] = validated_tool_calls


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
        self.is_tekken = isinstance(tokenizer_, Tekkenizer)
        self.is_spm = isinstance(tokenizer_, SentencePieceTokenizer)
        if self.is_tekken:
            # Make sure special tokens will not raise
            tokenizer_.special_token_policy = SpecialTokenPolicy.IGNORE
        elif self.is_spm:
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

    # the following attributes are set to fit VLLM's design and are used
    # by the guided structured output backends.
    @property
    def all_special_tokens_extended(self) -> List[str]:
        # tekken defines its own extended special tokens list
        if hasattr(self.tokenizer, "SPECIAL_TOKENS"):
            special_tokens = self.tokenizer.SPECIAL_TOKENS
        else:
            special_tokens = list(SpecialTokens)
        return [
            s.value if isinstance(s, SpecialTokens) else s
            for s in special_tokens
        ]

    @property
    def all_special_tokens(self) -> List[str]:
        return self.all_special_tokens_extended

    @property
    def all_special_ids(self) -> List[int]:
        return [
            self.all_special_tokens.index(t) for t in self.all_special_tokens
        ]

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
        if self.is_tekken:
            tokens = [
                t for t in tokens
                if (t is SpecialTokens.tool_calls
                    or t not in self.tokenizer._all_special_tokens)
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
            # make sure certain special tokens like Tool calls are
            # not decoded
            special_tokens = {SpecialTokens.tool_calls}
            regular_tokens: List[str] = []
            decoded_list = []

            for token in tokens:
                if token in special_tokens:
                    if regular_tokens:
                        decoded_list.append(
                            self.tokenizer.decode(regular_tokens))
                        regular_tokens = []
                    decoded_list.append(token)
                else:
                    regular_tokens.append(token)

            if regular_tokens:
                decoded_list.append(
                    self.decode(regular_tokens))  # type: ignore

            decoded = ''.join(decoded_list)

        return decoded

    def decode(self,
               ids: Union[List[int], int],
               skip_special_tokens: bool = True) -> str:
        assert (
            skip_special_tokens
        ), "skip_special_tokens=False is not supported for Mistral tokenizers."

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
        ), "skip_special_tokens=False is not supported for Mistral tokenizers."

        assert self.is_tekken or self.is_spm, type(self.tokenizer)

        if self.is_tekken:
            # skip special tokens except tool call
            ids = [
                i for i in ids if i > self.tokenizer.num_special_tokens or i ==
                self.tokenizer.get_control_token(SpecialTokens.tool_calls)
            ]

        tokens = [self.tokenizer.id_to_piece(id) for id in ids]

        if any("�" in t for t in tokens) and self.is_tekken:
            # if a decoded token contains the replacement character, then the
            # token has an incomplete UTF-8 character so we must use bytes
            # See: https://github.com/vllm-project/vllm/pull/8640
            #      https://github.com/vllm-project/vllm/pull/9625
            # if underlying tokenizeir is sentencepiece, we just add "�"
            tokens = [self.tokenizer.id_to_byte_piece(id) for id in ids]

        return tokens
