# SPDX-License-Identifier: Apache-2.0

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


class TokenizerBase(ABC):

    @property
    @abstractmethod
    def all_special_tokens_extended(self) -> List[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def all_special_tokens(self) -> List[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def all_special_ids(self) -> List[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def pad_token(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_fast(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_token_id(self) -> int:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.vocab_size

    @abstractmethod
    def __call__(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    @abstractmethod
    def get_added_vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    @abstractmethod
    def encode_one(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def encode(self,
               text: str,
               add_special_tokens: Optional[bool] = None) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def apply_chat_template(self,
                            messages: List["ChatCompletionMessageParam"],
                            tools: Optional[List[Dict[str, Any]]] = None,
                            **kwargs) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        raise NotImplementedError()

    @abstractmethod
    def decode(self,
               ids: Union[List[int], int],
               skip_special_tokens: bool = True) -> str:
        raise NotImplementedError()

    @abstractmethod
    def convert_ids_to_tokens(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        raise NotImplementedError()


class TokenizerRegistry:
    # Tokenizer name -> (tokenizer module, tokenizer class)
    REGISTRY: Dict[str, Tuple[str, str]] = {}

    @staticmethod
    def register(name: str, module: str, class_name: str) -> None:
        TokenizerRegistry.REGISTRY[name] = (module, class_name)

    @staticmethod
    def get_tokenizer(
        tokenizer_name: str,
        *args,
        **kwargs,
    ) -> TokenizerBase:
        tokenizer_cls = TokenizerRegistry.REGISTRY.get(tokenizer_name)
        if tokenizer_cls is None:
            raise ValueError(f"Tokenizer {tokenizer_name} not found.")

        tokenizer_module = importlib.import_module(tokenizer_cls[0])
        class_ = getattr(tokenizer_module, tokenizer_cls[1])
        return class_.from_pretrained(*args, **kwargs)
