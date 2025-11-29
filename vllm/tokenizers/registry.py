# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib

from .protocol import TokenizerLike


class TokenizerRegistry:
    # Tokenizer name -> (tokenizer module, tokenizer class)
    REGISTRY: dict[str, tuple[str, str]] = {}

    @staticmethod
    def register(name: str, module: str, class_name: str) -> None:
        TokenizerRegistry.REGISTRY[name] = (module, class_name)

    @staticmethod
    def get_tokenizer(
        tokenizer_name: str,
        *args,
        **kwargs,
    ) -> "TokenizerLike":
        tokenizer_cls = TokenizerRegistry.REGISTRY.get(tokenizer_name)
        if tokenizer_cls is None:
            raise ValueError(f"Tokenizer {tokenizer_name} not found.")

        tokenizer_module = importlib.import_module(tokenizer_cls[0])
        class_ = getattr(tokenizer_module, tokenizer_cls[1])
        return class_.from_pretrained(*args, **kwargs)
