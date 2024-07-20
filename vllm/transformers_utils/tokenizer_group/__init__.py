from typing import Optional, Type

from vllm.config import TokenizerPoolConfig
from vllm.executor.ray_utils import ray
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (
    BaseTokenizerGroup)
from vllm.transformers_utils.tokenizer_group.tokenizer_group import (
    TokenizerGroup)

if ray:
    from vllm.transformers_utils.tokenizer_group.ray_tokenizer_group import (
        RayTokenizerGroupPool)
else:
    RayTokenizerGroupPool = None  # type: ignore


def get_tokenizer_group(tokenizer_pool_config: Optional[TokenizerPoolConfig],
                        **init_kwargs) -> BaseTokenizerGroup:
    tokenizer_cls: Type[BaseTokenizerGroup]
    if tokenizer_pool_config is None:
        tokenizer_cls = TokenizerGroup
    elif isinstance(tokenizer_pool_config.pool_type, type) and issubclass(
            tokenizer_pool_config.pool_type, BaseTokenizerGroup):
        tokenizer_cls = tokenizer_pool_config.pool_type
    elif tokenizer_pool_config.pool_type == "ray":
        if RayTokenizerGroupPool is None:
            raise ImportError(
                "RayTokenizerGroupPool is not available. Please install "
                "the ray package to use the Ray tokenizer group pool.")
        tokenizer_cls = RayTokenizerGroupPool
    else:
        raise ValueError(
            f"Unknown pool type: {tokenizer_pool_config.pool_type}")
    return tokenizer_cls.from_config(tokenizer_pool_config, **init_kwargs)


__all__ = ["get_tokenizer_group", "BaseTokenizerGroup"]
