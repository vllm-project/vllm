from typing import Optional

from vllm.config import TokenizerPoolConfig
from vllm.engine.ray_utils import ray
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (
    BaseTokenizerGroup)
from vllm.transformers_utils.tokenizer_group.tokenizer_group import (
    TokenizerGroup)

if ray:
    from vllm.transformers_utils.tokenizer_group.ray_tokenizer_group import (
        RayTokenizerGroupPool)
else:
    RayTokenizerGroupPool = None


def get_tokenizer_group(tokenizer_pool_config: Optional[TokenizerPoolConfig],
                        **init_kwargs) -> BaseTokenizerGroup:
    if tokenizer_pool_config is None:
        return TokenizerGroup(**init_kwargs)
    if tokenizer_pool_config.pool_type == "ray":
        if RayTokenizerGroupPool is None:
            raise ImportError(
                "RayTokenizerGroupPool is not available. Please install "
                "the ray package to use the Ray tokenizer group pool.")
        return RayTokenizerGroupPool.from_config(tokenizer_pool_config,
                                                 **init_kwargs)
    else:
        raise ValueError(
            f"Unknown pool type: {tokenizer_pool_config.pool_type}")


__all__ = ["get_tokenizer_group", "BaseTokenizerGroup"]
