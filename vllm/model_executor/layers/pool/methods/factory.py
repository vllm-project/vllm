# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.pooler import PoolingTypeStr

from .base import PoolingMethod
from .token import CLSPool, LastPool, MeanPool


def get_token_pooling_method(pooling_type: PoolingTypeStr | str) -> PoolingMethod:
    if pooling_type == "LAST":
        return LastPool()
    if pooling_type == "CLS":
        return CLSPool()
    if pooling_type == "MEAN":
        return MeanPool()
    if pooling_type == "ALL":
        raise ValueError(
            "'ALL' pooling is handled by AllPooler "
            "and is not a single-token PoolingMethod."
        )
    if pooling_type == "STEP":
        raise ValueError(
            "'STEP' pooling is handled by StepPooler "
            "and is not a single-token PoolingMethod."
        )

    raise NotImplementedError(f"Unsupported token pooling method: {pooling_type!r}")
