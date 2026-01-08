# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.pooler import PoolingTypeStr

from .base import PoolingMethod
from .token import CLSPool, LastPool, MeanPool
from .tokenwise import AllPool


def get_pooling_method(pooling_type: PoolingTypeStr) -> PoolingMethod:
    if pooling_type == "LAST":
        return LastPool()
    if pooling_type == "ALL":
        return AllPool()
    if pooling_type == "CLS":
        return CLSPool()
    if pooling_type == "MEAN":
        return MeanPool()
    if pooling_type == "STEP":
        raise ValueError(
            "'STEP' pooling is handled by StepPooler "
            "and is not a standalone PoolingMethod."
        )

    raise NotImplementedError(f"Unsupported method: {pooling_type!r}")
