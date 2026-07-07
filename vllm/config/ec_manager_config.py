# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.utils import config
from abc import ABC, abstractmethod

@config
class EncoderCacheManagerConfig:
    encoder_cache_manager_cls: str | None = None

class EncoderCacheManagerMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate between the
    Scheduler EncoderCacheManager and Worker EncoderCacheManager.
    """

    pass
