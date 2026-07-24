# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC

from vllm.config.utils import config
from vllm.utils.import_utils import resolve_obj_by_qualname


@config
class EncoderCacheManagerConfig:
    encoder_cache_manager_cls: str | None = None
    """The qualified class name of the custom encoder cache manager.
        """

    def get_encoder_cache_manager_obj(self):
        cls_path = self.encoder_cache_manager_cls
        if cls_path is None:
            return None
        return resolve_obj_by_qualname(cls_path)


class EncoderCacheManagerMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate between the
    Scheduler EncoderCacheManager and Worker EncoderCacheManager.
    """

    pass
