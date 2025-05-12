# SPDX-License-Identifier: Apache-2.0

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import resolve_obj_by_qualname

from .punica_base import PunicaWrapperBase

logger = init_logger(__name__)


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    punica_wrapper_qualname = current_platform.get_punica_wrapper()
    punica_wrapper_cls = resolve_obj_by_qualname(punica_wrapper_qualname)
    punica_wrapper = punica_wrapper_cls(*args, **kwargs)
    assert punica_wrapper is not None, \
        "the punica_wrapper_qualname(" + punica_wrapper_qualname + ") is wrong."
    logger.info_once("Using %s.", punica_wrapper_qualname.rsplit(".", 1)[1])
    return punica_wrapper
