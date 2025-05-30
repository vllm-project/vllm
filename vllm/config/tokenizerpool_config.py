# SPDX-License-Identifier: Apache-2.0

from dataclasses import field

from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config
@dataclass
class TokenizerPoolConfig:
    """This config is deprecated and will be removed in a future release.

    Passing these parameters will have no effect. Please remove them from your
    configurations.
    """

    pool_size: int = 0
    """This parameter is deprecated and will be removed in a future release.
    Passing this parameter will have no effect. Please remove it from your
    configurations."""
    pool_type: str = "ray"
    """This parameter is deprecated and will be removed in a future release.
    Passing this parameter will have no effect. Please remove it from your
    configurations."""
    extra_config: dict = field(default_factory=dict)
    """This parameter is deprecated and will be removed in a future release.
    Passing this parameter will have no effect. Please remove it from your
    configurations."""

    def __post_init__(self) -> None:
        logger.warning_once(
            "TokenizerPoolConfig is deprecated and will be removed in a "
            "future release. Passing this parameter will have no effect. "
            "Please remove it from your configurations.")
