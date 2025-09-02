# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import sys
import typing
from logging import config as logging_config

from vllm.kernels import _alias_modules

if typing.TYPE_CHECKING:

    def config() -> None:
        pass


def test_alias_modules():
    _alias_modules(logging, __name__)
    from tests.test_kernels_alias_modules import config
    assert config == logging_config
    current_module = sys.modules[__name__]
    assert current_module.config == logging_config


def test_no_error_when_duplicating_alias_modules():
    _alias_modules(logging, __name__)
    _alias_modules(logging, __name__)
    from tests.test_kernels_alias_modules import config
    assert config == logging_config
    current_module = sys.modules[__name__]
    assert current_module.config == logging_config
