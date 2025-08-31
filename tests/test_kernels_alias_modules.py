# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import sys

from vllm.kernels import _alias_modules


def test_alias_modules():
    _alias_modules(logging, __name__)
    from tests.test_kernels import config
    assert config == logging.config
    current_module = sys.modules[__name__]
    assert current_module.config == logging.config


def test_no_error_when_duplicating_alias_modules():
    _alias_modules(logging, __name__)
    _alias_modules(logging, __name__)
    from tests.test_kernels import config
    assert config == logging.config
    current_module = sys.modules[__name__]
    assert current_module.config == logging.config
