# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast

import pytest

from tools.validate_config import validate_ast

_TestConfig1 = '''
@config
class _TestConfig1:
    pass
'''

_TestConfig2 = '''
@config
@dataclass
class _TestConfig2:
    a: int
    """docstring"""
'''

_TestConfig3 = '''
@config
@dataclass
class _TestConfig3:
    a: int = 1
'''

_TestConfig4 = '''
@config
@dataclass
class _TestConfig4:
    a: Union[Literal[1], Literal[2]] = 1
    """docstring"""
'''


@pytest.mark.parametrize(("test_config", "expected_error"), [
    (_TestConfig1, "must be a dataclass"),
    (_TestConfig2, "must have a default"),
    (_TestConfig3, "must have a docstring"),
    (_TestConfig4, "must use a single Literal"),
])
def test_config(test_config, expected_error):
    tree = ast.parse(test_config)
    with pytest.raises(Exception, match=expected_error):
        validate_ast(tree)
