# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast

import pytest

from tools.pre_commit.validate_config import validate_ast

_TestConfig1 = '''
@config
class _TestConfig1:
    a: int
    """docstring"""
'''

_TestConfig2 = """
@config
class _TestConfig2:
    a: int = 1
"""

_TestConfig3 = '''
@config
class _TestConfig3:
    a: Union[Literal[1], Literal[2]] = 1
    """docstring"""
'''

_TestConfig4 = '''
@dataclass
@config
class _TestConfig4:
    a: int = 1
    """docstring"""
'''

_TestConfig5 = '''
@config
class _TestConfig5:
    a: Literal[1] | Literal[2] = 1
    """docstring"""
'''

_TestConfig6 = '''
@config
class _TestConfig6:
    a: Literal[1] | Literal[2] | None = 1
    """docstring"""
'''

_TestConfig7 = '''
@config
class _TestConfig7:
    a: typing.Union[typing.Literal[1], typing.Literal[2]] = 1
    """docstring"""
'''

_TestConfig8 = '''
@vllm.config.config
class _TestConfig8:
    a: int
    """docstring"""
'''

_TestConfig9 = '''
@config
@dataclass
class _TestConfig9:
    a: int = 1
    """docstring"""
'''

_TestConfig10 = '''
@config
class _TestConfig10:
    a: Optional[Literal[1] | Literal[2]] = 1
    """docstring"""
'''

_TestConfig11 = '''
@config
class _TestConfig11:
    a: Union[Literal[1], Literal[2] | None] = 1
    """docstring"""
'''

_TestConfig12 = '''
@config
class _TestConfig12:
    a: typing.Optional[typing.Union[Literal[1], Literal[2]]] = 1
    """docstring"""
'''


@pytest.mark.parametrize(
    ("test_config", "expected_error"),
    [
        (_TestConfig1, "must have a default"),
        (_TestConfig2, "must have a docstring"),
        (_TestConfig3, "must use a single Literal"),
        (_TestConfig4, "should be used alone"),
        (_TestConfig5, "must use a single Literal"),
        (_TestConfig6, "must use a single Literal"),
        (_TestConfig7, "must use a single Literal"),
        (_TestConfig8, "must have a default"),
        (_TestConfig9, "should be used alone"),
        (_TestConfig10, "must use a single Literal"),
        (_TestConfig11, "must use a single Literal"),
        (_TestConfig12, "must use a single Literal"),
    ],
)
def test_config(test_config, expected_error):
    tree = ast.parse(test_config)
    with pytest.raises(Exception, match=expected_error):
        validate_ast(tree)


_ValidConfig1 = '''
@config(frozen=True)
class _ValidConfig1:
    a: typing.ClassVar[int]
    b: dataclasses.InitVar[int]
    c: Literal[1, 2] | None = None
    """docstring"""
    d: Literal[1] | None = None
    """docstring"""
    e: Union[Literal[1], int] = 1
    """docstring"""
    f: Optional[Literal[1, 2]] = None
    """docstring"""
    g: typing.Optional[Union[Literal[1], int]] = None
    """docstring"""
'''

_ValidConfig2 = """
@dataclass
class _NotAConfig:
    a: int
"""


@pytest.mark.parametrize("test_config", [_ValidConfig1, _ValidConfig2])
def test_valid_config(test_config):
    validate_ast(ast.parse(test_config))
