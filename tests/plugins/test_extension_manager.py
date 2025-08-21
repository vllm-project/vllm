# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.plugins.extension_manager import ExtensionManager
import pytest

class BaseA:
    def __init__(self) -> None:
        pass

class BaseB:
    def __init__(self) -> None:
        pass

extension_manager_a = ExtensionManager[BaseA]()
extension_manager_b = ExtensionManager[BaseB]()

@extension_manager_a.register(names=["a1"])
class ChildA1(BaseA):
    def __init__(self) -> None:
        super().__init__()

@extension_manager_a.register(names=["a2", "a2_alias"])
class ChildA2(BaseA):
    def __init__(self) -> None:
        super().__init__()

@extension_manager_a.register(names=["b1"])
class ChildB1(BaseB):
    def __init__(self) -> None:
        super().__init__()

@extension_manager_a.register(names=["b2"])
class ChildB2(BaseB):
    def __init__(self) -> None:
        super().__init__()


def test_extension_manager_can_register():
    a1_obj = extension_manager_a.create("a1")
    a2_obj = extension_manager_a.create("a2")
    a2_alias_obj = extension_manager_a.create("a2")

    assert isinstance(a1_obj, ChildA1)
    assert isinstance(a2_obj, ChildA2)
    assert isinstance(a2_alias_obj, ChildA2)
    
    b1_obj = extension_manager_a.create("b1")
    b2_obj = extension_manager_a.create("b2")

    assert isinstance(b1_obj, ChildB1)
    assert isinstance(b2_obj, ChildB2)
