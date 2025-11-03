# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any


class ExtensionManager:
    def __init__(self) -> None:
        self.name2class: dict[str, type] = {}

    def register(self, name: str):
        def wrap(cls_to_register):
            self.name2class[name] = cls_to_register
            return cls_to_register

        return wrap

    def load(self, cls_name: str, *args, **kwargs) -> Any:
        cls = self.name2class.get(cls_name)
        assert cls is not None, f"Extension class {cls_name} not found"
        return cls(*args, **kwargs)
