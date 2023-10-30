# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

class BaseOperator:
    OPERATOR: Any
    NAME: str
    OPERATOR_CATEGORY: str

    @classmethod
    def is_available(cls) -> bool:
        if cls.OPERATOR is None or cls.OPERATOR.__name__ == "no_such_operator":
            return False
        return True

    @classmethod
    def operator_flop(cls, *inputs) -> int:
        """Calculate number of FLOP given inputs to `OPERATOR`"""
        return -1
    