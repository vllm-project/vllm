# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any


def tensors_str_no_data(arg: Any):
    from torch._tensor_str import printoptions

    with printoptions(threshold=1, edgeitems=0):
        return str(arg)
