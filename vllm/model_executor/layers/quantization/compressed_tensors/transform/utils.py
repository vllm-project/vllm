# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

from compressed_tensors.transform import TransformArgs, TransformScheme

__all__ = ["TransformTuple"]


class TransformTuple(NamedTuple):
    scheme_name: str
    scheme: TransformScheme
    args: TransformArgs
