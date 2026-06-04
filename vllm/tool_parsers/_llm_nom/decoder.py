# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa

from typing import Callable, Optional, Tuple

IsPendingAndText = Tuple[bool, str]
CountConsumedTokensFn = Callable[[Optional[IsPendingAndText]], int]
