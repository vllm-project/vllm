# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Poolers that do not depend on the pooling type."""

from .poolers import DispatchPooler, DummyPooler

__all__ = ["DispatchPooler", "DummyPooler"]
