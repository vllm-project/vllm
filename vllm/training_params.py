# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# TODO(girfan): Check which ones we need and which can be removed
@dataclass
class TrainingParams:
    """Parameters for training requests."""
    is_eval: bool
